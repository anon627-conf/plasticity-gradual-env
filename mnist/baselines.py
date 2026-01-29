import sys 
import torch 
import pathlib
import wandb
import utils
import argparse
from mnist_abrupt import MNIST_Trainer
from torch.utils.data import DataLoader
from utils import PermutedMNISTDataset
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class SpectralRegularizer:
    def __init__(self, model, anchor_to_init=True):
        self.model = model
        self.spectral_vectors = {}
        self.initial_targets = {}
        if anchor_to_init:
            self._initialize_targets()

    def _initialize_targets(self):
        print("Initializing Spectral Regularization Targets (Anchoring to Initialization)...")
        self.model.eval()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad or param.ndim < 2:
                    continue
                if 'norm' in name or 'bn' in name or 'ln' in name:
                    continue
                
                # Compute exact spectral norm for initialization anchor
                sigma = self._compute_spectral_norm(name, param, n_iter=20)
                self.initial_targets[name] = sigma.item()
        print("Spectral Targets Initialized.")

    def get_loss(self, lambda_spec=0.1, k=2):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'norm' in name or 'bn' in name or 'ln' in name:
                continue

            # Biases: Standard L2 penalty
            if param.ndim < 2 or 'bias' in name:
                loss += torch.norm(param, p=2) ** k
                continue

            # Weights: Spectral penalty
            if param.ndim >= 2:
                current_sigma = self._compute_spectral_norm(name, param, n_iter=1)
                target = self.initial_targets.get(name, 1.0)
                loss += (current_sigma**k - target**k) ** 2

        return lambda_spec * loss

    def _compute_spectral_norm(self, name, param, n_iter=1):
        if param.ndim > 2:
            W = param.view(param.size(0), -1)
        else:
            W = param
            
        rows, cols = W.shape
        device = param.device

        if name not in self.spectral_vectors:
            v = torch.randn(cols, device=device)
            v = F.normalize(v, dim=0, eps=1e-12)
            u = torch.randn(rows, device=device)
            u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)
        
        u, v = self.spectral_vectors[name]

        # Power Iteration
        u = u.detach()
        v = v.detach()

        with torch.no_grad():
            for _ in range(n_iter):
                v = torch.matmul(W.t(), u)
                v = F.normalize(v, dim=0, eps=1e-12)
                u = torch.matmul(W, v)
                u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)

        sigma = torch.matmul(u.t(), torch.matmul(W, v))
        return sigma


class ReDoHelperMLP:
    def __init__(self, model, optimizer, redo_freq=1000, threshold=0.1):
        """
        Args:
            model: Instance of LayerNormMLP
            optimizer: The optimizer (e.g., AdamW)
            redo_freq: How often to check/reset (steps)
            threshold: Relative threshold (fraction of layer mean activation)
        """
        self.model = model
        self.optimizer = optimizer
        self.redo_freq = redo_freq
        self.threshold = threshold
        
        self.step_counter = 0
        self.activations = defaultdict(float)
        self.counts = defaultdict(int)
        self.hooks = []
        self.layer_map = {}
        
        self._register_hooks()
        print(f"ReDo Helper (MLP) initialized: Freq={self.redo_freq}, Threshold={self.threshold}")

    def _register_hooks(self):
        def get_activation_hook(name):
            def hook(model, input, output):
                with torch.no_grad():
                    # Simulate ReLU to see deadness after Norm
                    post_act = F.relu(output)
                    # Average over batch dimension
                    act_magnitude = post_act.mean(dim=0)
                    self.activations[name] += act_magnitude
                    self.counts[name] += 1
            return hook

        for i, norm_layer in enumerate(self.model.norms):
            name = f"layer_{i}"
            self.layer_map[name] = i
            self.hooks.append(norm_layer.register_forward_hook(get_activation_hook(name)))

    def on_step_end(self):
        """Call this after optimizer.step()"""
        self.step_counter += 1
        # Warmup: Skip first 1000 steps
        if self.step_counter > 1000 and self.step_counter % self.redo_freq == 0:
            self.reset_dead_neurons()

    def reset_dead_neurons(self):
        print(f"[Step {self.step_counter}] ReDo Checking...")
        total_reset = 0
        
        for name, layer_idx in self.layer_map.items():
            if self.counts[name] == 0: continue

            avg_act = self.activations[name] / self.counts[name]
            
            # Relative Threshold Logic
            layer_mean = avg_act.mean()
            if layer_mean < 1e-9: layer_mean = 1e-9
            
            relative_thresh_val = self.threshold * layer_mean
            dead_mask = avg_act < relative_thresh_val
            dead_indices = torch.nonzero(dead_mask).squeeze()
            
            if dead_indices.numel() == 0:
                self.activations[name] = 0.0
                self.counts[name] = 0
                continue
                
            if dead_indices.ndim == 0:
                dead_indices = dead_indices.unsqueeze(0)

            # Safety Cap: Limit reset to max 5% of layer size
            max_reset = int(avg_act.numel() * 0.05)
            if dead_indices.numel() > max_reset:
                sub_values = avg_act[dead_indices]
                sorted_sub = torch.argsort(sub_values)
                dead_indices = dead_indices[sorted_sub[:max_reset]]

            n_dead = dead_indices.numel()
            total_reset += n_dead

            with torch.no_grad():
                incoming_layer = self.model.linears[layer_idx]
                fan_in = incoming_layer.weight.size(1)
                stdv = 1. / (fan_in ** 0.5)
                
                incoming_layer.weight.data[dead_indices] = torch.empty_like(
                    incoming_layer.weight.data[dead_indices]
                ).uniform_(-stdv, stdv)
                
                if incoming_layer.bias is not None:
                    incoming_layer.bias.data[dead_indices] = 0.0

                if layer_idx < len(self.model.linears) - 1:
                    outgoing_layer = self.model.linears[layer_idx + 1]
                else:
                    outgoing_layer = self.model.output_layer
                
                outgoing_layer.weight.data[:, dead_indices] = 0.0

            self._reset_optimizer_state(incoming_layer, dead_indices, dim=0)
            self._reset_optimizer_state(outgoing_layer, dead_indices, dim=1)

            self.activations[name] = 0.0
            self.counts[name] = 0
            
        if total_reset > 0:
            print(f"   -> ReDo: Reset {total_reset} neurons total.")

    def _reset_optimizer_state(self, layer, dead_indices, dim=0):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p is layer.weight and p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    if 'exp_avg' in state:
                        if dim == 0:
                            state['exp_avg'][dead_indices] = 0.0
                            state['exp_avg_sq'][dead_indices] = 0.0
                        else:
                            state['exp_avg'][:, dead_indices] = 0.0
                            state['exp_avg_sq'][:, dead_indices] = 0.0

class LayerNormMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=256, num_layers=4):
        super(LayerNormMLP, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.linears.append(nn.Linear(in_d, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x = self.norms[i](x)
            x = torch.relu(x) # Helper hooks trigger here via self.norms
        
        out = self.output_layer(x)
        return out


class MNIST_Baselines(MNIST_Trainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.initial_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.redo_helper = None
        self.spec_reg = None

        if self.baseline == "spectral_reg":
            self.spec_reg = SpectralRegularizer(self.model)

    def train(self):
        # Initialize ReDo Helper ONLY ONCE if needed (requires optimizer to exist)
        if self.baseline == 'redo' and self.redo_helper is None:
            # Assuming self.model is LayerNormMLP instance or compatible
            self.redo_helper = ReDoHelperMLP(self.model, self.optimizer, redo_freq=1000, threshold=0.1)

        begin_task = self.task
        for task in range(begin_task, self.num_tasks):
            print("Starting task ", task)
            self.task = task
            if 'permute' in self.exp_name:
                permutation = utils.create_permutation(self.seed)
                random_permute_train_subset = PermutedMNISTDataset(self.train_uniform_subset, permutation)
                random_permute_test_subset = PermutedMNISTDataset(self.test_uniform_subset, permutation)
                self.train_loader = DataLoader(random_permute_train_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
                self.test_loader = DataLoader(random_permute_test_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            else:
                random_label_train_subset = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
                self.train_loader = DataLoader(random_label_train_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            
            self.total_train_batch = len(self.train_loader)
            self.train_task()
            self.task += 1

            if self.baseline == 'shrink_perturb':
                self.shrink_and_perturb(alpha=0.9, sigma=0.01) # Increased sigma

            if 'permute' in self.exp_name or 'pixel' in self.exp_name:
                self.eval()
            
            if self.save_models:
                save_file = "task{}.pt".format(task)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'task': task,
                    'seed': self.seed,
                    'global_epoch': self.global_epoch,
                    'learning_rate': self.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(self.log_dir, save_file))
            
            self.seed += 1

    def train_task(self):
        if self.baseline == 'redo':
            print(f">> [Baseline] Task {self.task}: ReDO is ENABLED")
            
        for epoch in range(self.epochs_per_task):
            loss_sum = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if self.model_name == 'lenet':
                        images = images.reshape(-1, 1, 28, 28).to(self.device)
                else:
                    images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                model_output = self.model(images)
                
                if config['model_name'] == "microsoft/resnet-18":
                    outputs = model_output.logits.to(self.device)
                else:
                    outputs = model_output.to(self.device)
                
                loss = self.criterion(outputs, labels)

                # --- MITIGATIONS: Loss Terms ---
                if self.baseline == 'l2_reg':
                    loss += self.l2_regularization(lambda_l2=1e-3)
                elif self.baseline == 'l2_towards_init':
                    loss += self.l2_towards_initialization(lambda_init=0.1) # Increased from 1e-3
                elif self.baseline == "spectral_reg":
                    loss += self.spec_reg.get_loss(lambda_spec=0.1, k=2) # Increased from 1e-4
                
                loss.backward()
                self.optimizer.step()

                # --- MITIGATIONS: ReDO (After Step) ---
                if self.baseline == 'redo' and self.redo_helper:
                    self.redo_helper.on_step_end()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                loss_sum += loss.item()

            train_loss = loss_sum / self.total_train_batch
            train_acc = correct / len(self.train_uniform_subset)
            if self.log_wandb:
                wandb.log({"train_loss": train_loss, "global_epoch": self.global_epoch})
                wandb.log({"train_acc": train_acc, "global_epoch": self.global_epoch})

            print('global_epoch: {:d}, train_loss: {:.3f}, train_acc: {:.3f}'.format(
                int(self.global_epoch), train_loss, train_acc))
            self.global_epoch += 1

    def shrink_and_perturb(self, alpha=0.9, sigma=0.01):
        print(f"Apply Shrink & Perturb (alpha={alpha}, sigma={sigma})")
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * sigma
                    param.data = (param.data * alpha) + noise
    
    def l2_regularization(self, lambda_l2):
        l2_norm = sum(torch.norm(p)**2 for p in self.model.parameters() if p.requires_grad)
        return lambda_l2 * l2_norm
    
    def l2_towards_initialization(self, lambda_init=0.1):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.initial_params and param.requires_grad:
                loss += torch.norm(param - self.initial_params[name])**2
        return lambda_init * loss


if __name__=="__main__":
    parser = argparse.ArgumentParser('MNIST Baselines')
    parser.add_argument('--config', type=str, default="random_mnist_config.json")
    args = parser.parse_args()
    print(" ".join(sys.argv))
    config = utils.read_json(args.config)
    
    trainer = MNIST_Baselines(config=config)
    
    trainer.train()
