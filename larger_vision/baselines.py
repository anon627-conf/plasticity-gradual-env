import torch 
import pathlib
import wandb
import utils
import argparse
from train_abrupt import Larger_Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from utils import PermutedDataset
from collections import defaultdict
import hashlib


def make_deterministic(seed: int):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def model_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def optimizer_fingerprint(optim) -> str:
    h = hashlib.sha256()
    sd = optim.state_dict()
    h.update(repr(sd['param_groups']).encode())
    for k, v in sd['state'].items():
        for tk, tv in v.items():
            if torch.is_tensor(tv):
                h.update(tv.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(tv).encode())
    return h.hexdigest()

class SpectralRegularizer:
    def __init__(self, model):
        self.model = model
        self.spectral_vectors = {}

    def get_loss(self, lambda_spec=1e-4, k=2):
        loss = 0.0
        # Iterate over all parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad: 
                continue
            
            if 'bn' in name or 'downsample.1' in name: 
                continue

            if 'bias' in name:
                bias_norm = torch.norm(param, p=2)
                loss += (bias_norm ** k) ** 2
                continue

            # Linear Weight
            if param.ndim >= 2:
                current_sigma = self._compute_spectral_norm(name, param)
                target = 1.0 
                
                loss += (current_sigma**k - target) ** 2

        return lambda_spec * loss

    def _compute_spectral_norm(self, name, param, n_iter=1):
        if param.ndim > 2:
            W = param.view(param.size(0), -1)
        else:
            W = param
            
        rows, cols = W.shape
        
        if name not in self.spectral_vectors:
            v = torch.randn(cols, device=param.device)
            v = F.normalize(v, dim=0, eps=1e-12)
            u = torch.randn(rows, device=param.device)
            u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)
        
        u, v = self.spectral_vectors[name]
        
        # Power Iteration
        with torch.no_grad():
            for _ in range(n_iter):
                # v = W^T * u
                v = torch.matmul(W.t(), u)
                v = F.normalize(v, dim=0, eps=1e-12)
                
                # u = W * v
                u = torch.matmul(W, v)
                u = F.normalize(u, dim=0, eps=1e-12)
            
            # Update cache
            self.spectral_vectors[name] = (u, v)

        # Calculate Sigma (Spectral Norm)
        # sigma = u^T * W * v
        sigma = torch.matmul(u.t(), torch.matmul(W, v))
        return sigma

class SpectralRegularizerPretrained:
    def __init__(self, model):
        self.model = model
        self.spectral_vectors = {}
        self.initial_targets = {}
        self._initialize_targets()

    def _initialize_targets(self):
        print("Initializing Spectral Regularization Targets (Anchoring to Pretrained)...")
        self.model.eval() 
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad: continue
                if 'bn' in name or 'downsample.1' in name: continue
                
                if param.ndim >= 2:
                    sigma = self._compute_spectral_norm(name, param, n_iter=10) 
                    self.initial_targets[name] = sigma.item()
        print("Spectral Targets Initialized.")

    def get_loss(self, lambda_spec=1e-4, k=2):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if 'bn' in name or 'downsample.1' in name: continue
            if 'bias' in name:
                loss += torch.norm(param, p=2) ** k
                continue
            if param.ndim >= 2:
                current_sigma = self._compute_spectral_norm(name, param)
                target = self.initial_targets.get(name, 1.0)
                loss += (current_sigma**k - target**k) ** 2
        return lambda_spec * loss

    def _compute_spectral_norm(self, name, param, n_iter=1):
        if param.ndim > 2:
            W = param.view(param.size(0), -1)
        else:
            W = param
        rows, cols = W.shape
        
        if name not in self.spectral_vectors:
            v = torch.randn(cols, device=param.device)
            v = F.normalize(v, dim=0, eps=1e-12)
            u = torch.randn(rows, device=param.device)
            u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)
        
        u, v = self.spectral_vectors[name]
        with torch.no_grad():
            for _ in range(n_iter):
                v = torch.matmul(W.t(), u)
                v = F.normalize(v, dim=0, eps=1e-12)
                u = torch.matmul(W, v)
                u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)

        sigma = torch.matmul(u.t(), torch.matmul(W, v))
        return sigma

class ReDoHelperResNet:
    def __init__(self, model, optimizer=None, redo_freq=800, threshold=0.1):
        self.model = model
        self.optimizer = optimizer
        self.redo_freq = redo_freq
        self.threshold = threshold
        
        self.step_counter = 0
        self.activations = defaultdict(float)
        self.counts = defaultdict(int)
        self.hooks = []
        self._register_hooks()
        
        print(f"ReDo Initialized: Freq={redo_freq}, Threshold={threshold}")

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _register_hooks(self):
        def get_activation_hook(name):
            def hook(model, input, output):
                with torch.no_grad():
                    post_act = F.relu(output)
                    # Average over spatial dims
                    act_magnitude = post_act.mean(dim=[0, 2, 3])
                    self.activations[name] += act_magnitude
                    self.counts[name] += 1
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Optimization: Track 3x3 convs (features) and conv1
                if module.kernel_size == (3, 3) or name == 'conv1':
                    self.hooks.append(module.register_forward_hook(get_activation_hook(name)))

    def on_step_end(self):
        self.step_counter += 1
        
        if self.step_counter < 2000:
            return

        if self.step_counter % self.redo_freq == 0:
            self.reset_dead_neurons()

    def reset_dead_neurons(self):
        total_reset = 0
        
        for name, conv_module in self.model.named_modules():
            if name in self.activations and self.counts[name] > 0:
                avg_act = self.activations[name] / self.counts[name]
                
                layer_mean = avg_act.mean()
                if layer_mean < 1e-6: layer_mean = 1e-6
                
                dead_mask = avg_act < (self.threshold * layer_mean)
                dead_indices = torch.nonzero(dead_mask).squeeze()
                
                if dead_indices.ndim == 0 and dead_indices.numel() == 1:
                    dead_indices = dead_indices.unsqueeze(0)
                
                # Safety Cap (5%)
                if dead_indices.numel() > 0:
                    n_total = avg_act.numel()
                    max_kill = max(1, int(n_total * 0.05))
                    
                    if dead_indices.numel() > max_kill:
                        # Sort by activity (smallest first) and pick lowest
                        vals = avg_act[dead_indices]
                        sorted_idx = torch.argsort(vals)
                        dead_indices = dead_indices[sorted_idx[:max_kill]]

                    total_reset += dead_indices.numel()
                    
                    # Reset
                    self._reinit_conv_filters(conv_module, dead_indices)
                    bn_module = self._find_associated_bn(name)
                    if bn_module:
                        self._reset_bn_params(bn_module, dead_indices)
                        if self.optimizer:
                            self._reset_optimizer_state(bn_module.weight, dead_indices)
                            self._reset_optimizer_state(bn_module.bias, dead_indices)

                    if self.optimizer:
                        self._reset_optimizer_state(conv_module.weight, dead_indices, dim=0)

                self.activations[name] = 0.0
                self.counts[name] = 0
        
        if total_reset > 0:
            print(f"[Step {self.step_counter}] ReDo: Reset {total_reset} filters.")

    def _reinit_conv_filters(self, module, indices):
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        gain = nn.init.calculate_gain('relu')
        std = gain / (fan_in ** 0.5)
        bound = (3.0 ** 0.5) * std
        
        with torch.no_grad():
            module.weight.data[indices] = torch.empty_like(module.weight.data[indices]).uniform_(-bound, bound)
            if module.bias is not None:
                module.bias.data[indices] = 0.0

    def _find_associated_bn(self, conv_name):
        if "conv" not in conv_name: return None
        bn_name = conv_name.replace("conv", "bn")
        
        # Search in top-level named_modules
        for name, mod in self.model.named_modules():
            if name == bn_name:
                return mod
        return None

    def _reset_bn_params(self, bn_module, indices):
        with torch.no_grad():
            if bn_module.weight is not None: 
                bn_module.weight.data[indices] = 0.0 
            if bn_module.bias is not None: 
                bn_module.bias.data[indices] = 0.0
            
            # Reset running stats
            bn_module.running_mean[indices] = 0.0
            bn_module.running_var[indices] = 1.0

    def _reset_optimizer_state(self, param, indices, dim=0):
        if self.optimizer is None: return
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p is param and p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    if 'exp_avg' not in state: continue
                    # Reset Momentum buffers
                    state['exp_avg'][indices] = 0.0
                    state['exp_avg_sq'][indices] = 0.0


class Larger_Baselines(Larger_Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.initial_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.baseline = config.get('baseline', 'none') # Ensure we have the string

        # Initialize Helpers
        self.spec_reg = None
        self.redo_helper = None

        if self.baseline == "spectral_reg":
            self.spec_reg = SpectralRegularizer(self.model)
        elif self.baseline == "redo":
            if self.dataset_name == 'cifar10':
                self.redo_helper = ReDoHelperResNet(
                    self.model, 
                    optimizer=None, 
                    redo_freq=1000, 
                    threshold=0.1
                )
            else:
                # tiny-imagenet
                self.redo_helper = ReDoHelperResNet(
                    self.model, 
                    optimizer=None, 
                    redo_freq=2000, 
                    threshold=0.1
                )

    def train(self):
        begin_task = self.task
        g = torch.Generator()
        g.manual_seed(self.seed)

        for task in range(begin_task, self.num_tasks):
            print("Starting task ", task)

            if self.baseline == 'shrink_perturb':
                print("Applying Shrink & Perturb before task starts...")
                self.shrink_and_perturb(alpha=0.8, sigma=0.01)

            self.task = task
            if self.exp_name == 'permute_cifar':
                permutation = utils.create_permutation(self.seed)
                random_permute_train_subset = PermutedDataset(self.train_uniform_subset, permutation)
                random_permute_test_subset = PermutedDataset(self.test_uniform_subset, permutation)
                self.train_loader = DataLoader(random_permute_train_subset, self.batch_size, generator=g, num_workers=8, shuffle=True, pin_memory=True)
                self.test_loader = DataLoader(random_permute_test_subset, self.batch_size, generator=g, num_workers=8, shuffle=True, pin_memory=True)
            else:
                random_label_train_subset = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
                self.train_loader = DataLoader(random_label_train_subset, self.batch_size, num_workers=8, generator=g, shuffle=True, pin_memory=True)
            
            self.total_train_batch = len(self.train_loader)
            self.train_task()
            self.task += 1

            if self.exp_name == 'permute_cifar':
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
        for epoch in range(self.epochs_per_task):
            loss_sum = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                model_output = self.model(images)
                
                if config['model_name'] == "microsoft/resnet-18":
                    outputs = model_output.logits.to(self.device)
                else:
                    outputs = model_output.to(self.device)
                
                _, predicted = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                
                # --- BASELINES ---
                if self.baseline == 'l2_reg':
                    loss += self.l2_regularization(lambda_l2=1e-5)
                elif self.baseline == 'l2_towards_init':
                    loss += self.l2_towards_initialization(lambda_init=1e-3)
                elif self.baseline == "spectral_reg":
                    # Use helper for Anchored Spectral Reg
                    if self.dataset_name == 'tiny-imagenet':
                        loss += self.spec_reg.get_loss(lambda_spec=1e-4, k=2)
                    else:
                        loss += self.spec_reg.get_loss(lambda_spec=1e-3, k=2)
               
                
                loss.backward()
                self.optimizer.step()
                
                if self.baseline == 'redo':
                    if self.redo_helper.optimizer is None or self.redo_helper.optimizer != self.optimizer:
                        self.redo_helper.update_optimizer(self.optimizer)
                    
                    self.redo_helper.on_step_end()

                correct += (predicted == labels).sum().item()
                loss_sum += loss.item()

            train_loss = loss_sum / self.total_train_batch
            train_acc = correct / len(self.train_uniform_subset)
            
            if self.log_wandb:
                wandb.log({"train_loss": train_loss, "global_epoch": self.global_epoch})
                wandb.log({"train_acc": train_acc, "global_epoch": self.global_epoch})

            print('global_epoch: {:d}, train_loss: {:.3f}'.format(int(self.global_epoch), train_loss))
            print('global_epoch: {:d}, train_acc: {:.3f}'.format(int(self.global_epoch), train_acc))
            self.global_epoch += 1

    def shrink_and_perturb(self, alpha=0.9, sigma=1e-5):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = alpha * param.data + sigma * torch.randn_like(param)
    
    def l2_regularization(self, lambda_l2):
        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name or len(param.shape) == 1: continue 
            l2_loss += torch.norm(param) ** 2
        return lambda_l2 * l2_loss 
    
    def l2_towards_initialization(self, lambda_init):
        return lambda_init * sum(((p - self.initial_params[n]) ** 2).sum() for n, p in self.model.named_parameters() if p.requires_grad)

if __name__=="__main__":
    parser = argparse.ArgumentParser('Larger Vision Model Baselines')
    parser.add_argument('--config', type=str, default="random_resnet_config.json")
    args = parser.parse_args()
    config = utils.read_json(args.config)
    
    trainer = Larger_Baselines(config=config)
    trainer.train()
