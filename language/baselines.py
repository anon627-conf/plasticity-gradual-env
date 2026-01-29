import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
import os
import transformers
import numpy as np
import nltk
import random
from pathlib import PosixPath
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
import pathlib
import wandb
import datetime
from dateutil import tz
from typing import List
from pathlib import Path
import dataset
import json
from collections import OrderedDict
from trainer import Trainer, read_json, T5ForConditionalGeneration, T5Config, AutoTokenizer, count_model_parameters

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SpectralRegularizer:
    def __init__(self, model):
        self.model = model
        self.spectral_vectors = {} 
        self.initial_targets = {} 
        self._initialize_targets()

    def _initialize_targets(self):
       
        self.model.eval()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.ndim < 2 or not param.requires_grad:
                    continue
                
                if 'layer_norm' in name or 'embed' in name:
                    continue
                
                sigma = self._compute_spectral_norm(name, param, n_iter=20)
                self.initial_targets[name] = sigma.item()

    def get_loss(self, lambda_spec=1e-4, k=2):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.initial_targets:
                continue
            
            # Get current sigma (with gradients)
            current_sigma = self._compute_spectral_norm(name, param, n_iter=1)
            target = self.initial_targets[name]
            
            # Loss = (sigma^k - target^k)^2
            loss += (current_sigma**k - target**k) ** 2
            
        return lambda_spec * loss

    def _compute_spectral_norm(self, name, param, n_iter=1):
        # T5 Linear weights are (Out, In) -> standard matrix
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
                v = torch.matmul(W.t(), u)
                v = F.normalize(v, dim=0, eps=1e-12)
                u = torch.matmul(W, v)
                u = F.normalize(u, dim=0, eps=1e-12)
            self.spectral_vectors[name] = (u, v)

        # Calculate Sigma
        sigma = torch.matmul(u.t(), torch.matmul(W, v))
        return sigma


import torch
import torch.nn.functional as F
from collections import defaultdict

class ReDoHelper:
    def __init__(self, model, optimizer=None, redo_freq=100, threshold=1e-4):
        self.model = model
        self.optimizer = optimizer
        self.redo_freq = redo_freq
        self.threshold = threshold
        
        self.step_counter = 0
        self.activations = defaultdict(float)
        self.counts = defaultdict(int)
        self.hooks = []
        self._register_hooks()
        
        print(f"ReDo initialized: Frequency={self.redo_freq} steps, Threshold={self.threshold}")

    def _register_hooks(self):
        """Attaches hooks to T5 DenseReluDense layers."""
        def get_activation_hook(name):
            def hook(model, input, output):
                with torch.no_grad():
                    post_act = F.relu(output)
                    act_magnitude = post_act.mean(dim=[0, 1])
                    
                    self.activations[name] += act_magnitude
                    self.counts[name] += 1
            return hook

        for name, module in self.model.named_modules():
            if "DenseReluDense" in name or "dense_relu_dense" in name:
                if hasattr(module, 'wi'):
                    self.hooks.append(module.wi.register_forward_hook(get_activation_hook(name + ".wi")))

    def on_step_end(self):
        """
        Must be called by the Trainer at the end of every training step (batch).
        """
        self.step_counter += 1
        
        if self.step_counter > 0 and self.step_counter % self.redo_freq == 0:
            self.reset_dead_neurons()

    def reset_dead_neurons(self):
        print(f"[Step {self.step_counter}] Running ReDo: Checking for dead neurons...")
        total_reset = 0
        
        for name, module in self.model.named_modules():
            key = name + ".wi"
            if key in self.activations and self.counts[key] > 0:
                avg_act = self.activations[key] / self.counts[key]
                
                # identify dead neurons
                dead_mask = avg_act < self.threshold
                dead_indices = torch.nonzero(dead_mask).squeeze()
                
                if dead_indices.numel() > 0:
                    n_dead = dead_indices.numel()
                    total_reset += n_dead
                    
                    stdv = 1. / (module.wi.weight.size(1) ** 0.5)
                    with torch.no_grad():
                        module.wi.weight.data[dead_indices] = torch.empty_like(
                            module.wi.weight.data[dead_indices]
                        ).uniform_(-stdv, stdv)
                    
                    if hasattr(module, 'wo'):
                        with torch.no_grad():
                            module.wo.weight.data[:, dead_indices] = torch.empty_like(
                                module.wo.weight.data[:, dead_indices]
                            ).uniform_(-stdv, stdv)

                    # reset optimizer state
                    if self.optimizer is not None:
                        self._reset_optimizer_state(module.wi, dead_indices)
                        if hasattr(module, 'wo'):
                            self._reset_optimizer_state(module.wo, dead_indices, dim=1)

                self.activations[key] = 0.0
                self.counts[key] = 0
        
        print(f"ReDo Complete: Reset {total_reset} dead neurons.")

    def _reset_optimizer_state(self, layer, dead_indices, dim=0):
        """
        Clears momentum/variance stats for the reset weights so the optimizer 
        treats them as fresh parameters.
        """
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


class BaselineTrainer(Trainer):
    def __init__(self, config, model, tokenizer, baseline_type='none', 
                 reg_strength=1e-4, shrink_factor=0.5, perturb_std=0.01):
        super().__init__(config, model, tokenizer)
        
        self.baseline_type = baseline_type
        self.reg_strength = reg_strength # L2 / Spectral
        self.shrink_factor = shrink_factor
        self.perturb_std = perturb_std
        
        self.spec_reg = None
        self.redo_helper = None
        
        if self.baseline_type == 'spectral_reg':
            self.spec_reg = SpectralRegularizer(self.model)
        elif self.baseline_type == 'redo':
            self.redo_helper = ReDoHelper(
                self.model, 
                optimizer=self.optimizer, 
                redo_freq=100
            )

        # Print shell command
        print('> Command:', ' '.join(sys.argv))
        print()

    def compute_loss(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            labels=batch['target_ids'].to(self.device),
        )
        total_loss = outputs.loss
        
        if self.baseline_type == 'l2_reg':
            l2_loss = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    l2_loss += torch.norm(param, p=2) ** 2
            total_loss += self.reg_strength * l2_loss
            
        elif self.baseline_type == 'spectral_reg':
            spec_loss = self.spec_reg.get_loss(lambda_spec=self.reg_strength)
            total_loss += spec_loss

        elif self.baseline_type == 'redo' and self.model.training:
            if self.redo_helper.optimizer is None and hasattr(self, 'optimizer'):
                 self.redo_helper.update_optimizer(self.optimizer)
            
            self.redo_helper.on_step_end()
            
        return total_loss

    def multi_task_train(self):
        for task in range(self.num_tasks):
            print(f"---- starting task {task} (Baseline: {self.baseline_type}) ----")
            self.task = task
            self.task_seed = task
            
            if 'cipher' in self.exp_name:
                train_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=self.data_size, 
                    seed=self.task_seed
                )
            
                test_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=512,
                    seed=self.task_seed
                )
                train_dataset = dataset.SyntheticDataset(self.tokenizer, train_df)
                test_dataset = dataset.SyntheticDataset(self.tokenizer, test_df)
                
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
                self.test_loader = DataLoader(test_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False, num_workers=4)
            else:
                train_df = dataset.create_random_sentences_synth(sent_len=self.sent_len, word_len=self.word_len, data_size=self.data_size, seed=self.task_seed)
                train_dataset = dataset.SyntheticDataset(tokenizer, train_df)
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
        
            self.train()
            self.evaluation_check()

            if self.baseline_type == 'shrink_perturb':
                self.apply_shrink_perturb()
            elif self.baseline_type == 'redo':
                self.redo_helper.reset_dead_neurons()

    def apply_shrink_perturb(self):
        print(f"Applying Shrink (x{self.shrink_factor}) and Perturb (std={self.perturb_std})...")
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # shrink
                    param.data.mul_(self.shrink_factor)
                    # perturb
                    noise = torch.randn_like(param.data) * self.perturb_std
                    param.data.add_(noise)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('NLP Baselines Script')
    parser.add_argument('--config', type=str, default="train_config.json")
    
    parser.add_argument('--baseline', type=str, default='none', 
                        choices=['none', 'l2_reg', 'spectral_reg', 'shrink_perturb', 'redo'],
                        help="Choose the plasticity baseline to run")
    parser.add_argument('--reg_strength', type=float, default=1e-7, help="Strength for L2/Spectral")
    parser.add_argument('--shrink', type=float, default=0.5, help="Shrink factor")
    parser.add_argument('--perturb', type=float, default=0.01, help="Perturbation std")

    args = parser.parse_args()
    config = read_json(args.config)

    set_seed(int(config['seed']))

    model_name = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    conf = T5Config.from_pretrained("t5-small")
    conf.num_layers = 1
    conf.num_decoder_layers = 1
    conf.d_model = 256
    conf.num_heads = 4
    conf.dropout_rate = 0.0
    conf.layer_norm_eps = 0.0
    
    model = T5ForConditionalGeneration(conf)
    model.resize_token_embeddings(len(tokenizer))

    trainer = BaselineTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        baseline_type=args.baseline,
        reg_strength=args.reg_strength,
        shrink_factor=args.shrink,
        perturb_std=args.perturb
    )

    trainer.multi_task_train()
