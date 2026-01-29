import random
import pathlib
import argparse
from tqdm import tqdm
from dateutil import tz
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import wandb
import utils
from utils import PermutedDataset
from utils import Logger
from vit import CustomViT
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

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

import hashlib
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
    # include hyperparams and state tensors
    h.update(repr(sd['param_groups']).encode())
    for k, v in sd['state'].items():
        for tk, tv in v.items():
            if torch.is_tensor(tv):
                h.update(tv.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(tv).encode())
    return h.hexdigest()


class Larger_Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_name = config['exp_name']
        self.baseline = config['baseline']
        tzone = tz.gettz(config['timezone'])
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

        self.log_root_dir = pathlib.PosixPath("saved_experiments/")
        # Set up for logging
        if not self.log_root_dir.exists():
            self.log_root_dir.mkdir()
        
        self.global_epoch = 0
        self.task = 0
        self.seed = config['seed']
        # torch.random.manual_seed(self.seed)
        # random.seed(self.seed)
        make_deterministic(self.seed)
        
        self.dataset_name = config['dataset']
        self.classes = list(range(config['num_classes']))
        self.num_classes = len(self.classes)
        self.num_samples_per_class = config['samples_per_class']
        self.num_samples_per_class_test = config['samples_per_class_test']
        
        # default Adam
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        
        if config['model_name'] == "microsoft/resnet-18":
            config_resnet = AutoConfig.from_pretrained("microsoft/resnet-18", num_labels=self.num_classes)
            self.model = AutoModelForImageClassification.from_config(config_resnet)

            if self.dataset_name == 'tiny-imagenet':
                self.model.resnet.embedder.embedder.convolution = nn.Conv2d(
                    3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                ).to(self.device)
                self.model.resnet.embedder.embedder.pooler = nn.Identity()

            self.model = self.model.to(self.device)
            print(self.model)
        else:
            img_size = 32 if "cifar" in self.dataset_name else 64
        
            self.model = CustomViT(
                num_classes=len(self.classes), 
                img_size=img_size
            ).to(self.device)
            
            print(self.model)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        
        
        checkpoint_path = self.config['from_checkpoint']
        if checkpoint_path:
            ckpt = torch.load(pathlib.PosixPath(checkpoint_path))

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_epoch = ckpt['global_epoch']
            self.task = ckpt['task']+1
            self.seed = ckpt['seed']
            self.lr = ckpt['learning_rate']

            # continue logging for a resumed checkpoint
            self.log_dir = pathlib.PosixPath(checkpoint_path).parent
        else:
            if config['svd_init']:
                utils.initialize_lenet_with_svd(self.model, singular_value=30.0)
            self.log_dir = pathlib.PosixPath(self.log_root_dir, self.exp_name + '_' + self.baseline + '_' + self.timestamp)
            self.log_dir.mkdir()
        
        self.log_txt_path = pathlib.PosixPath(self.log_dir, self.timestamp + '.log')
        self.logger = Logger(self.log_txt_path)
        sys.stdout = self.logger
        sys.stderr = self.logger
        print(self.config)

        self.epochs_per_task = config['epochs_per_task']
        self.num_tasks = config['num_tasks']
        self.save_models = config['save_models']

        self.log_wandb = config['wandb']
        if self.log_wandb:
            wandb.init(
                project="plasticity-project",
                config=self.config,
                name=self.exp_name+self.timestamp
            )

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        
        if self.dataset_name == "cifar10":
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensors
                # transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.2023, 0.1994, 0.2010])  # Normalize pixel values to range [0, 1]
            ])
            self.permute_size=3*32*32
            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        
        elif self.dataset_name == "cifar100":
            # === ADDED: CIFAR-100 SETUP ===
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # CIFAR-100 Mean/Std
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
            ])
            self.permute_size = 3 * 32 * 32
            self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)
        
        elif self.dataset_name == "tiny-imagenet":
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],  
                                     std=[0.2302, 0.2265, 0.2262]),
            ])
            self.permute_size=64*64*3
            tiny_imagenet = load_dataset("zh-plus/tiny-imagenet")
            
            self.train_dataset = utils.HFDatasetWrapper(tiny_imagenet["train"], transform=self.transform)
            self.test_dataset = utils.HFDatasetWrapper(tiny_imagenet["valid"], transform=self.transform)
        
        elif self.dataset_name == "cifar100":
            # === ADDED: CIFAR-100 SETUP ===
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # CIFAR-100 Mean/Std
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
            ])
            self.permute_size = 3 * 32 * 32
            self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)

        
        elif self.dataset_name == "EMNIST":
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                transforms.Resize((224, 224)),                # Resize to match ResNet input size
                transforms.ToTensor(),
                transforms.Normalize([0.1307]*3, [0.3081]*3)  # Normalize pixel values to range [0, 1]
            ])
            self.permute_size=3*224*224
            self.train_dataset = datasets.EMNIST(root='./data', 
                train=True, 
                split='balanced',
                transform=self.transform,
                download=True)
        
            self.test_dataset = datasets.EMNIST(root='./data', 
                train=False, 
                split='balanced',
                transform=self.transform,
                download=True)
        else:
            raise SystemExit("Dataset not defined")


        # select subset with uniform labels from cifar 
        self.train_uniform_subset = utils.uniform_dataset(self.train_dataset, self.num_samples_per_class, self.classes)
        self.test_uniform_subset = utils.uniform_dataset(self.train_dataset, self.num_samples_per_class_test, self.classes)

    def eval(self):
        # test phase
        if 'permute' in self.exp_name:
            self.model.eval()
            test_loss_sum = 0
            num_test_batch = len(self.test_loader)
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in self.test_loader:
                    # conv2d expects to see shape: [batch_size, channels, height, width]
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    if self.config['model_name'] == "microsoft/resnet-18":
                        outputs = outputs.logits.to(self.device)
                    else:
                        outputs = outputs.to(self.device)

                    loss = self.criterion(outputs, labels)
                    test_loss_sum += loss.item()
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                    test_loss = test_loss_sum / num_test_batch
                    test_acc = n_correct / n_samples
                
                if self.log_wandb:
                    wandb.log({'epoch_test/loss': test_loss_sum / num_test_batch, "task": self.task})
                    wandb.log({'epoch_test/acc': n_correct / n_samples, "task": self.task})
                print('task: {:d}, test_loss: {:.3f}'.format(self.task, test_loss))
                print('task: {:d}, test_acc: {:.3f}'.format(self.task, test_acc))
    
    def train(self):
        # print(f"cudnn.deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}")
        # print("Transform:", self.transform)

        # print("INIT model fp:", model_fingerprint(self.model))
        # print("INIT optim fp:", optimizer_fingerprint(self.optimizer))
        self.model.train()

        begin_task = self.task
        for task in range(begin_task, self.num_tasks):
            print("Starting task ", task)
            self.task = task
            if 'permute' in self.exp_name:
                permutation = utils.create_permutation(self.seed, self.permute_size)
                random_permute_train_subset = PermutedDataset(self.train_uniform_subset, permutation)
                random_permute_test_subset = PermutedDataset(self.test_uniform_subset, permutation)
                self.train_loader = DataLoader(random_permute_train_subset, self.batch_size, num_workers=8, generator=self.generator, shuffle=True, pin_memory=True)
                self.test_loader = DataLoader(random_permute_test_subset, self.batch_size, num_workers=8, generator=self.generator, shuffle=True, pin_memory=True)
            else:
                # shuffle label from every task
                random_label_train_subset = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
                self.train_loader = DataLoader(random_label_train_subset, self.batch_size, num_workers=8, generator=self.generator, shuffle=True, pin_memory=True)
            
            self.total_train_batch = len(self.train_loader)
            self.train_task()
            self.task += 1

            if 'permute' in self.exp_name:
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
        self.model.train()
        for epoch in range(self.epochs_per_task):
            loss_sum = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # images = images.reshape(-1, 28*28).to(self.device)
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
                loss.backward()
                self.optimizer.step()

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
        
        
if __name__=="__main__":

    parser = argparse.ArgumentParser('cifar Abrupt Change')

    parser.add_argument('--config', type=str, default="random_resnet_config.json")
    args = parser.parse_args()
    config = utils.read_json(args.config)
    
    trainer = Larger_Trainer(config=config)
    trainer.train()
    
          
