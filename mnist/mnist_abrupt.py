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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import utils
from utils import PermutedMNISTDataset
from utils import Logger
from model import LeNet5, LayerNormMLP

class MNIST_Trainer():
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
        torch.random.manual_seed(self.seed)
        random.seed(self.seed)
        # mnist does not have a uniform distribution
        self.classes = list(range(config['num_classes']))
        self.num_samples_per_class = config['samples_per_class']
        self.num_samples_per_class_test = config['samples_per_class_test']
        
        # default Adam
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.model_name = config['model_name']
        if self.model_name == 'lenet':
            self.model = LeNet5().to(self.device)
        elif self.model_name == 'mlp':
            self.redo = (self.baseline == 'redo')
            self.model = LayerNormMLP(output_dim=config['num_classes']).to(self.device)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        
        checkpoint_path = self.config['from_checkpoint']
        if checkpoint_path:
            ckpt = torch.load(pathlib.PosixPath(checkpoint_path))

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_epoch = ckpt['global_epoch']
            self.task = ckpt['task']
            self.seed = ckpt['seed']
            self.lr = ckpt['lr']

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
        
        self.dataset_name = config['dataset']

        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensors
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize pixel values to range [0, 1]
            ])

            self.train_dataset = datasets.MNIST(root='./data', 
                train=True, 
                transform=self.transform,
                download=True)
        
            self.test_dataset = datasets.MNIST(root='./data', 
                train=False, 
                transform=self.transform,
                download=True)

        elif self.dataset_name == "EMNIST":
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # shape: [1, 28, 28], values in [0, 1]
                transforms.Normalize((0.1307,), (0.3081,))  # match MNIST normalization
            ])

            self.permute_size=28*28
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

        # select subset with uniform labels from mnist 
        self.train_uniform_subset = utils.uniform_random_mnist(self.train_dataset, self.num_samples_per_class, self.classes)
        self.test_uniform_subset = utils.uniform_random_mnist(self.test_dataset, self.num_samples_per_class_test, self.classes)
    
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
                    if self.model_name == 'lenet':
                        images = images.reshape(-1, 1, 28, 28).to(self.device)
                    else:
                        images = images.view(images.size(0), -1).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
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
                # shuffle label from every task
                random_label_train_subset = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
                self.train_loader = DataLoader(random_label_train_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            
            self.total_train_batch = len(self.train_loader)

            self.train_task()

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
                outputs = self.model(images).to(self.device)
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

    parser = argparse.ArgumentParser('MNIST Abrupt Change')

    parser.add_argument('--config', type=str, default="random_mnist_config.json")
    args = parser.parse_args()
    config = utils.read_json(args.config)
    
    trainer = MNIST_Trainer(config=config)
    trainer.train()
    
          
