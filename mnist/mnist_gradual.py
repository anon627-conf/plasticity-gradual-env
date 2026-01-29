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
from utils import PermutedMNISTDataset
from utils import Logger
from model import LeNet5, LayerNormMLP
import numpy as np

class MNIST_Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_name = config['exp_name']
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
        self.num_classes = len(self.classes)
        self.num_samples_per_class = config['samples_per_class']
        self.num_samples_per_class_test = config['samples_per_class_test']
        
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.model_name = config['model_name']
        if self.model_name == 'lenet':
            self.model = LeNet5().to(self.device)
        elif self.model_name == 'mlp':
            self.model = LayerNormMLP(output_dim=self.num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # smooth settings
        self.k = config['k']
        self.smooth_inc = config['smooth_inc']
        
        
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
            self.log_dir = pathlib.PosixPath(self.log_root_dir, self.exp_name + '_' + self.timestamp)
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
            self.permute_size=28*28
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
                transforms.ToTensor(),
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
    
    def gradual_soft_label(self):
        '''
            Train on 0 smoothing for certain number of epochs
            Then use k * num_epochs as a transition stage where the smooth factor increase 0.9/k * num_epochs after each epoch
        '''
        random_train = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
        self.seed += 1
        
        for task in range(self.num_tasks):
            print("Starting task ", self.task)
            self.task += 1
            
            # learn with 1-hot labels first
            random_class_train_soft_subset = utils.smooth_labels(random_train, num_classes=self.num_classes, smoothing=0)
            self.train_loader = DataLoader(random_class_train_soft_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        
            self.total_train_batch = len(self.train_loader)
            self.train_task()
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
            
            # transition stage to uniform distribution
            transition_epochs = int(self.k * self.epochs_per_task)
            assert (self.k/self.smooth_inc) <= transition_epochs, "Pick larger smooth_inc or larger k"

            for alpha in np.arange(0, 1, self.smooth_inc):

                random_class_train_soft_subset = utils.smooth_labels(random_train, num_classes=self.num_classes, smoothing=alpha)
                self.train_loader = DataLoader(random_class_train_soft_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

                # calculate number of epochs to be trained per smoothing step
                for epoch in range(int(transition_epochs * self.smooth_inc / 2)):
                    self.train_one_epoch_soft_label()

            if self.save_models:
                save_file = "task{}_uniform.pt".format(task)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'task': task,
                    'seed': self.seed,
                    'global_epoch': self.global_epoch,
                    'learning_rate': self.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(self.log_dir, save_file))

            # task change at uniform distribution
            random_train = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
            self.seed += 1

            # transition back to 1-hot
            for alpha in np.arange(1, 0, -self.smooth_inc):
                random_class_train_soft_subset = utils.smooth_labels(random_train, num_classes=self.num_classes, smoothing=alpha)
                self.train_loader = DataLoader(random_class_train_soft_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

                # calculate number of epochs to be trained per smoothing step
                for epoch in range(int(transition_epochs * self.smooth_inc / 2)):
                    self.train_one_epoch_soft_label()
    
    def pixel_value_interpolation(self):

        ''' Interpolate pixel values from old task to new task '''
        
        for task in range(self.num_tasks):
            print("Starting task ", task)
            self.task += 1
            permutation = utils.create_permutation(self.seed, self.permute_size)
            next_permutation = utils.create_permutation(self.seed+1, self.permute_size)
            random_permute_train_subset = PermutedMNISTDataset(self.train_uniform_subset, permutation)
            random_permute_test_subset = PermutedMNISTDataset(self.test_uniform_subset, permutation)
            self.train_loader = DataLoader(random_permute_train_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            self.test_loader = DataLoader(random_permute_test_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            self.seed += 1

            self.total_train_batch = len(self.train_loader)

            self.train_task() # generalizable task will be evaluated every epoch
            print('-----------')
            
            
            # save here for each complete task
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

            # start interpolating
            transition_epochs = int(self.k * self.epochs_per_task)
            assert (1/self.smooth_inc) <= transition_epochs, "Pick larger smooth_inc or larger k"
            for smooth in np.arange(0, 1, self.smooth_inc):
                soft_permute_train = utils.interpolate_images_by_pixel(smooth, permutation, next_permutation, random_permute_train_subset)
                self.train_loader = DataLoader(soft_permute_train, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
                
                for epoch in range(int(transition_epochs * self.smooth_inc)):
                    self.train_one_epoch()

            if self.save_models:
                save_file = "task_uniform{}.pt".format(task)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'task': task,
                    'seed': self.seed,
                    'global_epoch': self.global_epoch,
                    'learning_rate': self.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(self.log_dir, save_file))

    def random_mix_samples(self):
        ''' Mix samples of two random mnist tasks, increase ratio of new task gradually '''
        
        for task in range(self.num_tasks):
            print("Starting task ", task)
            self.task += 1
            random_label_train_subset1 = utils.shuffle_subset_label(self.train_uniform_subset, self.seed)
            random_label_train_subset2 = utils.shuffle_subset_label(self.train_uniform_subset, self.seed+1)
            self.train_loader = DataLoader(random_label_train_subset1, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            
            self.seed += 1

            self.total_train_batch = len(self.train_loader)

            self.train_task()
            print('------------------------')
            
            # save here for each complete task
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

            # start interpolating
            transition_epochs = int(self.k * self.epochs_per_task)
            assert (1/self.smooth_inc) <= transition_epochs, "Pick larger smooth_inc or larger k"
            for smooth in np.arange(0, 1, self.smooth_inc):
                indices = torch.arange(round(len(random_label_train_subset1) * (1-smooth)))
                current_train_subset = Subset(random_label_train_subset1, indices)
                self.train_loader1 = DataLoader(current_train_subset, round(self.batch_size*(1-smooth)), num_workers=8, shuffle=True, pin_memory=True)
                indices = torch.arange(round(len(random_label_train_subset2) * smooth))
                next_train_subset = Subset(random_label_train_subset2, indices)
                self.train_loader2 = DataLoader(next_train_subset, round(self.batch_size*smooth), num_workers=8, shuffle=True, pin_memory=True)
                
                for epoch in range(int(transition_epochs * self.smooth_inc)):
                    self.train_one_epoch_mix_data()

            if self.save_models:
                save_file = "task_uniform{}.pt".format(task)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'task': task,
                    'seed': self.seed,
                    'global_epoch': self.global_epoch,
                    'learning_rate': self.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(self.log_dir, save_file))
    
    def permute_mix_samples(self):

        ''' Mix samples of two permute mnist tasks, increase ratio of new task gradually '''
        
        for task in range(self.num_tasks):
            print("Starting task ", task)
            self.task += 1
            permutation = utils.create_permutation(self.seed, self.permute_size)
            next_permutation = utils.create_permutation(self.seed+1, self.permute_size)
            random_permute_train_subset = PermutedMNISTDataset(self.train_uniform_subset, permutation)
            random_permute_test_subset = PermutedMNISTDataset(self.test_uniform_subset, permutation)
            self.train_loader = DataLoader(random_permute_train_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            self.test_loader = DataLoader(random_permute_test_subset, self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

            next_random_permute_train_subset = PermutedMNISTDataset(self.train_uniform_subset, next_permutation)
            self.seed += 1

            self.total_train_batch = len(self.train_loader)

            self.train_task() # generalizable task will be evaluated every epoch
            print('-----------')
            
            
            # save here for each complete task
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

            # start interpolating
            transition_epochs = int(self.k * self.epochs_per_task)
            assert (1/self.smooth_inc) <= transition_epochs, "Pick larger smooth_inc or larger k"
            for smooth in np.arange(0, 1, self.smooth_inc):
                indices = torch.arange(round(len(random_permute_train_subset) * (1-smooth)))
                current_permute_train_subset = Subset(random_permute_train_subset, indices)
                self.train_loader1 = DataLoader(current_permute_train_subset, round(self.batch_size*(1-smooth)), num_workers=8, shuffle=True, pin_memory=True)
                indices = torch.arange(round(len(next_random_permute_train_subset) * smooth))
                next_permute_train_subset = Subset(next_random_permute_train_subset, indices)
                self.train_loader2 = DataLoader(next_permute_train_subset, round(self.batch_size*smooth), num_workers=8, shuffle=True, pin_memory=True)
                
                for epoch in range(int(transition_epochs * self.smooth_inc)):
                    self.train_one_epoch_mix_data()

            if self.save_models:
                save_file = "task_uniform{}.pt".format(task)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'task': task,
                    'seed': self.seed,
                    'global_epoch': self.global_epoch,
                    'learning_rate': self.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(self.log_dir, save_file))

    def eval(self):
        # test phase evaluates generalization
        self.model.eval()
        test_loss_sum = 0
        num_test_batch = len(self.test_loader)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in self.test_loader:
                images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss_sum += loss.item()
                
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

    def train_one_epoch_mix_data(self):
        # put the model in training mode
        self.model.train()

        loss_sum = 0
        correct = 0
        for batch_idx, ((images1, labels1), (images2, labels2)) in enumerate(zip(self.train_loader1, self.train_loader2)):
            images = torch.cat((images1, images2), dim=0)
            labels = torch.cat((labels1, labels2), dim=0)
    
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

    def train_one_epoch_soft_label(self):
        # put the model in training mode
        self.model.train()

        loss_sum = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            
            images = images.view(images.size(0), -1).to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images).to(self.device)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.max(labels, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            correct += (predicted == true_labels.indices).sum().item()
            loss_sum += loss.item()
            # log at the very last batch
        train_loss = loss_sum / self.total_train_batch
        train_acc = correct / len(self.train_uniform_subset)
        if self.log_wandb:
            wandb.log({"train_loss": train_loss, "global_epoch": self.global_epoch})
            wandb.log({"train_acc": train_acc, "global_epoch": self.global_epoch})

        print('global_epoch: {:d}, train_loss: {:.3f}'.format(int(self.global_epoch), train_loss))
        print('global_epoch: {:d}, train_acc: {:.3f}'.format(int(self.global_epoch), train_acc))
        self.global_epoch += 1

    def train_one_epoch(self):
        # put the model in training mode
        self.model.train()

        loss_sum = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # for cnn
            # images = images.reshape(-1, 1, 28, 28).to(self.device)
            # mlp
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

    def train_task(self):
        for epoch in range(self.epochs_per_task):
            if 'gradual_soft_label' in self.exp_name:
                self.train_one_epoch_soft_label()
            else:
                self.train_one_epoch()
        if 'permute' in self.exp_name or 'mix' in self.exp_name or 'pixel' in self.exp_name:
            self.eval()

    def train(self):
        print(self.exp_name)
        if 'gradual_soft_label' in self.exp_name:
            self.gradual_soft_label()
        elif 'pixel' in self.exp_name or 'permute' in self.exp_name:
            self.pixel_value_interpolation()
        elif 'random_mix' in self.exp_name:
            self.random_mix_samples()
        else:
            self.permute_mix_samples()

        
if __name__=="__main__":

    parser = argparse.ArgumentParser('MNIST smoothing')

    parser.add_argument('--config', type=str, default="random_mnist_config_smooth.json")
    args = parser.parse_args()
    config = utils.read_json(args.config)
    
    trainer = MNIST_Trainer(config=config)
    trainer.train()
    
          
