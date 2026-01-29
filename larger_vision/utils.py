import os
import sys
import random
import torch
import json
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def initialize_high_sv_weights(weight, singular_value=30.0):
    """
    Initialize a weight matrix with high singular values.
    The `singular_value` argument allows control over the magnitude of the singular values.
    """

    # svd only works for 2D matrices
    if weight.ndimension() != 2:
        return weight

    # Perform SVD: W = U * S * V^T
    U, S, V = torch.svd(weight)

    # Set the singular values to the desired magnitude
    S = torch.ones_like(S) * singular_value

    # Reconstruct the matrix with large singular values
    W_high_sv = U @ torch.diag(S) @ V.t()

    return W_high_sv
from torch.utils.data import Dataset

from torch.utils.data import Dataset

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.targets = [example["label"] for example in hf_dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

def initialize_lenet_with_svd(model, singular_value=10.0):
    for name, param in model.named_parameters():
        # Initialize weight parameters (ignore biases)
        if 'weight' in name:
            if param.ndimension() == 4:  # Convolutional layer
                # Reshape to 2D for SVD and then reshape back to 4D
                weight_reshaped = param.data.view(param.size(0), -1)
                param.data = initialize_high_sv_weights(weight_reshaped, singular_value).view(param.size())
            elif param.ndimension() == 2:  # Fully connected layer
                param.data = initialize_high_sv_weights(param.data, singular_value)

def uniform_dataset(dataset, num_sample_per_class, classes, seed=0):
    '''
    Samples even number of samples for each class
    '''
    random.seed(seed)
    sampled_indices = []
    targets = torch.tensor(dataset.targets)
    print(targets[:10])
    
    for class_label in classes:
        # class_indices = torch.where(dataset.targets == class_label)[0]
        class_indices = torch.where(targets == class_label)[0]

        sampled_indices.extend(random.sample(class_indices.tolist(), num_sample_per_class))
    
    subset = Subset(dataset, sampled_indices)
    return subset

def smooth_labels(subset, num_classes, smoothing):
    '''Given a dataset with 1-hot label, change the label to soft label based on smoothing factor'''
    new_subset = []
    for i in range(len(subset)):
        image, orig_label = subset[i]
        soft_label = np.full((1, num_classes),  smoothing/num_classes)
        soft_label[0][orig_label] += 1-smoothing
        new_subset.append((image, soft_label[0]))
       
    return new_subset

def shuffle_class_label(subset, seed):
        '''
        Map the ground truth of one class to another class
        '''
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # shuffle the order of classes (without replacement)
        random_order_classes = random.sample(self.classes, len(self.classes))
        random_label_map = {k: v for k, v in zip(self.classes, random_order_classes)}
        # change the class label of original mnist to random class
        new_subset = []
        for i in range(len(subset)):
            image, label = subset[i]
            if label in mapping:
                new_subset.append((subset[i][0], random_label_map[subset[i][1]]))
        
        return new_subset

def shuffle_subset_label(subset, seed):
    '''Randomly shuffle the position of labels'''
    torch.manual_seed(seed)
    random.seed(seed)
    labels = [subset[i][1] for i in range(len(subset))]
    shuffled_labels = torch.randperm(len(labels))  # Shuffle the labels, note this returns label indices
    subset_with_shuffled_labels = [(subset[i][0], labels[shuffled_labels[i]]) for i in range(len(subset))]  # Combine images with shuffled labels
    return subset_with_shuffled_labels

class PermutedDataset(Dataset):
    def __init__(self, dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Permute the pixels
        permuted_image = permute_pixels(image, self.permutation)
        return permuted_image, label
    
def permute_pixels(image, permutation):
    np_img = image.numpy()
    flat_img = np_img.flatten()
    permuted_img = flat_img[permutation]
    permuted_img = permuted_img.reshape(np_img.shape)
    
    # Convert back to a torch tensor
    return torch.tensor(permuted_img, dtype=image.dtype)
    
def create_permutation(seed, permute_size):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    permutation = np.random.permutation(permute_size)
    
    return permutation

def imshow(image, title=None):
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def interpolate_images_by_pixel(alpha, perm1, perm2, subset):
    interpolated_subset = []
    
    for i in range(len(subset)):
        image, label = subset[i]
        image1 = image.flatten()[perm1].reshape(image.shape)
        image2 = image.flatten()[perm2].reshape(image.shape)
        
        # interpolate pixel values based on alpha
        interpolated_image = (1 - alpha) * image1 + alpha * image2
        
        torch.clamp(interpolated_image, 0, 255)
        interpolated_image = interpolated_image.to(torch.float32)
        
        interpolated_subset.append((interpolated_image, label))
    
    return interpolated_subset

def visualize_permuted_images(dataset, folder_path='vimg', num_images=5):
    os.makedirs(folder_path, exist_ok=True)
    for i in range(num_images):
        idx = random.randint(1, len(dataset))
        image, label = dataset[idx]
        image = image.squeeze().numpy()  # Remove channel dimension and convert to NumPy
        filename = os.path.join(folder_path, f'image_{i}_label_{label}.png')
        plt.imsave(filename, image, cmap='gray')
