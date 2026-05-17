#Deep Learning Commom Dataset Sampling Lib ~ Pytorch 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import torchvision
import torchvision.utils as vutils
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import re
import copy
import time
import math
import logging
import os


class SamplerBasedOnIndices(Dataset):
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example,label = self.dataset[self.indices[idx]]

        return example,label

    def load_data(self):
        images = []
        labels = []
        for idx in self.indices:
            example,label = self.dataset[idx]
            images.append(example)
            labels.append(label)
        batch_images = torch.stack(images)
        #batch_labels = torch.tensor(labels)

        return batch_images, labels

class NoisyLabelSampler(Dataset):
    def __init__(self, dataset, indices):

        self.dataset = dataset
        self.indices = indices
        
        self.noisy_labels = {}

    def make_noisy_examples(self,noise_ratio=0.1, num_classes=None):
        dataset = self.dataset
        indices = self.indices
        
        self.noisy_indices = random.sample(indices, int(len(indices) * noise_ratio))
        
        for idx in self.noisy_indices:
            original_label = dataset[idx][1]
            noisy_label = original_label
            while noisy_label == original_label:
                noisy_label = random.randint(0, num_classes - 1)
            self.noisy_labels[idx] = noisy_label

    def load_noisy_labels(self,noisy_labels):
        self.noisy_labels = noisy_labels

    def get_noisy_sampler(self):
        dataset = self.dataset
        indices = list(self.noisy_labels.keys())
        
        sampler = NoisyLabelSampler(dataset, indices)
        sampler.load_noisy_labels(self.noisy_labels)

        return sampler

    def get_clean_sampler(self):
        noisy_set = set(self.noisy_labels.keys())  # Convert to set for O(1) lookups
        clean_indices = list(filter(lambda idx: idx not in noisy_set, self.indices))
        return NoisyLabelSampler(self.dataset, clean_indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        example, label = self.dataset[real_idx]
        
        if real_idx in self.noisy_labels:
            label = self.noisy_labels[real_idx]
        
        return example, label
    
    # def load_data(self):
    #     images = []
    #     labels = []
    #     for idx in self.indices:
    #         example, label = self.__getitem__(idx)
    #         images.append(example)
    #         labels.append(label)
    #     batch_images = torch.stack(images)
    #     batch_labels = torch.tensor(labels)
    #     return batch_images, batch_labels


class RamdomSampler(Dataset):
    
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.indices = np.random.choice(np.arange(self.dataset_size), size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example,label = self.dataset[self.indices[idx]]
        
        return example,label

class ClassSampler(Dataset):
    def __init__(self, dataset, class_no):
        self.dataset = dataset
        self.indices = np.where(np.array(self.dataset.labels) == class_no)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example,label = self.dataset[self.indices[idx]]
        
        return example,label

# class DirichletSampler(Dataset):
#     def __init__(self, dataset, dirichlet_p, class_num):
#         self.dataset = dataset
#         self.indices = []

#         for class_index in range(class_num):
#             self.indices += np.where(np.array(self.dataset.labels) == class_index)[0][dirichlet_p[class_index][0],dirichlet_p[class_index][1]]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         example,label = self.dataset[self.indices[idx]]
        
#         return example,label

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = copy.copy(dataloader)
        self.iterator = iter(copy.copy(self.dataloader))

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(copy.copy(self.dataloader))
            batch = next(self.iterator)
        return batch


    
def split_training_test(dataset, test_size = 0.2, random_state=42):
    training_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    from sklearn.model_selection import train_test_split

    train_images, test_images, train_labels, test_labels = train_test_split(
        dataset.images, dataset.labels, test_size = test_size, random_state=random_state)

    training_dataset.images = train_images
    training_dataset.labels = train_labels

    test_dataset.images = test_images
    test_dataset.labels = test_labels

    return training_dataset, test_dataset
    

def combine_dataset(dataset1, dataset2):
    combined_dataset = copy.copy(dataset1)
    combined_dataset.images.extend(copy.deepcopy(dataset2.images))
    combined_dataset.labels.extend(copy.deepcopy(dataset2.labels))
    
    return combined_dataset

def sampling_dataset(dataset, indices):
    sampled_dataset = copy.copy(dataset)

    sampled_dataset.images = (np.array(sampled_dataset.images)[indices]).tolist()
    sampled_dataset.labels = (np.array(sampled_dataset.labels)[indices]).tolist()

    return sampled_dataset

def delete_class_dataset(dataset, class_no):
    # Create a copy of the dataset
    modified_dataset = copy.copy(dataset)

    # Convert the labels to a numpy array for easy filtering
    labels_array = np.array(modified_dataset.labels)

    # Find the indices of the labels that are not equal to the class_index
    indices_to_keep = np.where(labels_array != class_no)[0]

    # Filter the images and labels to keep only the desired indices
    modified_dataset.images = (np.array(modified_dataset.images)[indices_to_keep]).tolist()
    modified_dataset.labels = (np.array(modified_dataset.labels)[indices_to_keep]).tolist()

    return modified_dataset
    
    