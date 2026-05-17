#Deep Learning Commom Dataset Lib ~ Pytorch 

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
import sys

"""
- utils
- datasets
1. Mnist
2. Cifar10
3. Cifar100
4. TinyImageNet
5. CalTech256
"""



# 1. Mnist
class Mnist(Dataset):
    train_exist = False
    test_exist = False
    mnist_train = None
    mnist_test = None

    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset_name = "Mnist"
        self.Train = train
        self.root_dir = os.path.join(root, "Mnist")
        self.transform = transform
        self.download = download

        self.mnist_train = torchvision.datasets.MNIST(root=self.root_dir,train=True,download=self.download,transform=self.transform)
        self.mnist_test = torchvision.datasets.MNIST(root=self.root_dir,train=False,download=self.download,transform=self.transform)

        if self.Train:
            self.images = list(range(len(self.mnist_train)))
            self.labels = self.mnist_train.targets
            self.mnist = self.mnist_train
        else:
            self.images = list(range(len(self.mnist_test)))
            self.labels = self.mnist_test.targets
            self.mnist = self.mnist_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample,label = self.mnist[self.images[idx]]
        # if self.transform != None:
        #     sample = self.transform(sample)
        
        return sample,label


# 2. Cifar10
class Cifar10(Dataset):
    train_exist = False
    test_exist = False
    cifar10_train = None
    cifar10_test = None

    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset_name = "Cifar10"
        self.Train = train
        self.root_dir = os.path.join(root, "Cifar10")
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "test")
        self.download = download


        self.cifar10_train = torchvision.datasets.CIFAR10(root=self.train_dir,train=True,download=self.download,transform=self.transform)
        self.cifar10_test = torchvision.datasets.CIFAR10(root=self.val_dir,train=False,download=self.download,transform=self.transform)

        if self.Train:
            self.images = list(range(len(self.cifar10_train)))
            self.labels = self.cifar10_train.targets
            self.cifar10 = self.cifar10_train
        else:
            self.images = list(range(len(self.cifar10_test)))
            self.labels = self.cifar10_test.targets
            self.cifar10 = self.cifar10_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        sample,_ = self.cifar10[self.images[idx]]
        label = self.labels[idx]

        # if self.transform is not None:
        #     sample = self.transform(sample)
        
        
        return sample,label


# 3. Cifar100
class Cifar100(Dataset):
    train_exist = False
    test_exist = False
    cifar100_train = None
    cifar100_test = None

    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset_name = "Cifar100"
        self.Train = train
        self.root_dir = os.path.join(root, "Cifar100")
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "test")
        self.download = download

        if self.Train:
            self.dataset = torchvision.datasets.CIFAR100(root=self.train_dir, train=True, transform=self.transform)
            self.images = list(range(len(self.dataset)))
            self.labels = self.dataset.targets
        else:
            self.dataset = torchvision.datasets.CIFAR100(root=self.val_dir, train=False, transform=self.transform)
            self.images = list(range(len(self.dataset)))
            self.labels = self.dataset.targets
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        sample, label = self.dataset[self.images[idx]]
        
        # print(len(sample))
        # label = self.labels[idx]

        # if self.transform is not None:
        #     sample = self.transform(sample)
        
        return sample,label

# 4. TinyImageNet
class TinyImageNet(Dataset):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset_name = "TinyImageNet"
        self.Train = train
        self.root_dir = os.path.join(root, "TinyImageNet")
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "tiny-imagenet-200/train")
        self.val_dir = os.path.join(self.root_dir, "tiny-imagenet-200/val")

        if download:
            self.download()

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "tiny-imagenet-200/words.txt")
        wnids_file = os.path.join(self.root_dir, "tiny-imagenet-200/wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def download(self):
        if not self._check_exists():
            return download_and_extract_archive(self.url, self.root_dir, filename='tiny-imagenet-200.zip',
                                                remove_finished=False, md5=self.zip_md5)
    
    def _check_exists(self):
        return os.path.exists(self.root_dir+'tiny-imagenet-200.zip')

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        self.labels = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            label = self.class_to_tgt_idx[tgt]
                        else:
                            label = self.class_to_tgt_idx[self.val_img_to_class[fname]]
                        self.images.append(path)
                        self.labels.append(label)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        tgt = self.labels[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = np.array(sample)

        return sample, tgt
    
# 5. CalTech256    
class CalTech256(Dataset):
    def __init__(self, root, train=True, transform=None, split = 0.8, download=True):
        self.dataset_name = "CalTech256"
        self.Train = train
        self.root_dir = os.path.join(root, "CalTech256")
        
        self.transform = transform
        
        if download:
            self.download()
        
        self.doc_dir = os.path.join(self.root_dir,"caltech256/256_ObjectCategories")

        if os.path.exists(os.path.join(self.doc_dir,"198.spider/RENAME2")):
            os.remove(os.path.join(self.doc_dir,"198.spider/RENAME2"))
        if os.path.exists(os.path.join(self.doc_dir,"056.dog/greg")):
            os.rmdir(os.path.join(self.doc_dir,"056.dog/greg/vision309"))
            os.rmdir(os.path.join(self.doc_dir,"056.dog/greg"))

        self.class_dir = os.listdir(self.doc_dir)
        self.class_dir.sort()
        self.class_dir = [os.path.join(self.doc_dir,class_path) for class_path in self.class_dir][:-1]
        self.split = split

        self.images = []
        self.labels = []

        if (self.Train):
            for label, class_path in enumerate(self.class_dir):
                #print(label,":",class_path)
                image_dir = os.listdir(class_path)
                image_dir.sort()
                image_dir = image_dir[:int(len(image_dir)*self.split)]
                for image_path in image_dir:
                    self.images.append(os.path.join(class_path,image_path))
                    self.labels.append(label)
        else:
            for label, class_path in enumerate(self.class_dir):
                image_dir = os.listdir(class_path)
                image_dir.sort()
                image_dir = image_dir[int(len(image_dir)*self.split):]
                for image_path in image_dir:
                    self.images.append(os.path.join(class_path,image_path))
                    self.labels.append(label)
                    
    def download(self):
        if not self._check_exists():
            torchvision.datasets.Caltech256(root=self.root_dir,download=self.download)

    
    def _check_exists(self):
        return os.path.exists(self.root_dir+'256_ObjectCategories.tar')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        with open(img_path, 'rb') as f:
            image = Image.open(img_path)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.array(image)
        label = torch.tensor(label,dtype=torch.long)
        return image, label

    
def read_data_from_dataset(dataset,index_list,mergedim=True):
    image_list = []
    label_list = []
    
    for index in index_list:
        image,label = dataset[index]
        image_list.append(image)
        label_list.append(label)
        
    if mergedim:
        return torch.stack(image_list),torch.tensor(label_list)
    else:
        return image_list,label_list


def transform_example(example, 
                    data_transform = transforms.Compose([
                      transforms.Lambda(lambda x: x.unsqueeze(0))]),
                    label_transform = transforms.Compose([
                      transforms.Lambda(lambda x: torch.tensor(x)),
                      transforms.Lambda(lambda x: x.unsqueeze(0))
                                        ])):
    X, y = example
    X = data_transform(X) if data_transform is not None else X
    
    y = label_transform(y) if label_transform is not None else y

    return (X,y)


