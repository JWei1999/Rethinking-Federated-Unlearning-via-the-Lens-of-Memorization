import torch

import torchvision
import torchvision.utils as vutils
from torchvision import models, utils, datasets, transforms

import sys
import os
import re
import copy
import time
import math
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def image2tensor_PIL(image,transform=None):
    '''
    image2tensor based on PIL
    '''
    
    if not transform:
        transform = transforms.ToTensor()
    
    return transform(image)

def tensor2image_PIL(single_image_tensor):
    '''
    image2tensor based on tensor
    '''
    transform = transforms.ToPILImage()
    
    return transform(single_image_tensor)

def show_image(image, title=None):
    '''
    Show image, tensor and dim = 3
    '''
    image = image[0]
    image = transforms.ToPILImage()(image)
    plt.imshow(image, cmap=None)
    if title is not None:
        plt.title(str(title))
    plt.axis('off')
    plt.show()


def show_images(images, titles=None):
    '''
    Show images, tensor and dim = 4
    '''
    num = images.shape[0]
    plt.figure(figsize=(12, 4))
    for i in range(num):
        plt.subplot(math.ceil(num/8), 8, i + 1)
        plt.imshow(transforms.ToPILImage()(images[i]))
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')

    plt.show()

def randomly_pick_images_from_dataset(dataset, num, target_class = None, with_label=True, tensor=True):
    """
    Randomly pick images from the dataset.

    Args:
        dataset: The dataset object from which images will be picked.
        num (int): Number of images to pick.
        target_class (int, optional): If specified, images will be picked only from this class.
        with_label (bool, optional): If True, return image-label pairs. If False, return only images.
        tensor (bool, optional): If True, return images as PyTorch tensors. If False, return as PIL images.

    Returns:
        List of sampled images or image-label pairs.
    """

    img2tensor = transforms.ToTensor()
    img2PIL = transforms.ToPILImage()
    
    if target_class is not None:
        indices = [idx for idx, label in enumerate(dataset.labels) if label == target_class]
        indices = random.sample(indices, min(num, len(indices)))
    else:
        indices = random.sample(range(len(dataset)), num)

    sampled_data = []
    sampled_labels = []
    for idx in indices:
        if with_label:
            sample, label = dataset[idx]
            if tensor and not torch.is_tensor(sample):
                sample = img2tensor(sample)
            elif not tensor:
                sample = img2PIL(sample)
            sampled_data.append(sample)
            sampled_labels.append(label)
        else:
            sample, _ = dataset[idx]
            if tensor and not torch.is_tensor(sample):
                sample = img2tensor(sample)
            elif not tensor:
                sample = img2PIL(sample)
            sampled_data.append(sample)

    if tensor and with_label:
        return (torch.stack(sampled_data),sampled_labels)
    elif not tensor and with_label:
        return (sampled_data,sampled_labels)
    elif tensor:
        return torch.stack(sampled_data)
    else:
        return sampled_data
