
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.models import *

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

def save_model(model, path):
    torch.save(model.state_dict(),path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

    return model

def load_model_based_generator(model_generator, model_path, device='cuda'):
    model = model_generator().to(device)
    model.load_state_dict(torch.load(model_path))

    return model

def replace_bn_with_gn_in_model(module, default_groups = 16, affine=False, device = 'cuda'):
    if isinstance(module, nn.Sequential):
        # If the module is Sequential, recursively replace BN with GN for all submodules
        for name, sub_module in module.named_children():
            setattr(module, name, replace_bn_with_gn_in_model(sub_module, device = device))
    elif isinstance(module, nn.ModuleList):
        # If the module is a ModuleList, recursively replace BN with GN for all submodules
        for i, sub_module in enumerate(module):
            module[i] = replace_bn_with_gn_in_model(sub_module, device = device)
    elif isinstance(module, nn.BatchNorm2d):
        # If the module is a BN layer, replace it with a GN layer
        num_channels = module.num_features
        num_groups = max(num_channels // default_groups, 1)  # Assuming 16 is the default value for GN's num_groups
        gn = nn.GroupNorm(num_groups, num_channels, affine=affine)
        return gn.to(device)
    elif isinstance(module, nn.Module):
        # If the module is a single module, recursively replace BN with GN
        for name, sub_module in module.named_children():
            setattr(module, name, replace_bn_with_gn_in_model(sub_module, device = device))
    return module


def find_gn_in_model(module,  group_normalization_list, default_groups = 16, device = 'cuda'):
    if isinstance(module, nn.Sequential):
        # If the module is Sequential, recursively replace BN with GN for all submodules
        for name, sub_module in module.named_children():
            setattr(module, name, find_gn_in_model(sub_module, group_normalization_list, device = device))
    elif isinstance(module, nn.ModuleList):
        # If the module is a ModuleList, recursively replace BN with GN for all submodules
        for i, sub_module in enumerate(module):
            module[i] = find_gn_in_model(sub_module, group_normalization_list, device = device)
    elif isinstance(module, nn.GroupNorm):
        group_normalization_list.append(module)
        return module
    elif isinstance(module, nn.Module):
        # If the module is a single module, recursively replace BN with GN
        for name, sub_module in module.named_children():
            setattr(module, name, find_gn_in_model(sub_module, group_normalization_list, device = device))
    return module


def replace_bn_with_identity_in_model(module, device = 'cuda'):
    if isinstance(module, nn.Sequential):
        for name, sub_module in module.named_children():
            setattr(module, name, replace_bn_with_identity_in_model(sub_module, device = device))
    elif isinstance(module, nn.ModuleList):
        for i, sub_module in enumerate(module):
            module[i] = replace_bn_with_identity_in_model(sub_module, device = device)
    elif isinstance(module, nn.BatchNorm2d):
        identity = nn.Identity()
        if device is not None:
            return identity.to(device)
        else:
            return identity
    elif isinstance(module, nn.Module):
        for name, sub_module in module.named_children():
            setattr(module, name, replace_bn_with_identity_in_model(sub_module, device = device))
    return module
    
def classification_model_generator(architecture, output_dimension, weights='DEFAULT'):
    
    generator = {"AlexNet":generate_alexnet,
                "Inception":generate_inception,
                "Resnet18":generate_resnet18,
                "Resnet34":generate_resnet34,
                "Resnet50":generate_resnet50,
                "Resnet101":generate_resnet101,
                "Resnet152":generate_resnet152}
    
    
    return generator[architecture](output_dimension,weights)

def get_classification_model_generator(architecture, output_dimension, weights='DEFAULT'):
    
    generator_dict = {"AlexNet":generate_alexnet,
                "Inception":generate_inception,
                "Resnet18":generate_resnet18,
                "Resnet34":generate_resnet34,
                "Resnet50":generate_resnet50,
                "Resnet101":generate_resnet101,
                "Resnet152":generate_resnet152}

    def generator():
        
        return generator_dict[architecture](output_dimension,weights)
    
    return generator

def get_classification_model_generator_without_BN(architecture, output_dimension, weights='DEFAULT'):
    
    generator_dict = {"AlexNet":generate_alexnet,
                "Inception":generate_inception,
                "Resnet18":generate_resnet18,
                "Resnet34":generate_resnet34,
                "Resnet50":generate_resnet50,
                "Resnet101":generate_resnet101,
                "Resnet152":generate_resnet152}

    def generator():
        model = generator_dict[architecture](output_dimension,weights)
        replace_bn_with_identity_in_model(model, device=None)
        
        return model
    
    return generator


def get_classification_model_generator_with_GN(architecture, output_dimension, weights='DEFAULT'):
    
    generator_dict = {"AlexNet":generate_alexnet,
                "Inception":generate_inception,
                "Resnet18":generate_resnet18,
                "Resnet34":generate_resnet34,
                "Resnet50":generate_resnet50,
                "Resnet101":generate_resnet101,
                "Resnet152":generate_resnet152}

    def generator():
        model = generator_dict[architecture](output_dimension,weights)
        replace_bn_with_gn_in_model(model, device=None)
        
        return model
    
    return generator



def generate_alexnet(output_dimension, weights='DEFAULT'):  
    from torchvision.models import alexnet
    if weights != None:
        model = alexnet(weights)
    else:
        model = alexnet()
        
    model.classifier.append(nn.Linear(1000,output_dimension,bias=True))
    return model

def generate_inception(output_dimension, weights='DEFAULT'):  
    from torchvision.models import inception_v3
    if weights != None:
        model = inception_v3(weights)
    else:
        model = inception_v3()
        
    model.fc = nn.Linear(2048,output_dimension,bias=True)
    return model

    
def generate_resnet18(output_dimension, weights='DEFAULT'):  
    from torchvision.models import resnet18
    if weights != None:
        model = resnet18(weights)
    else:
        model = resnet18()
        
    model.fc = nn.Sequential(
        nn.Linear(512, output_dimension, bias=True)
    )
    return model

def generate_resnet34(output_dimension, weights='DEFAULT'):  
    from torchvision.models import resnet34
    if weights != None:
        model = resnet34(weights)
    else:
        model = resnet34()
    model.fc = nn.Sequential(
        nn.Linear(512, output_dimension, bias=True)
    )
    return model

def generate_resnet50(output_dimension, weights='DEFAULT'):  
    from torchvision.models import resnet50
    if weights != None:
        model = resnet50(weights)
    else:
        model = resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048,1000,bias=True),
        nn.Linear(1000, output_dimension, bias=True)
    )
    return model

def generate_resnet101(output_dimension, weights='DEFAULT'):  
    from torchvision.models import resnet101
    if weights != None:
        model = resnet101(weights)
    else:
        model = resnet101()
    model.fc = nn.Sequential(
        nn.Linear(2048,1000,bias=True),
        nn.Linear(1000, output_dimension, bias=True)
    )
    return model

def generate_resnet152(output_dimension, weights='DEFAULT'):  
    from torchvision.models import resnet152
    if weights != None:
        model = resnet152(weights)
    else:
        model = resnet152()
    model.fc = nn.Sequential(
        nn.Linear(2048,1000,bias=True),
        nn.Linear(1000, output_dimension, bias=True)
    )
    return model
    

    
def weight_reset(module):
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        module.reset_parameters()


def model_parameter_reset(model):
    model.apply(weight_reset)


def get_model_distance(model1, model2):
    with torch.no_grad():
        model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
        model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
        distance = torch.square(torch.norm(model1_flattened - model2_flattened))
    return distance