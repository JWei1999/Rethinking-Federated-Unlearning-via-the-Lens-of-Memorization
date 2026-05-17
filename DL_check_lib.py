import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
import torchvision.utils as vutils
from torchvision import models, datasets, transforms
from collections import defaultdict, OrderedDict
from copy import deepcopy
import re
import copy
import time
import math
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import sys
import os
from PIL import Image


from DL_classification_lib import *

def check_parameters(model, only_shape = False):
    for name, p in model.named_parameters():
        
        print(name)
        print(p.shape)
        
        if not only_shape:
            print(p)


def check_parameter_diff(model1, model2):
    for o1, o2 in zip(model1.named_parameters(), model2.named_parameters()):
        name1, parameter1 = o1
        name2, parameter2 = o2
        
        print(name1)
        print(torch.norm(parameter1-parameter2)/sum([*parameter1.shape]))


def check_with_one_batch(loader, model, device='cuda'):
    model.eval()
    
    for X,y in loader:
        X = X.to(device)
        y = y.to(device)

        yp = model(X)
        
        print("Prediction")
        print(yp)

        print("Labels")
        print(y)

        break

def check_with_one_example(loader, model, index, device='cuda'):
    model.eval()
    
    # Access the dataset directly
    dataset = loader.dataset
    
    # Get the specific example
    X, y = dataset[index]
    X = X.unsqueeze(0).to(device)  # Add batch dimension
    y = torch.tensor([y], device=device)  # Ensure y is a tensor and on the same device
    
    # Forward pass through the model
    with torch.no_grad():
        yp = model(X)
    
    # Print predictions and labels
    print("Prediction (logits):", yp)
    print("Labels:", y)

def find_max_loss_example_indices(loader, model, max_n = 3, device='cuda'):
    model.eval()
    all_losses = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)

            # Get predictions
            yp = model(X)

            # Compute loss for the batch
            criterion = nn.CrossEntropyLoss(reduction='none')
            batch_losses = criterion(yp, y)

            # Store the losses and corresponding indices
            all_losses.extend(batch_losses.cpu().tolist())
            #all_indices.extend(list(range(batch_idx*len(batch_losses), (batch_idx+1)*len(batch_losses))))

    # Convert to tensors for processing
    all_losses = torch.tensor(all_losses)
    # all_indices = torch.tensor(all_indices)

    # print(all_indices)

    # Get the indices of the top n losses
    top_n_loss_indices = torch.topk(all_losses, max_n, largest=True).indices

    top_n_examples = [(idx, all_losses[idx].item()) for idx in top_n_loss_indices]
    # Print the results
    print(f"Top {max_n} Losses and Corresponding Indices:")
    for idx, loss in top_n_examples:
        print(f"Index: {idx}, Loss: {loss:.4f}")
    
    return top_n_loss_indices


def find_dismatch_example_indices(loader, main_model, compared_model):
    
    main_target_acc_list = epoch_test_each_example(loader, main_model)
    compared_target_acc_list = epoch_test_each_example(loader, compared_model)

    main_target_acc_list = np.array(main_target_acc_list)
    compared_target_acc_list = np.array(compared_target_acc_list)

    tp_indices = []
    fp_indices = []
    fn_indices = []
    tn_indices = []

    # Identify indices for TP, FP, FN, and TN
    for idx, (main, compared) in enumerate(zip(main_target_acc_list, compared_target_acc_list)):
        if main == compared == 1:  # True Positive
            tp_indices.append(idx)
        elif main == 0 and compared == 1:  # False Positive
            fp_indices.append(idx)
        elif main == 1 and compared == 0:  # False Negative
            fn_indices.append(idx)
        elif main == compared == 0:  # True Negative
            tn_indices.append(idx)

    # Return the indices
    return {
        "tp": tp_indices,
        "fp": fp_indices,
        "fn": fn_indices,
        "tn": tn_indices
    }


def check_parameters_grads_with_one_example(loader, model, index, ifprint = True, logger = None, device='cuda'):
    model.train()
    
    # Access the dataset directly
    dataset = loader.dataset
    
    # Get the specific example
    X, y = dataset[index]
    X = X.unsqueeze(0).to(device)  # Add batch dimension
    y = torch.tensor([y], device=device)  # Ensure y is a tensor and on the same device
    
    
    yp = model(X)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(yp, y)
    loss.backward()

    if ifprint:
        print("Parameters:")
        for name, p in model.named_parameters():
            print("Layer Name", name)
            print(p.shape)
            print(p)
    
        print("#######################################")
    
        print("Gradients:")
        for name, p in model.named_parameters():
            print("Layer Name", name)
            if p.grad is not None:
                print(p.grad.shape)
                print(p.grad)
            else:
                print("None")

    if logger is not None:
        logger.info("Parameters:")
        for name, p in model.named_parameters():
            logger.info(f"Layer Name: {name}")
            logger.info(p.shape)
            logger.info(p)
    
        logger.info("#######################################")
    
        logger.info("Gradients:")
        for name, p in model.named_parameters():
            logger.info(f"Layer Name: {name}")
            if p.grad is not None:
                logger.info(p.grad.shape)
                logger.info(p.grad)
            else:
                logger.info("None!")

