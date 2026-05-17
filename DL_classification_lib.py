#Deep Learning Commom Supervised Classification Lib ~ Pytorch 

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

"""
1. Common Training Method


"""


def epoch(loader, model, opt, device='cuda'):
    '''
    For normal training.
    '''
    model.train()
    total_loss, total_acc = 0., 0.

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_with_rounds(loader, model, opt, rounds, device='cuda'):
    '''
    For normal training with rounds.
    '''
    model.train()
    total_loss, total_acc = 0., 0.
    curr_round = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        curr_round += 1
        if curr_round == rounds:
            break

    return total_acc / (X.shape[0] * rounds), total_loss / (X.shape[0] * rounds)

def epoch_test(loader, model, device='cuda'):
    '''
    For normal test.
    '''
    model.eval()
    total_loss, total_acc = 0., 0.

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
           
            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
            
        return total_acc / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_test_each_class(loader, model, device='cuda'):
    '''
    For normal test with additional metrics (accuracy, precision, recall, F1-score) for each class.
    '''
    model.eval()
    total_loss = 0.
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            
            total_loss += loss.item() * X.shape[0]
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(yp.max(dim=1)[1].cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    # Calculate per-class accuracy
    unique_classes = sorted(set(all_labels))
    class_accuracies = []
    for cls in unique_classes:
        cls_indices = [i for i, label in enumerate(all_labels) if label == cls]
        cls_labels = [all_labels[i] for i in cls_indices]
        cls_preds = [all_preds[i] for i in cls_indices]
        class_accuracy = accuracy_score(cls_labels, cls_preds)
        class_accuracies.append(class_accuracy)
    
    class_metrics = {
        'overall_accuracy': overall_accuracy,
        'class_metrics': {}
    }
    
    for idx, cls in enumerate(unique_classes):
        class_metrics['class_metrics'][cls] = {
            'accuracy': class_accuracies[idx],
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx]
        }
    
    return avg_loss, class_metrics

def epoch_test_each_example(loader, model, device='cuda'):
    '''
    For normal test.
    '''
    model.eval()
    example_accuracy = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            #loss = nn.CrossEntropyLoss()(yp, y)
    
            example_accuracy += (yp.max(dim=1)[1] == y).cpu().tolist()
            
        return example_accuracy

def epoch_test_target(loader, model, target_class, device='cuda'):
    model.eval()

    target_num = 0.
    total_acc = 0.
    total_loss = 0.

    with torch.no_grad():
        for X, y in loader:
            X = X[y == target_class]
            y = y[y == target_class]
    
            X, y = X.to(device), y.to(device)
            if len(y) == 0:
                continue
            else:
                target_num += len(y)
    
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
    
            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
            
        return total_acc / target_num, total_loss / target_num 

def epoch_test_while_recording(loader, model, device='cuda'):
    '''
    Recording misclassified examples.
    '''
    model.eval()
    total_loss, total_acc = 0., 0.
    batch_size = loader.batch_size
    misclassified_indices_list = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            X, y = data
            X, y = X.to(device), y.to(device)
            
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
    
            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
    
            misclassified_batch_indices = (~(yp.max(dim=1)[1] == y)).nonzero()
            misclassified_indices = misclassified_batch_indices + (i * batch_size)
            misclassified_indices_list += misclassified_indices.cpu().numpy().tolist()
            # print(yp.max(dim=1)[1])
            # print(y)
            # print(misclassified_batch_indices)
            # print(misclassified_indices)
            
            # raise Exception
            
        return total_acc / len(loader.dataset), total_loss / len(loader.dataset), misclassified_indices_list

def epoch_test_logits(loader, model, device='cuda'):
    '''
    For logits test.
    '''
    model.eval()
    logits_list = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
    
            logits_list.append(yp)
        logits_tensor = torch.cat(logits_list, dim=0)
            
        return logits_tensor

def epoch_test_logits(loader, model, device='cuda'):
    '''
    For logits test.
    '''
    model.eval()
    logits_list = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
    
            logits_list.append(yp)
        logits_tensor = torch.cat(logits_list, dim=0)
            
        return logits_tensor

def epoch_test_probability(loader, model, only_label=False, device='cuda'):
    '''
    For logits test, converts logits to probabilities using softmax.
    '''
    model.eval()
    probabilities_list = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            probabilities = F.softmax(yp, dim=1)

            if only_label:
                probabilities = probabilities.gather(1, y.view(-1, 1)).squeeze(1)
                
            probabilities_list.append(probabilities)
            
        probabilities_tensor = torch.cat(probabilities_list, dim=0)
            
        return probabilities_tensor


def epoch_test_with_metrics(loader, model, flip = False, device='cuda'):
    '''
    For normal test with additional metrics: precision, recall, F1 score, and accuracy.
    '''
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
           
            preds = yp.max(dim=1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    if flip:
        all_preds = [1 - x for x in all_preds]
        all_labels = [1 - x for x in all_labels]
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    print("Confusion Matrix:\n", cm)

    class_accuracies = {}
    for i in range(len(cm)):
        class_accuracies[f'Class {i}'] = cm[i, i] / cm[i].sum()

    print("Class Accuracies:\n", class_accuracies)
    
    return accuracy, precision, recall, f1