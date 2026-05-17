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
from DL_vision_dataset import *
from DL_vision_dataset_sampling import *
from DL_vision_model import *
from General_utils import *
from Image_utils import *

def fedAvg_epoch_with_tracing_gradient_stat(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, logger = None,
                 batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, target_client_index=None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    server.average_model_update_stat_dict = {}
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client in clients:
            client.update_local_model(global_model_state_dict)
            
            client_loss, client_acc, client_examples = client.training_local_dataset(
                local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        audit_client_gradients(server.average_model_update_stat_dict, model_update_list, target_client_index = target_client_index)
        
        server.train_model(model_update_list)


        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    return train_acc, train_loss, test_acc, test_loss, time_elapsed


def audit_client_gradients(average_model_update_stat_dict, model_update_list, windows_size = 1, target_client_index = None, device="cuda"):
    from collections import deque
    
    if len(average_model_update_stat_dict) == 0:
        example_update = model_update_list[0]
        for key in example_update:
            average_model_update_stat_dict[key] = deque(maxlen=windows_size)

    update_dict = {key: [] for key in model_update_list[0]}

    for client_index, model_update_dict in enumerate(model_update_list):
       
        if client_index == target_client_index:
            continue
        
        for key in model_update_dict:
            update_dict[key].append(model_update_dict[key])

    for key in average_model_update_stat_dict:
        average_model_update_layer = torch.mean(
            torch.stack(update_dict[key]), dim=0)
        average_model_update_stat_dict[key].append(average_model_update_layer)

    

    return average_model_update_stat_dict


def fedAvg_epoch_with_saving_gradients(server, clients, client_datasets, test_data_loader, epochs, local_epochs, target_client_index, val_data_loader = None, logger = None, batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client in clients:
            client.update_local_model(global_model_state_dict)
            
            client_loss, client_acc, client_examples = client.training_local_dataset(
                local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        server.saving_gradient = model_update_list[target_client_index]
        
        server.train_model(model_update_list)


        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    return train_acc, train_loss, test_acc, test_loss, time_elapsed


def fedAvg_epoch_fine_tuning(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, logger = None,
                 batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, target_client_index=None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            # if client_index == target_client_index:
                
            #     client_loss, client_acc, client_examples = client.training_local_dataset(
            #         local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle, loss_func = acsent_loss)
            # else:
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        
        server.train_model(model_update_list, epoch_index=epoch_index)


        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    return train_acc, train_loss, test_acc, test_loss, time_elapsed


# def acsent_loss(yp, y):

#     return -nn.CrossEntropyLoss()(yp, y)

"""
Server momentum
"""

class CyclicLR:
    def __init__(self, base_lr, max_lr, step_size, mode='cosine'):

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode.lower()
        self.iteration = 0
        
    def get_lr(self, epoch_index = None):
        
        if epoch_index is not None:
            self.iteration = epoch_index
        
        cycle = self.iteration // (2 * self.step_size)
        x = (self.iteration % (2 * self.step_size)) / self.step_size
        
        if self.mode == 'cosine':
            if x <= 1:
                scale_factor = 0.5 * (1 - math.cos(math.pi * x))  # 从 base_lr 递增到 max_lr
            else:
                scale_factor = 0.5 * (1 + math.cos(math.pi * (x-1)))  # 从 max_lr 递减到 base_lr
        else:
            raise ValueError("Unsupported mode. Only 'cosine' is implemented.")
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * scale_factor
        
        
        self.iteration += 1
        return lr

def aggregate_parameter_with_momentum_sa(server, client_lr, max_lr, step_size, momentum = 0.5):

    lr_scheduler = CyclicLR(client_lr, max_lr, step_size)
    
    def _aggregate_grads(model_update_list, epoch_index, return_candidx, **kwargs):

        # print(len(model_update_list))
        # print(epoch_index)
        
        size = len(model_update_list[0])
        update_dict = {key: [] for key in model_update_list[0]}

        for model_update_dict in model_update_list:
            for key in model_update_dict:
                update_dict[key].append(model_update_dict[key])

        average_model_update = []

        for key in update_dict:
            average_model_update_layer = torch.mean(
                torch.stack(update_dict[key]), dim=0)
            
            average_model_update.append(average_model_update_layer)

        model_update = []
        server_lr = lr_scheduler.get_lr(epoch_index)
        
        # print(server_lr)
        # print(hasattr(server, 'last_model_update'))
        
        if hasattr(server, 'last_model_update'):

            last_model_update = server.last_model_update
            
            for current_layer_update, last_layer_update in zip(average_model_update, last_model_update):
                
                layer_update = (last_layer_update * momentum + (1 - momentum) * current_layer_update) * (server_lr/client_lr)
                model_update.append(layer_update)
    
            server.last_model_update = model_update

        else:

            for current_layer_update in average_model_update:
                
                layer_update = current_layer_update * (server_lr/client_lr)
                model_update.append(layer_update)
    
            server.last_model_update = model_update

        if return_candidx:
            return model_update, list(range(len(model_update_list)))
        else:
            return model_update
    
    return _aggregate_grads



"""
Training & Pruning
"""


def get_initialized_weights(shape, method="normal", device="cuda"):
    import torch.nn.init as init
    """Returns a new tensor with initialized values based on the chosen method."""
    new_tensor = torch.empty(shape).to(device)  # Create an empty tensor of the given shape
    if method == "xavier_uniform":
        init.xavier_uniform_(new_tensor)
    elif method == "xavier_normal":
        init.xavier_normal_(new_tensor)
    elif method == "kaiming_uniform":
        init.kaiming_uniform_(new_tensor, nonlinearity='relu')
    elif method == "kaiming_normal":
        init.kaiming_normal_(new_tensor, nonlinearity='relu')
    elif method == "normal":
        init.normal_(new_tensor, mean=0.0, std=0.1)
    elif method == "uniform":
        init.uniform_(new_tensor, a=-0.1, b=0.1)
    elif method == "zeros":
        init.zeros_(new_tensor)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return new_tensor


def select_pruning_neurons(pruning_epoch_index,indices,ratio=0.5,random_sampling=False):
    length = len(indices)
    ratio_length = int(length*ratio)

    if not random_sampling:
        start_index = int(pruning_epoch_index*ratio_length)
        end_index = int((pruning_epoch_index+1)*ratio_length)
        if start_index >= length:
            return []
        if end_index >= length:
            end_index = length
        
        pruning_indices = indices[start_index:end_index]
    
        return pruning_indices
    else:
        #print(indices)
        pruning_indices = indices[torch.randperm(len(indices))[:ratio_length]]
    
        return pruning_indices
    

def mapping_indices(key_info, indices, keys, device='cuda'):
    # print("##############")
    # print(indices)
    start_indices,key_index_ranges = key_info
    start_indices = torch.tensor(start_indices, device=device)
    key_bins = torch.searchsorted(start_indices, indices, right=True) - 1 

    relative_indices = indices - start_indices[key_bins]

    sorted_indices_by_key = {key: None for key in keys}
    unique_keys = torch.unique(key_bins)

    for key_idx in unique_keys:
        key = list(key_index_ranges.keys())[key_idx.item()]
        shape = key_index_ranges[key][2] 

        mask = key_bins == key_idx
        rel_indices_for_key = relative_indices[mask]

        multi_dim_indices = torch.unravel_index(rel_indices_for_key, shape)

        sorted_indices_by_key[key] = torch.stack(multi_dim_indices, dim=1).long()

    return sorted_indices_by_key
    

def find_memorization_neurons(average_model_update_stat_dict, ratio=0.3, device='cuda'):
    average_windows_abs_model_update_stat_dict = {}

    for key in average_model_update_stat_dict:
        average_windows_abs_model_update_stat_dict[key] = torch.mean(torch.stack(
            [torch.abs(update) for update in average_model_update_stat_dict[key]]), dim=0)

    grad_list = []
    key_index_ranges = {}
    start_indices = []

    start_idx = 0
    for key, tensor in average_windows_abs_model_update_stat_dict.items():
        flat_tensor = tensor.view(-1)
        grad_list.append(flat_tensor)

        key_index_ranges[key] = (start_idx, start_idx + len(flat_tensor), tensor.shape)
        start_indices.append(start_idx)
        start_idx += len(flat_tensor)

    grad_list = torch.cat(grad_list).to(device)
    _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))

    return (start_indices,key_index_ranges), indices
    

def drop_neuron_by_indices(model, indices_by_keys, neuron_level = False, init_method="kaiming"):
    with torch.no_grad():
        state_dict = copy.deepcopy(model.state_dict())

        for key in indices_by_keys:
            weight = state_dict[key]
            #print(weight.shape)
            
            indices = indices_by_keys[key]
            #print(indices.shape)
            #print(weight.shape)
            if indices == None:
                continue
                
            if not neuron_level:
                idx = indices.t()
                #weight[tuple(idx)] = 0.
                #print(weight[tuple(idx)].shape)

                if init_method == "kaiming":
                    if len(weight.shape) > 1:
                        initialized_weights = get_initialized_weights(weight.shape, method="kaiming_uniform", device="cuda")
                        weight[tuple(idx)] = initialized_weights[tuple(idx)]
                    else:
                        weight[tuple(idx)] = get_initialized_weights(weight[tuple(idx)].shape, method="uniform", device="cuda")
                elif init_method == "zeros":
                    weight[tuple(idx)] = get_initialized_weights(weight[tuple(idx)].shape, method="zeros", device="cuda")
            else:
                idx = [indices] + [slice(None)] * (weight.ndim - 1)  # affects dim 0 (neurons)
                idx = tuple(idx)

                if init_method == "kaiming":
                    initialized_weights = get_initialized_weights(weight.shape, method="kaiming_uniform", device=weight.device)
                elif init_method == "zeros":
                    initialized_weights = get_initialized_weights(weight.shape, method="zeros", device=weight.device)
                weight[idx] = initialized_weights[idx]
                
        
            state_dict[key] = weight
            
    return state_dict

def measure_memorization_scores(within_model,without_models,dataset):

    loader = DataLoader(dataset, batch_size = 128, shuffle = False)

    baseline_examples_prob = epoch_test_probability(loader, within_model, only_label=True)

    exclude_examples_probs = []

    for without_model in without_models:
    
        exclude_examples_prob = epoch_test_probability(loader, without_model, only_label=True)
        exclude_examples_probs.append(exclude_examples_prob)

    memorization_scores = baseline_examples_prob - torch.mean(torch.stack(exclude_examples_probs), dim=0)

    return memorization_scores

def fedAvg_epoch_fine_tuning_while_pruning_redundant_neurons(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, 
                                                                logger = None,
                   batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, 
                   target_client_index=None, 
                   pruning_epoches=None,
                   return_dynamic = False,
                   drop_ratio=0.2,
                   round_drop_ratio=0.75,
                   round_random_drop=True,
                   init_method="kaiming"):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    server.average_model_update_stat_dict = {}
    if pruning_epoches is None:
        pruning_epoches = [0]
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    epoch_data = {
        "training_acc":[],
        "training_loss":[],
        "test_acc":[],
        "test_loss":[],
        "val_acc":[],
        "val_loss":[],
        "time":[]
    }
    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            # if client_index == target_client_index:
                
            #     client_loss, client_acc, client_examples = client.training_local_dataset(
            #         local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle, loss_func = acsent_loss)
            # else:
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)
        
        server.train_model(model_update_list, epoch_index=epoch_index)

        if epoch_index in pruning_epoches:
            pruning_epoch_index = pruning_epoches.index(epoch_index)
            if epoch_index == pruning_epoches[0]:
                average_model_update_stat_dict = update_average_model_update_stat_dict(server.average_model_update_stat_dict, model_update_list)
                key_info, memorziation_indices = find_memorization_neurons(average_model_update_stat_dict, ratio=drop_ratio)
                
                server.memorziation_indices = memorziation_indices
                server.key_info = key_info
                server.keys = average_model_update_stat_dict.keys()
                
            pruning_indices = select_pruning_neurons(pruning_epoch_index, server.memorziation_indices, ratio=round_drop_ratio, random_sampling=round_random_drop)
            pruning_indices_by_key = mapping_indices(server.key_info, pruning_indices, server.keys)
            unlearning_stat_dict = drop_neuron_by_indices(server.global_model, pruning_indices_by_key, init_method=init_method)
            server.global_model.load_state_dict(unlearning_stat_dict)

        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        epoch_data["training_acc"].append(train_acc)
        epoch_data["training_loss"].append(train_loss)
        epoch_data["test_acc"].append(test_acc)
        epoch_data["test_loss"].append(test_loss)
        epoch_data["val_acc"].append(val_acc)
        epoch_data["val_loss"].append(val_loss)
        epoch_data["time"].append(time_elapsed)

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    if return_dynamic:
        return train_acc, train_loss, test_acc, test_loss, time_elapsed,epoch_data

    return train_acc, train_loss, test_acc, test_loss, time_elapsed

def update_average_model_update_stat_dict(average_model_update_stat_dict, model_update_list, windows_size = 1, device="cuda"):
    from collections import deque
    
    if len(average_model_update_stat_dict) == 0:
        example_update = model_update_list[0]
        for key in example_update:
            average_model_update_stat_dict[key] = deque(maxlen=windows_size)

    update_dict = {key: [] for key in model_update_list[0]}

    for client_index, model_update_dict in enumerate(model_update_list):
        for key in model_update_dict:
            update_dict[key].append(model_update_dict[key])

    for key in average_model_update_stat_dict:
        average_model_update_layer = torch.mean(
            torch.stack(update_dict[key]), dim=0)
        average_model_update_stat_dict[key].append(average_model_update_layer)

    return average_model_update_stat_dict

def find_critic_neurons_by_salun(target_model_update_stat_dict, target_model_stat_dict, ratio=0.3, device='cuda', **kwargs):
    target_abs_model_update_stat_dict = {}

    for key in target_model_update_stat_dict:
        target_abs_model_update_stat_dict[key] = torch.abs(target_model_update_stat_dict[key])

    grad_list = []
    key_index_ranges = {}
    start_indices = []

    start_idx = 0
    for key, tensor in target_abs_model_update_stat_dict.items():
        flat_tensor = tensor.view(-1)
        grad_list.append(flat_tensor)

        key_index_ranges[key] = (start_idx, start_idx + len(flat_tensor), tensor.shape)
        start_indices.append(start_idx)
        start_idx += len(flat_tensor)

    grad_list = torch.cat(grad_list).to(device)
    _, indices = torch.topk(grad_list, int(len(grad_list) * ratio))

    start_indices = torch.tensor(start_indices, device=device)
    key_bins = torch.searchsorted(start_indices, indices, right=True) - 1 

    relative_indices = indices - start_indices[key_bins]

    sorted_indices_by_key = {key: None for key in target_abs_model_update_stat_dict}
    unique_keys = torch.unique(key_bins)

    for key_idx in unique_keys:
        key = list(key_index_ranges.keys())[key_idx.item()]
        shape = key_index_ranges[key][2] 

        mask = key_bins == key_idx
        rel_indices_for_key = relative_indices[mask]

        multi_dim_indices = torch.unravel_index(rel_indices_for_key, shape)

        sorted_indices_by_key[key] = torch.stack(multi_dim_indices, dim=1).long()

    return sorted_indices_by_key
    #return (start_indices,key_index_ranges), indices


def find_critic_neurons_by_local_strategy(target_model_update_stat_dict, target_model_stat_dict, ratio=0.3, device='cuda', top_h=5, **kwargs):
    criticality_scores = {}

    for key in target_model_stat_dict:
        weight = target_model_stat_dict[key]          # θ^o
        grad = target_model_update_stat_dict[key]           # g(θ^o, S)

        if weight.ndim < 2:  # e.g., bias or LayerNorm params, skip
            continue

        # Compute per-parameter criticality scores: |θ_j * g_j|
        score = torch.abs(weight * grad)  # same shape as weight

        # Reduce per-channel (first dimension, usually output channels)
        per_channel_scores = []
        for i in range(score.shape[0]):  # over output channels
            flattened_scores = score[i].flatten()
            top_scores = torch.topk(flattened_scores, min(top_h, flattened_scores.numel())).values
            channel_score = top_scores.mean().item()
            per_channel_scores.append(channel_score)

        criticality_scores[key] = per_channel_scores

    indices_by_keys = {}

    for key, scores in criticality_scores.items():
        scores_tensor = torch.tensor(scores)
        num_to_drop = int(len(scores_tensor) * ratio)
        if num_to_drop == 0:
            indices_by_keys[key] = None
            continue

        topk = torch.topk(scores_tensor, num_to_drop).indices
        indices_by_keys[key] = topk
        #print(indices_by_keys)
    
    return indices_by_keys

def find_critic_neurons_by_deep_layers(target_model_update_stat_dict, target_model_stat_dict, ratio=0.3, device='cuda', split_layer_index=11, **kwargs):
    target_abs_model_update_stat_dict = {}

    for key in target_model_update_stat_dict:
        target_abs_model_update_stat_dict[key] = torch.abs(target_model_update_stat_dict[key])

    grad_list = []
    key_index_ranges = {}
    start_indices = []

    start_idx = 0
    for layer_index, item in enumerate(target_abs_model_update_stat_dict.items()):
        
        key, tensor = item
        flat_tensor = tensor.view(-1)
        
        if layer_index < split_layer_index:
            grad_list.append(torch.zeros_like(flat_tensor, device=device))
        else:
            grad_list.append(flat_tensor)

        key_index_ranges[key] = (start_idx, start_idx + len(flat_tensor), tensor.shape)
        start_indices.append(start_idx)
        start_idx += len(flat_tensor)

    grad_list = torch.cat(grad_list).to(device)
    _, indices = torch.topk(grad_list, int(len(grad_list) * ratio))

    start_indices = torch.tensor(start_indices, device=device)
    key_bins = torch.searchsorted(start_indices, indices, right=True) - 1 

    relative_indices = indices - start_indices[key_bins]

    sorted_indices_by_key = {key: None for key in target_abs_model_update_stat_dict}
    unique_keys = torch.unique(key_bins)

    for key_idx in unique_keys:
        key = list(key_index_ranges.keys())[key_idx.item()]
        shape = key_index_ranges[key][2] 

        mask = key_bins == key_idx
        rel_indices_for_key = relative_indices[mask]

        multi_dim_indices = torch.unravel_index(rel_indices_for_key, shape)

        sorted_indices_by_key[key] = torch.stack(multi_dim_indices, dim=1).long()

    return sorted_indices_by_key
    #return (start_indices,key_index_ranges), indices

def find_critic_neurons_by_shallow_layers(target_model_update_stat_dict, target_model_stat_dict, ratio=0.3, device='cuda', split_layer_index=11, **kwargs):
    target_abs_model_update_stat_dict = {}

    for key in target_model_update_stat_dict:
        target_abs_model_update_stat_dict[key] = torch.abs(target_model_update_stat_dict[key])

    grad_list = []
    key_index_ranges = {}
    start_indices = []

    start_idx = 0
    for layer_index, item in enumerate(target_abs_model_update_stat_dict.items()):
        
        key, tensor = item
        flat_tensor = tensor.view(-1)
        
        if layer_index >= split_layer_index:
            grad_list.append(torch.zeros_like(flat_tensor, device=device))
        else:
            grad_list.append(flat_tensor)

        key_index_ranges[key] = (start_idx, start_idx + len(flat_tensor), tensor.shape)
        start_indices.append(start_idx)
        start_idx += len(flat_tensor)

    grad_list = torch.cat(grad_list).to(device)
    _, indices = torch.topk(grad_list, int(len(grad_list) * ratio))

    start_indices = torch.tensor(start_indices, device=device)
    key_bins = torch.searchsorted(start_indices, indices, right=True) - 1 

    relative_indices = indices - start_indices[key_bins]

    sorted_indices_by_key = {key: None for key in target_abs_model_update_stat_dict}
    unique_keys = torch.unique(key_bins)

    for key_idx in unique_keys:
        key = list(key_index_ranges.keys())[key_idx.item()]
        shape = key_index_ranges[key][2] 

        mask = key_bins == key_idx
        rel_indices_for_key = relative_indices[mask]

        multi_dim_indices = torch.unravel_index(rel_indices_for_key, shape)

        sorted_indices_by_key[key] = torch.stack(multi_dim_indices, dim=1).long()

    return sorted_indices_by_key
    #return (start_indices,key_index_ranges), indices


from typing import Literal
def fedAvg_epoch_fine_tuning_while_pruning_critic_neurons(server, clients, client_datasets, test_data_loader, epochs, local_epochs,
                   neuron_selection_func: Literal["SalUn", "Deep", "Shallow", "Localized"] = "SalUn",
                   split_layer_index = 11,
                   val_data_loader = None, logger = None,
                   batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, 
                   target_client_index=None, 
                   pruning_epoches=None,
                   return_dynamic=False,
                   drop_ratio = 0.2,
                   round_drop_ratio=0.75,
                   round_random_drop=True):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    neuron_selection_func_dict = {
        "SalUn":find_critic_neurons_by_salun,
        "Deep":find_critic_neurons_by_deep_layers,
        "Shallow":find_critic_neurons_by_shallow_layers,
        "Localized":find_critic_neurons_by_local_strategy
    }

    neuron_level = (neuron_selection_func == "Localized") 

    # if stop_pruning_epoch is None:
    #     stop_pruning_epoch = int(epochs/3)

    if pruning_epoches is None:
        pruning_epoches = [0]

    epoch_data = {
        "training_acc":[],
        "training_loss":[],
        "test_acc":[],
        "test_loss":[],
        "val_acc":[],
        "val_loss":[],
        "time":[]
    }
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            if epoch_index != pruning_epoches[0] and client_index == target_client_index:
                continue
                
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        if epoch_index == pruning_epoches[0]:
            target_model_stat_dict = copy.deepcopy(global_model_state_dict)
            target_model_update_stat_dict = model_update_list[target_client_index]
            model_update_list = model_update_list[:target_client_index] + model_update_list[target_client_index+1:]
        
        server.train_model(model_update_list, epoch_index=epoch_index)

        # if epoch_index in pruning_epoches:
        #     pruning_epoch_index = pruning_epoches.index(epoch_index)
        #     if epoch_index == pruning_epoches[0]:
        #         key_info, memorziation_indices = find_critic_neurons_by_salun(target_model_update_stat_dict, ratio=drop_ratio)
                
        #         server.memorziation_indices = memorziation_indices
        #         server.key_info = key_info
        #         server.keys = target_model_update_stat_dict.keys()
                
        #     pruning_indices = select_pruning_neurons(pruning_epoch_index, server.memorziation_indices, ratio=round_drop_ratio, random_sampling=round_random_drop)
        #     pruning_indices_by_key = mapping_indices(server.key_info, pruning_indices, server.keys)
        #     unlearning_stat_dict = drop_neuron_by_indices(server.global_model, pruning_indices_by_key)
        #     server.global_model.load_state_dict(unlearning_stat_dict)

        if epoch_index in pruning_epoches:
            critic_indices_by_key = neuron_selection_func_dict[neuron_selection_func](target_model_update_stat_dict, 
                                                                                      target_model_stat_dict, 
                                                                                      ratio=drop_ratio,
                                                                                      split_layer_index=split_layer_index)
            
            unlearning_stat_dict = drop_neuron_by_indices(server.global_model, critic_indices_by_key, neuron_level)
            server.global_model.load_state_dict(unlearning_stat_dict)
        
        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        epoch_data["training_acc"].append(train_acc)
        epoch_data["training_loss"].append(train_loss)
        epoch_data["test_acc"].append(test_acc)
        epoch_data["test_loss"].append(test_loss)
        epoch_data["val_acc"].append(val_acc)
        epoch_data["val_loss"].append(val_loss)
        epoch_data["time"].append(time_elapsed)
        

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    if return_dynamic:
        return train_acc, train_loss, test_acc, test_loss, time_elapsed,epoch_data
        
    return train_acc, train_loss, test_acc, test_loss, time_elapsed


def fedAvg_epoch_fine_tuning(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, logger = None,
                   batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, 
                   target_client_index=None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            # if client_index == target_client_index:
                
            #     client_loss, client_acc, client_examples = client.training_local_dataset(
            #         local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle, loss_func = acsent_loss)
            # else:
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        
        
        server.train_model(model_update_list, epoch_index=epoch_index)

        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    return train_acc, train_loss, test_acc, test_loss, time_elapsed

def fedAvg_epoch_check(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, logger = None,
                   batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, 
                   target_client_index=None, 
                   stop_pruning_epoch=None,
                   drop_ratio=0.2):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

 
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)
    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        remaining_model_update_list = []
        target_model_update_dict = None
        
        global_model_state_dict = server.despatch()
        
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            if client_index == target_client_index:
                client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)
                target_model_update_dict = client.get_model_upate()
                
            else:
                client_loss, client_acc, client_examples = client.training_local_dataset(
                        local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

                client_update_dict = client.get_model_upate()
                remaining_model_update_list.append(client_update_dict)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
        
        remaining_gradient = {}
        remaining_average_model_update_stat_dict = update_average_model_update_stat_dict(remaining_gradient, remaining_model_update_list)
        memorziation_indices_by_key = find_memorization_neurons(remaining_average_model_update_stat_dict, ratio=drop_ratio)

            
        critic_indices_by_key = find_critic_neurons(target_model_update_dict, ratio=drop_ratio)
            
    return memorziation_indices_by_key, critic_indices_by_key

def find_random_neurons(average_model_update_stat_dict, ratio=0.3, device='cuda'):
    average_windows_abs_model_update_stat_dict = {}

    for key in average_model_update_stat_dict:
        average_windows_abs_model_update_stat_dict[key] = torch.mean(torch.stack(
            [torch.abs(update) for update in average_model_update_stat_dict[key]]), dim=0)

    grad_list = []
    key_index_ranges = {}
    start_indices = []

    start_idx = 0
    for key, tensor in average_windows_abs_model_update_stat_dict.items():
        flat_tensor = tensor.view(-1)
        grad_list.append(flat_tensor)

        key_index_ranges[key] = (start_idx, start_idx + len(flat_tensor), tensor.shape)
        start_indices.append(start_idx)
        start_idx += len(flat_tensor)

    grad_list = torch.cat(grad_list).to(device)
    indices = torch.randperm(len(grad_list), device=device)[:int(len(grad_list) * ratio)]

    # start_indices = torch.tensor(start_indices, device=device)
    # key_bins = torch.searchsorted(start_indices, indices, right=True) - 1 

    # relative_indices = indices - start_indices[key_bins]

    # sorted_indices_by_key = {key: None for key in average_model_update_stat_dict}
    # unique_keys = torch.unique(key_bins)

    # for key_idx in unique_keys:
    #     key = list(key_index_ranges.keys())[key_idx.item()]
    #     shape = key_index_ranges[key][2] 

    #     mask = key_bins == key_idx
    #     rel_indices_for_key = relative_indices[mask]

    #     multi_dim_indices = torch.unravel_index(rel_indices_for_key, shape)

    #     sorted_indices_by_key[key] = torch.stack(multi_dim_indices, dim=1).long()

    # return sorted_indices_by_key

    return (start_indices,key_index_ranges), indices

def fedAvg_epoch_fine_tuning_while_pruning_random_neurons(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, 
                                                          logger = None, 
                                                          batch_size = 128, 
                                                          lr_scheduler = False, 
                                                          run_local_dataset = True, 
                                                          shuffle=True,    
                                                          epoch_end_hook= None, 
                                                          target_client_index=None, 
                                                          pruning_epoches=None, 
                                                          return_dynamic=False,
                                                          drop_ratio=0.2,
                                                          round_drop_ratio=0.75,
                                                          round_random_drop=True):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    server.average_model_update_stat_dict = {}
    # if stop_pruning_epoch is None:
    #     stop_pruning_epoch = int(epochs/3)
    if pruning_epoches is None:
        pruning_epoches = [0]
    
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    epoch_data = {
        "training_acc":[],
        "training_loss":[],
        "test_acc":[],
        "test_loss":[],
        "val_acc":[],
        "val_loss":[],
        "time":[]
    }

    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            # if client_index == target_client_index:
                
            #     client_loss, client_acc, client_examples = client.training_local_dataset(
            #         local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle, loss_func = acsent_loss)
            # else:
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)
        
        server.train_model(model_update_list, epoch_index=epoch_index)

        # if epoch_index in pruning_epoches:
        #     average_model_update_stat_dict = update_average_model_update_stat_dict(server.average_model_update_stat_dict, model_update_list)
        #     random_indices_by_key = find_random_neurons(average_model_update_stat_dict, ratio=drop_ratio)
        #     unlearning_stat_dict = drop_neuron_by_indices(server.global_model, random_indices_by_key)
        #     server.global_model.load_state_dict(unlearning_stat_dict)

        if epoch_index in pruning_epoches:
            pruning_epoch_index = pruning_epoches.index(epoch_index)
            if epoch_index == pruning_epoches[0]:
                average_model_update_stat_dict = update_average_model_update_stat_dict(server.average_model_update_stat_dict, model_update_list)
                key_info, memorziation_indices = find_random_neurons(average_model_update_stat_dict, ratio=drop_ratio)
                
                server.memorziation_indices = memorziation_indices
                server.key_info = key_info
                server.keys = average_model_update_stat_dict.keys()
                
            pruning_indices = select_pruning_neurons(pruning_epoch_index, server.memorziation_indices, ratio=round_drop_ratio, random_sampling=round_random_drop)
            pruning_indices_by_key = mapping_indices(server.key_info, pruning_indices, server.keys)
            unlearning_stat_dict = drop_neuron_by_indices(server.global_model, pruning_indices_by_key)
            server.global_model.load_state_dict(unlearning_stat_dict)

        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
        else:
            val_acc, val_loss = 0., 0.
        end = time.time()
        time_elapsed = end-begin

        epoch_data["training_acc"].append(train_acc)
        epoch_data["training_loss"].append(train_loss)
        epoch_data["test_acc"].append(test_acc)
        epoch_data["test_loss"].append(test_loss)
        epoch_data["val_acc"].append(val_acc)
        epoch_data["val_loss"].append(val_loss)
        epoch_data["time"].append(time_elapsed)

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    if return_dynamic:
        return train_acc, train_loss, test_acc, test_loss, time_elapsed,epoch_data

    return train_acc, train_loss, test_acc, test_loss, time_elapsed




def fedAvg_retraining_epoch(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loaders = None, val_data_labels = None, logger = None,
                 batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client in clients:
            client.update_local_model(global_model_state_dict)
            
            client_loss, client_acc, client_examples = client.training_local_dataset(
                local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)

        server.train_model(model_update_list)


        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)

        val_res = []
        if val_data_loaders is not None:
            for val_data_loader in val_data_loaders:
                val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
                val_res.append((val_acc, val_loss))
        
        end = time.time()
        time_elapsed = end-begin

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))

        for val_idx in range(len(val_res)):
            val_acc, val_loss = val_res[val_idx]
            print("Val_Label:{}, Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_data_labels[val_idx], val_acc, val_loss))
                
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            
            for val_idx in range(len(val_res)):
                val_acc, val_loss = val_res[val_idx]
                logger.info("Val_Label:{}, Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_data_labels[val_idx], val_acc, val_loss))
            
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    return train_acc, train_loss, test_acc, test_loss, time_elapsed



def fedAvg_epoch_fine_tuning_while_pruning_redundant_neurons_for_multiple_clients(server, clients, client_datasets, test_data_loader, epochs, local_epochs,
                   target_client_indices,
                   val_data_loaders = None, 
                   val_data_labels = None, 
                   logger = None,
                   batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None, 
                   pruning_epoches=None,
                   return_dynamic = False,
                   drop_ratio=0.2,
                   round_drop_ratio=0.75,
                   round_random_drop=True):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))

    server.average_model_update_stat_dict = {}
    if pruning_epoches is None:
        pruning_epoches = [0]
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    epoch_data = {
        "training_acc":[],
        "training_loss":[],
        "test_acc":[],
        "test_loss":[],
        "val_acc":[],
        "val_loss":[],
        "val_res":[],
        "time":[]
    }
    
    for epoch_index in range(epochs):
        begin = time.time()
        
        total_loss, total_acc, total_examples =0., 0., 0
        
        model_update_list = []
        global_model_state_dict = server.despatch()
    
        for client_index, client in enumerate(clients):
            client.update_local_model(global_model_state_dict)

            # if client_index == target_client_index:
                
            #     client_loss, client_acc, client_examples = client.training_local_dataset(
            #         local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle, loss_func = acsent_loss)
            # else:
            client_loss, client_acc, client_examples = client.training_local_dataset(
                    local_epochs, batch_size = batch_size, lr_scheduler = lr_scheduler, shuffle = shuffle)

            total_loss += client_loss
            total_acc += client_acc
            total_examples += client_examples
            
            client_update_dict = client.get_model_upate()
            
            model_update_list.append(client_update_dict)
        
        server.train_model(model_update_list, epoch_index=epoch_index)

        if epoch_index in pruning_epoches:
            pruning_epoch_index = pruning_epoches.index(epoch_index)
            if epoch_index == pruning_epoches[0]:
                average_model_update_stat_dict = update_average_model_update_stat_dict(server.average_model_update_stat_dict, model_update_list)
                key_info, memorziation_indices = find_memorization_neurons(average_model_update_stat_dict, ratio=drop_ratio)
                
                server.memorziation_indices = memorziation_indices
                server.key_info = key_info
                server.keys = average_model_update_stat_dict.keys()
                
            pruning_indices = select_pruning_neurons(pruning_epoch_index, server.memorziation_indices, ratio=round_drop_ratio, random_sampling=round_random_drop)
            pruning_indices_by_key = mapping_indices(server.key_info, pruning_indices, server.keys)
            unlearning_stat_dict = drop_neuron_by_indices(server.global_model, pruning_indices_by_key)
            server.global_model.load_state_dict(unlearning_stat_dict)

        train_acc = total_acc/total_examples
        train_loss = total_loss/total_examples
    
        test_acc, test_loss = epoch_test(test_data_loader, server.global_model)
        
        val_res = []
        if val_data_loaders is not None:
            for val_data_loader in val_data_loaders:
                val_acc, val_loss = epoch_test(val_data_loader, server.global_model)
                val_res.append((val_acc, val_loss))
                
        end = time.time()
        time_elapsed = end-begin

        epoch_data["training_acc"].append(train_acc)
        epoch_data["training_loss"].append(train_loss)
        epoch_data["test_acc"].append(test_acc)
        epoch_data["test_loss"].append(test_loss)
        epoch_data["val_res"].append(val_res)
        epoch_data["time"].append(time_elapsed)

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        
        for val_idx in range(len(val_res)):
            val_acc, val_loss = val_res[val_idx]
            print("Val_Label:{}, Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_data_labels[val_idx], val_acc, val_loss))
            
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            
            for val_idx in range(len(val_res)):
                val_acc, val_loss = val_res[val_idx]
                logger.info("Val_Label:{}, Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_data_labels[val_idx], val_acc, val_loss))
                
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if epoch_end_hook is not None:
            epoch_end_hook()

    if return_dynamic:
        return train_acc, train_loss, test_acc, test_loss, time_elapsed,epoch_data

    return train_acc, train_loss, test_acc, test_loss, time_elapsed