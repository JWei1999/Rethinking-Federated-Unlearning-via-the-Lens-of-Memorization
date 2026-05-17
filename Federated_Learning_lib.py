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


"""
-Server
-Client
-FedSGD Framework
"""


class Server:
    def __init__(self, model_generator, device='cuda'):
        self.global_model = model_generator().to(device)
        self.client_num = None
        self.aggregate_rule = Server.aggregate_parameter_standardly
        self.preprocess_updates = None
    
    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def get_client_num(self):
        return self.client_num

    def get_weights(self):
        model = self.global_model
        return [p.detach().clone() for p in model.parameters()]

    def save_model(self, path):
        torch.save(self.global_model.state_dict(), path)

    def load_model(self, path):
        self.global_model.load_state_dict(torch.load(path))

    def set_aggregate_rule(self, aggregate_rule):
        self.aggregate_rule = aggregate_rule

    def set_preprocess_updates(self, preprocess_function):
        self.preprocess_updates = preprocess_function
    
    def train_model(self, model_update_list, preprocess = False, epoch_index = -1):
        if len(model_update_list) == 0:
            return

        if self.preprocess_updates is not None and preprocess:
            baseline_global_model_state_dict = copy.deepcopy(self.global_model.state_dict())
            model_update_list = self.preprocess_updates(baseline_global_model_state_dict, model_update_list)

        self.client_num = len(model_update_list)

        average_model_update, user_index = self.aggregate_rule(model_update_list,return_candidx=True, epoch_index=epoch_index)

        if len(user_index) == 0:
            return

        for parameter, update in zip(self.global_model.parameters(), average_model_update):
        
            parameter.data += update.data

    def train_batch_normalization(self, model_batch_normlization_list):
        aggregated_batchnorm_info = Server.aggregate_batchnorm(model_batch_normlization_list)

        bn_layers = self.global_model.find_bn_layers()
        for layer_index, bn_layer in enumerate(bn_layers):
            running_mean, running_var = aggregated_batchnorm_info[layer_index]
            
            bn_layer.running_mean.data.copy_(running_mean)
            bn_layer.running_var.data.copy_(running_var)

            
    
    
    def despatch(self):
        self.global_model.zero_grad()
        return copy.deepcopy(self.global_model.state_dict())

    def update_model(self, update_gradient):
        for parameter, name in zip(self.global_model.parameters(), update_gradient):
            parameter.data += update_gradient[name].data

    def aggregate_parameter_standardly(model_update_list, return_candidx=True, **kwargs):
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

        if return_candidx:
            candidate_indices = list(range(len(model_update_list)))
            return average_model_update, candidate_indices
        else:
            return average_model_update
    
    def aggregate_batchnorm(model_batch_normlization_list):
        aggregated_batch_info_list = []
        for layers in zip(*model_batch_normlization_list):
            # layers is a tuple containing the same layer from all clients
            means, vars_ = zip(*layers)  # Separate means and vars
    
            # Stack and compute the mean for means and vars
            avg_mean = torch.stack(means).mean(dim=0)
            avg_var = torch.stack(vars_).mean(dim=0)
    
            # Append the aggregated [mean, var] for this layer
            aggregated_batch_info_list.append([avg_mean, avg_var])
       
        # print(len(aggregated_batch_info_list))
        # print(len(aggregated_batch_info_list[0]))
        return aggregated_batch_info_list

    
    def get_weight(self):
        model = self.global_model
        return [p.detach().clone() for p in model.parameters()]

    
# Client
class Client:
    def __init__(self, model_generator, optimizer_type, optimizer_args, optimizer_scheduler_type=None, optimizer_scheduler_args=None, client_index=None, init_optimizer=None, loss_coef=[1.,1.,1.], device='cuda'):
        self.model_generator = model_generator
        self.local_model = model_generator().to(device)

        self.reference_model = model_generator().to(device)
        
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.optimizer_scheduler_type = optimizer_scheduler_type
        self.optimizer_scheduler_args = optimizer_scheduler_args
        
        self.init_optimizer = init_optimizer
        
        if self.init_optimizer is None:
            self.optimizer = optimizer_type(
                self.local_model.parameters(), **optimizer_args)
            
            if optimizer_scheduler_type is not None:
                self.optimizer_scheduler = optimizer_scheduler_type(
                    self.optimizer, **optimizer_scheduler_args
                )
        else:
            self.init_optimizer(self, self.local_model)
    
        #self.local_dataset = local_dataset
        self.client_index = client_index
        self.loss_coef = loss_coef
    
    def set_local_dataset(self, local_dataset):
        self.local_dataset = local_dataset
    
    def set_server(self, server):
        self.server = server

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def set_optimizer(self, optimizer_type, optimizer_args, optimizer_scheduler_type=None, optimizer_scheduler_args=None):
        self.optimizer = optimizer_type(
            self.local_model.parameters(), **optimizer_args)
        
        if optimizer_scheduler_type is not None:
            self.optimizer_scheduler = optimizer_scheduler_type(
                self.optimizer, **optimizer_scheduler_args
            )
            
    def step_scheduler(self):
        if self.optimizer_scheduler is not None:
            self.optimizer_scheduler.step()

    def get_weights(self):
        model = self.local_model
        return [p.detach().clone() for p in model.parameters()]

    def save_model(self, path):
        torch.save(self.local_model.state_dict(), path)

    def load_model(self, path):
        self.local_model.load_state_dict(torch.load(path))

    def update_local_model(self, global_model_state_dict):
        self.baseline_model_state_dict = global_model_state_dict
        self.local_model.load_state_dict(global_model_state_dict)

        if self.init_optimizer is None:
            self.optimizer = self.optimizer_type(
                self.local_model.parameters(), **self.optimizer_args)
            
            if self.optimizer_scheduler_type is not None:
                self.optimizer_scheduler = self.optimizer_scheduler_type(
                    self.optimizer, **self.optimizer_scheduler_args
                )
        else:
            self.init_optimizer(self, self.local_model)

    def create_local_dataloader(self,batch_size):
        self.local_dataloader = InfiniteDataLoader(DataLoader(self.local_dataset, batch_size=batch_size, shuffle=True))

    def delete_local_dataloader(self):
        del(self.local_dataloader)
    
    def training_batches(self, local_epochs, batch_size, device='cuda'):
        self.local_model.train()
        
        model = self.local_model
        model.zero_grad()

        total_loss, total_acc, total_examples = 0., 0., 0
        
        for epoch in range(local_epochs):
            X, y = next(self.local_dataloader)
            
            X, y = X.to(device), y.to(device)
            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            size = X.shape[0]
            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * size
            total_examples += size
        
        return total_loss, total_acc, total_examples

    def training_local_dataset(self, local_epochs, batch_size, lr_scheduler, shuffle=True, loss_func=nn.CrossEntropyLoss(), device='cuda'):
        self.local_model.train()
        
        model = self.local_model
        model.zero_grad()

        total_loss, total_acc, total_examples = 0., 0., 0
        
        if len(self.local_dataset) == 0:
            return total_loss, total_acc, total_examples
        
        loader = DataLoader(self.local_dataset, batch_size=batch_size, shuffle=shuffle)
        
        for epoch in range(local_epochs):
            
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                yp = model(X)
                loss = loss_func(yp, y)
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                size = X.shape[0]
                total_acc += (yp.max(dim=1)[1] == y).sum().item()
                total_loss += loss.item() * size
                total_examples += size

            if lr_scheduler:
                self.step_scheduler()
                
            # if lr_scheduler:
            #     self.step_scheduler()
        
        return total_loss, total_acc, total_examples

    # def training_local_dataset(self, local_epochs, batch_size, lr_scheduler, shuffle=True, device='cuda'):
    #     self.local_model.train()
    #     self.reference_model.eval()
        
    #     self.reference_model.load_state_dict(self.local_model.state_dict())
        
    #     model = self.local_model
    #     model.zero_grad()

    #     reference_model = self.reference_model
    #     reference_model.zero_grad()
        

    #     total_loss, total_acc, total_examples = 0., 0., 0

    #     if len(self.local_dataset) == 0:
    #         return total_loss, total_acc, total_examples
        
    #     loader = DataLoader(self.local_dataset, batch_size=batch_size, shuffle=shuffle)
        
    #     for epoch in range(local_epochs):
            
    #         for X, y in loader:
    #             X, y = X.to(device), y.to(device)
    #             yp, grouped_output = model(X, grouped=True)
            
    #             classification_loss = nn.CrossEntropyLoss()(yp, y)
    
    #             ref_yp, ref_grouped_output = reference_model(X, grouped=True)
                
    #             generalization_feature_loss = infoNCE_loss(grouped_output, ref_grouped_output, -1)
    #             client_feature_loss = infoNCE_loss(grouped_output, ref_grouped_output, self.client_index)
    
    #             a = self.loss_coef
                
    #             loss = a[0] * classification_loss + a[1] * generalization_feature_loss + a[2]* client_feature_loss

    #             #print(f"Losses-Classification: {classification_loss:.4f}, Generalization: {generalization_feature_loss:.4f}, Client:{client_feature_loss:.4f}")
                
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             size = X.shape[0]
    #             total_acc += (yp.max(dim=1)[1] == y).sum().item()
    #             total_loss += loss.item() * size
    #             total_examples += size

    #             if lr_scheduler:
    #                 self.step_scheduler()
                
    #         # if lr_scheduler:
    #         #     self.step_scheduler()
        
    #     return total_loss, total_acc, total_examples
    

    def training_one_batch(self, X, y, device='cuda'):
        self.local_model.train()
        
        model = self.local_model
        model.zero_grad()

        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_model_upate(self):
        current_state_dict = self.local_model.state_dict()
        update_dict = {}

        def check_key(key):
            flag = True
            words = ['bn', 'running', 'batch']
            for word in words:
                if word in key:
                    return False
            return flag
        
        for key in current_state_dict:
            if check_key(key):
                update_dict[key] = current_state_dict[key] - self.baseline_model_state_dict[key]
        
        return update_dict

    def get_group_normalization(self):
        self.gn_layer_list = []
        find_gn_in_model(self.global_model, self.gn_layer_list)

        group_normalization_list = []
        for gn_layer in self.gn_layer_list:
            group_normalization_list.append({
                'weight': gn_layer.weight.data,
                'bias': gn_layer.bias.data
            })

        return group_normalization_list
        
        
    def cal_gradient(self, X, y, device='cuda'):
        model = self.local_model
        model.eval()

        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        gradient = torch.autograd.grad(loss, model.parameters())
        gradient = [grad for grad in gradient]

        return gradient

    def get_weight(self):
        model = self.local_model
        return [p.detach().clone() for p in model.parameters()]

    def reset(self, reset_state_dict):
        self.local_model.load_state_dict(reset_state_dict)

        
        
        
def fedSGD_epoch(server, clients, train_loaders, test_data_loader):
    loader_size = len(train_loaders[0])
    dataset_size = len(train_loaders[0].dataset)
    client_num = len(clients)

    begin = time.time()
    total_acc, total_loss = 0., 0.
    train_iteration_loaders = [iter(dataloader)
                               for dataloader in train_loaders]

    for client in clients:
        client.set_server(server)

    for index in range(0, loader_size):
        gradient_list = []
        bn_data_list = []
        clients_acc, clients_loss = 0., 0.
        for client, train_loader in zip(clients, train_iteration_loaders):
            X, y = next(train_loader)
            gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(
                X, y)
            gradient_list.append(gradient)
            bn_data_list.append(bn_data)
            clients_acc += accuracy
            clients_loss += loss_value

        total_acc += clients_acc/client_num
        total_loss += clients_loss/client_num

        server.train_sgd_batch(gradient_list, bn_data_list)
        global_model_state_dict = server.despatch()

        for client in clients:
            client.update_local_model(global_model_state_dict)

        #print("Update Normally")
    train_acc = total_acc/dataset_size
    train_loss = total_loss/dataset_size
    test_acc, test_loss = server.test(test_data_loader)
    end = time.time()
    time_elapsed = end-begin

    # print("-------------Epoch: %d--------------" % epoch_index)
    # print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
    # print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return train_acc, train_loss, test_acc, test_loss, time_elapsed



def fedAvg_epoch(server, clients, client_datasets, test_data_loader, epochs, local_epochs, val_data_loader = None, logger = None,
                 batch_size = 128, lr_scheduler = False, run_local_dataset = True, shuffle=True, epoch_end_hook= None):

    client_num = len(clients)
    dataset_size = len(client_datasets[0])
    loader_loop = int(dataset_size/(batch_size * local_epochs))
    
    for client, client_dataset in zip(clients, client_datasets):
        client.set_server(server)
        client.set_local_dataset(client_dataset)

    if run_local_dataset:
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


    else:
        for client in clients:
            client.create_infinite_dataloader(client.local_datset, batch_size)
            
        for epoch_index in range(epochs):
            begin = time.time()
            
            total_loss, total_acc, total_examples =0., 0., 0
    
                
            for _ in range(loader_loop):
                model_update_list = []
                global_model_state_dict = server.despatch()
                
                for client in clients:
                    client.update_local_model(global_model_state_dict)
                    
                    client_loss, client_acc, client_examples = client.training_batches(
                        local_epochs, batch_size)
        
                    total_loss += client_loss
                    total_acc += client_acc
                    total_examples += client_examples
                    
                    client_update_dict = client.get_model_upate()
                    
                    model_update_list.append(client_update_dict)
        
                server.train_model(model_update_list)

            if lr_scheduler:
                for client in clients:
                    client.step_scheduler()
    
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

    return train_acc, train_loss, test_acc, test_loss, time_elapsed

#Standard
def aggregate_standard(logger,ifprint=True):
    def _aggregate_grads(gradient_list, n_attackers, mal_client_idx = -1, return_candidx = False):
        size = len(gradient_list[0])
        gradients = {index: [] for index in range(0, size)}

        for gradient in gradient_list:
            for index, grad in enumerate(gradient):
                gradients[index].append(grad)

        avg_gradients = []
        for index in gradients.keys():
            avg_gradient_layer = torch.mean(torch.stack(gradients[index]), dim=0)
            avg_gradients.append(avg_gradient_layer)
        
        candidate_indices = list(range(len(gradient_list)))
        if return_candidx:
            return avg_gradients, candidate_indices
        return avg_gradients
    
    return _aggregate_grads