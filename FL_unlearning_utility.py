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

from Federated_Learning_lib import *


def measure_loss(loader, model, device='cuda'):
    with torch.no_grad():
        model.eval()
        examples_loss = torch.empty(0).to(device)
    
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss(reduction='none')(yp, y)
    
            examples_loss = torch.cat((examples_loss,loss),dim=0)
    
        return examples_loss

def measure_memorization_scores(within_model,without_model,dataset):

    loader = DataLoader(dataset, batch_size = 128, shuffle = False)

    baseline_examples_prob = epoch_test_probability(loader, within_model, only_label=True)
    exclude_examples_prob = epoch_test_probability(loader, without_model, only_label=True)

    memorization_scores = baseline_examples_prob - exclude_examples_prob

    return memorization_scores

class RetrainingMeasurement:
    def __init__(self, retraining_model, test_dataset, target_dataset, high_memorization_dataset, low_memorization_dataset, model_generator=None, logger=None):
        self.retraining_model = retraining_model
        self.target_dataset = target_dataset
        self.test_dataset = test_dataset
        self.high_memorization_dataset = high_memorization_dataset
        self.low_memorization_dataset = low_memorization_dataset

        self.test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)
        self.target_loader = DataLoader(target_dataset, batch_size = 128, shuffle = False)
        self.high_memorization_loader = DataLoader(high_memorization_dataset, batch_size = 128, shuffle = False)
        self.low_memorization_loader = DataLoader(low_memorization_dataset, batch_size = 128, shuffle = False)

        self.model_generator = model_generator
        self.logger = logger
    
    def retraining_eval(self):
        model = self.retraining_model
        target_loader = self.target_loader
        test_loader = self.test_loader
        high_memorization_loader = self.high_memorization_loader
        low_memorization_loader = self.low_memorization_loader
        logger = self.logger
        
        val_acc, val_loss = epoch_test(target_loader, model)
        val_hms_acc, val_hms_loss = epoch_test(high_memorization_loader, model)
        val_lms_acc, val_lms_loss = epoch_test(low_memorization_loader, model)

        test_acc, test_loss = epoch_test(test_loader, model)
        
        self.retraining_loss_list = measure_loss(target_loader, model)
        self.retraining_acc_list = epoch_test_each_example(target_loader, model)

        self.retraining_high_mem_loss_list = measure_loss(high_memorization_loader, model)
        self.retraining_high_mem_acc_list = epoch_test_each_example(high_memorization_loader, model)
        self.retraining_low_mem_loss_list = measure_loss(low_memorization_loader, model)
        self.retraining_low_mem_acc_list = epoch_test_each_example(low_memorization_loader, model)

        print("Model:{}".format("Retraining Model"))
        print("Test Acc:{:.4f}, Test Loss:{:.4f}".format(test_acc, test_loss))
        print("Validation Acc:{:.4f}, Validation Loss:{:.4f}".format(val_acc, val_loss))
        print("Validation High Memorization Acc:{:.4f}, Validation High Memorization Loss:{:.4f}".format(val_hms_acc, val_hms_loss))
        print("Validation Low Memorization Acc:{:.4f}, Validation Low Memorization Loss:{:.4f}".format(val_lms_acc, val_lms_loss))
        
        if logger is not None:
            logger.info("Model:{}".format("Retraining Model"))
            logger.info("Test Acc:{:.4f}, Test Loss:{:.4f}".format(test_acc, test_loss))
            logger.info("Validation Acc:{:.4f}, Validation Loss:{:.4f}".format(val_acc, val_loss))
            logger.info("Validation High Memorization Acc:{:.4f}, Validation High Memorization Loss:{:.4f}".format(val_hms_acc, val_hms_loss))
            logger.info("Validation Low Memorization Acc:{:.4f}, Validation Low Memorization Loss:{:.4f}".format(val_lms_acc, val_lms_loss))

    def compare_with_retraining(self, compared_model, model_label = "Compared Model"):
        target_loader = self.target_loader
        test_loader = self.test_loader
        high_memorization_loader = self.high_memorization_loader
        low_memorization_loader = self.low_memorization_loader
        logger = self.logger

        retraining_loss_list = self.retraining_loss_list
        retraining_acc_list = self.retraining_acc_list
        
        retraining_high_mem_loss_list = self.retraining_high_mem_loss_list
        retraining_high_mem_acc_list =  self.retraining_high_mem_acc_list
        retraining_low_mem_loss_list = self.retraining_low_mem_loss_list
        retraining_low_mem_acc_list = self.retraining_low_mem_acc_list
        
       
        compared_acc_list = epoch_test_each_example(target_loader, compared_model)
        compared_loss_list = measure_loss(target_loader, compared_model)
        
        compared_high_mem_acc_list = epoch_test_each_example(high_memorization_loader, compared_model)
        compared_high_mem_loss_list = measure_loss(high_memorization_loader, compared_model)
        compared_low_mem_acc_list = epoch_test_each_example(low_memorization_loader, compared_model)
        compared_high_mem_loss_list = measure_loss(low_memorization_loader, compared_model)

        val_acc, val_loss = epoch_test(target_loader, compared_model)
        val_hms_acc, val_hms_loss = epoch_test(high_memorization_loader, compared_model)
        val_lms_acc, val_lms_loss = epoch_test(low_memorization_loader, compared_model)

        test_acc, test_loss = epoch_test(test_loader, compared_model)

        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        target_confusion_matrix = confusion_matrix(retraining_acc_list, compared_acc_list)
        
        high_mem_confusion_matrix = confusion_matrix(retraining_high_mem_acc_list, compared_high_mem_acc_list)
        low_mem_confusion_matrix = confusion_matrix(retraining_low_mem_acc_list, compared_low_mem_acc_list)

        print("Model:{}".format(model_label))
        print("Test Acc:{:.4f}, Test Loss:{:.4f}".format(test_acc, test_loss))
        print("Validation Acc:{:.4f}, Validation Loss:{:.4f}".format(val_acc, val_loss))
        print("Validation High Memorization Acc:{:.4f}, Validation High Memorization Loss:{:.4f}".format(val_hms_acc, val_hms_loss))
        print("Validation Low Memorization Acc:{:.4f}, Validation Low Memorization Loss:{:.4f}".format(val_lms_acc, val_lms_loss))

        if logger is not None:
            logger.info("Model:{}".format(model_label))
            logger.info("Test Acc:{:.4f}, Test Loss:{:.4f}".format(test_acc, test_loss))
            logger.info("Validation Acc:{:.4f}, Validation Loss:{:.4f}".format(val_acc, val_loss))
            logger.info("Validation High Memorization Acc:{:.4f}, Validation High Memorization Loss:{:.4f}".format(val_hms_acc, val_hms_loss))
            logger.info("Validation Low Memorization Acc:{:.4f}, Validation Low Memorization Loss:{:.4f}".format(val_lms_acc, val_lms_loss))
            

        self.measure_unlearning("Target Dataset", target_confusion_matrix, logger)
        self.measure_unlearning("High Mem Dataset", high_mem_confusion_matrix, logger)
        self.measure_unlearning("Low Mem Dataset", low_mem_confusion_matrix, logger)

    def compare_with_retraining_based_state_dict(self, compared_model_state_dict, model_label="Compared Model", device='cuda'):
        model_generator = self.model_generator
        
        compared_model = model_generator().to(device)
        compared_model.load_state_dict(compared_model_state_dict)
        
        self.compare_with_retraining(compared_model, model_label)

        del compared_model

    def measure_unlearning(self, label, confusion_matrix, logger):
        tn, fp, fn, tp = confusion_matrix.ravel()
        counts = (tn + fp + fn + tp)

        print(f"{label}")
        
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        
        print(f"Consistency Rate (TP+TN)/SUM: {((tp+tn)/counts):.4f}")
        print(f"Unlearning Failure Rate (FP)/SUM: {((fp)/counts):.4f}")
        print(f"Generaliation Loss Rate (FN)/SUM: {((fn)/counts):.4f}")
    
        if logger is not None:
            logger.info(f"{label}")
            
            logger.info(f"True Positives (TP): {tp}")
            logger.info(f"False Positives (FP): {fp}")
            logger.info(f"True Negatives (TN): {tn}")
            logger.info(f"False Negatives (FN): {fn}")
            
            logger.info(f"Consistency Rate (TP+TN)/SUM: {((tp+tn)/counts):.4f}")
            logger.info(f"Unlearning Failure Rate (FP)/SUM: {((fp)/counts):.4f}")
            logger.info(f"Generaliation Loss Rate (FP)/SUM: {((fn)/counts):.4f}")
        

class GeneralMeasurement:
    def __init__(self, test_data_loader, val_data_loader=None, logger = None):
        self.logger = logger
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader
        
        self.total_training_acc = 0.
        self.total_training_loss = 0.
        self.total_examples = 0
        
    def record(self, training_acc, training_loss, exampels):
        self.total_training_acc += training_acc
        self.total_training_loss += training_loss
        self.total_examples += exampels

    def epoch_start(self):
        self.total_training_acc = 0.
        self.total_training_loss = 0.
        self.total_examples = 0

        self.begin = time.time()
        
    def epoch_end(self):
        self.end = time.time()
    
    def epoch_evaluate(self,model,epoch_index):
        time_elapsed = self.end - self.begin
        test_data_loader = self.test_data_loader
        val_data_loader = self.val_data_loader

        train_acc = self.total_training_acc/self.total_examples
        train_loss = self.total_training_loss/self.total_examples
        test_acc, test_loss = epoch_test(test_data_loader, model)

        logger = self.logger
        
        if val_data_loader is not None:
            val_acc, val_loss = epoch_test(val_data_loader, model)
    

        print("-------------Epoch: %d--------------" % epoch_index)
        print("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
        print("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
        
        if val_data_loader is not None:
            print("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss)) 
        
        print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        if logger is not None:
            logger.info("-------------Epoch: {}--------------".format(epoch_index))
            logger.info("Train_Acc: {:.6f} ,Train_Loss: {:.6f}".format(train_acc, train_loss))
            logger.info("Test_Acc: {:.6f} ,Test_Loss: {:.6f}".format(test_acc, test_loss))
            
            if val_data_loader is not None:
                logger.info("Val_Acc: {:.6f} ,Val_Loss: {:.6f}".format(val_acc, val_loss))
                
            logger.info("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

class MemorizationMeasurement:
    def __init__(self, target_dataset, memorization_scores, bounds=None, model_generator=None, logger=None):
        
        self.target_dataset = target_dataset
        self.memorization_scores = memorization_scores
        self.bounds = bounds

        self.target_loader = DataLoader(target_dataset, batch_size = 128, shuffle = False)

        self.model_generator = model_generator
        self.logger = logger

    def compute_memorization_accuries(self, model, bounds=None):
        memorization_scores = self.memorization_scores
        
        target_acc_list = epoch_test_each_example(self.target_loader, model)
        target_acc_list = np.array(target_acc_list)

        high_memorization_scores_accuries = []
        low_memorization_scores_accuries = []

        if bounds is None:
            bounds = self.bounds
            
        for bound in bounds:
            threshold = np.percentile(memorization_scores, bound)

            high_memorization_scores_indices = np.where(memorization_scores > threshold)[0]
            low_memorization_scores_indices = np.where(memorization_scores < threshold)[0] 
            
            high_memorization_scores_acc_list = target_acc_list[high_memorization_scores_indices]
            low_memorization_scores_acc_list = target_acc_list[low_memorization_scores_indices]

            high_memorization_scores_accuries.append(np.sum(high_memorization_scores_acc_list)/len(high_memorization_scores_acc_list))
            low_memorization_scores_accuries.append(np.sum(low_memorization_scores_acc_list)/len(low_memorization_scores_acc_list))

        return (high_memorization_scores_accuries, low_memorization_scores_accuries)

    def compute_memorization_accuries_by_groups(self, model, bounds=None):
        memorization_scores = self.memorization_scores
        target_acc_list = epoch_test_each_example(self.target_loader, model)
        target_acc_list = np.array(target_acc_list)
    
        group_accuracies = []
    
        if bounds is None:
            bounds = self.bounds
    
        # Ensure bounds are sorted in descending order
        bounds = sorted(bounds, reverse=True)
    
        for i in range(len(bounds) - 1):
            upper = np.percentile(memorization_scores, bounds[i])
            lower = np.percentile(memorization_scores, bounds[i + 1])

            # print(i)
            # print(upper)
            # print(lower)
            
            # Get indices for memorization scores in this percentile range
            group_indices = np.where((memorization_scores <= upper) & (memorization_scores > lower))[0]
    
            if len(group_indices) == 0:
                group_accuracy = 1.
            else:
                group_accs = target_acc_list[group_indices]
                group_accuracy = np.sum(group_accs) / len(group_accs)
    
            group_accuracies.append(group_accuracy)
    
        return group_accuracies

    def compute_memorization_average_accuries(self, models, bounds=None):
        memorization_scores = self.memorization_scores
    
        all_high_memorization_scores_accuries = []
        all_low_memorization_scores_accuries = []
    
        if bounds is None:
            bounds = self.bounds
    
        for model in models:
            target_acc_list = epoch_test_each_example(self.target_loader, model)
            target_acc_list = np.array(target_acc_list)
    
            high_memorization_scores_accuries = []
            low_memorization_scores_accuries = []
    
            for bound in bounds:
                threshold = np.percentile(memorization_scores, bound)
    
                high_memorization_scores_indices = np.where(memorization_scores > threshold)[0]
                low_memorization_scores_indices = np.where(memorization_scores < threshold)[0] 
                
                high_memorization_scores_acc_list = target_acc_list[high_memorization_scores_indices]
                low_memorization_scores_acc_list = target_acc_list[low_memorization_scores_indices]
    
                high_memorization_scores_accuries.append(np.sum(high_memorization_scores_acc_list)/len(high_memorization_scores_acc_list))
                low_memorization_scores_accuries.append(np.sum(low_memorization_scores_acc_list)/len(low_memorization_scores_acc_list))

            all_high_memorization_scores_accuries.append(high_memorization_scores_accuries)
            all_low_memorization_scores_accuries.append(low_memorization_scores_accuries)

        avg_high_memorization_scores_accuries = np.mean(all_high_memorization_scores_accuries, axis=0)
        avg_low_memorization_scores_accuries = np.mean(all_low_memorization_scores_accuries, axis=0)
    
        return (avg_high_memorization_scores_accuries, avg_low_memorization_scores_accuries)

    def compare_consistency(self, retraining_model, unlearning_model, bounds=None):
        memorization_scores = self.memorization_scores
        
        retraining_target_acc_list = epoch_test_each_example(self.target_loader, retraining_model)
        unlearning_target_acc_list = epoch_test_each_example(self.target_loader, unlearning_model)

        retraining_target_acc_list = np.array(retraining_target_acc_list)
        unlearning_target_acc_list = np.array(unlearning_target_acc_list)

        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        target_confusion_matrix = confusion_matrix(retraining_target_acc_list, unlearning_target_acc_list)

        target_consistency_result = self.compare_consistency_with_confusion_matrix(target_confusion_matrix)

        memorization_consistency_results = []
        
        if bounds is None:
            bounds = self.bounds
            
        for bound in bounds:
            threshold = np.percentile(memorization_scores, bound)

            high_memorization_scores_indices = np.where(memorization_scores > threshold)[0]
            low_memorization_scores_indices = np.where(memorization_scores < threshold)[0] 
            
            retraining_high_memorization_scores_acc_list = retraining_target_acc_list[high_memorization_scores_indices]
            retraining_low_memorization_scores_acc_list = retraining_target_acc_list[low_memorization_scores_indices]

            unlearning_high_memorization_scores_acc_list = unlearning_target_acc_list[high_memorization_scores_indices]
            unlearning_low_memorization_scores_acc_list = unlearning_target_acc_list[low_memorization_scores_indices]

            memorization_consistency_results.append(list(self.compare_consistency_with_confusion_matrix(target_confusion_matrix)))
                
        return memorization_consistency_results

    def compare_consistency_with_confusion_matrix(self, confusion_matrix):
        tn, fp, fn, tp = confusion_matrix.ravel()
        counts = (tn + fp + fn + tp)

        consistency_rate = (tp+tn)/counts
        unlearning_failure_rate = (fp)/counts
        generalization = (fn)/counts
    

        return (consistency_rate, unlearning_failure_rate, generalization)

    def compare_consistency_based_state_dict(self, retraining_model_state_dict, unlearning_model_state_dict, bounds=None, device='cuda'):
        model_generator = self.model_generator
        
        retraining_model = model_generator().to(device)
        retraining_model.load_state_dict(retraining_model_state_dict)

        unlearning_model = model_generator().to(device)
        unlearning_model.load_state_dict(unlearning_model_state_dict)

        return self.compare_consistency(retraining_model, unlearning_model, unlearning_model_state_dict)


class FairMeasurement:
    def __init__(self, client_datasets, utility_mode = "loss", logger = None, device = "cuda"):
        self.client_datasets = client_datasets
        self.client_num = len(client_datasets)
        self.client_loaders = [DataLoader(client_dataset, batch_size = 128, shuffle = False) for client_dataset in client_datasets]
        # self.unlearned_model = unlearned_model
        # self.baseline_model = baseline_model
        self.utility_mode = utility_mode

        self.logger = logger
        self.device = device

    
    def compute_client_fairness(self, unlearned_model, original_model):
        total_delta_client_loss = 0.
        total_delta_client_acc = 0.

        unlearned_acc_list = []
        original_acc_list = []
        unlearned_loss_list = []
        original_loss_list = []

        for client_loader in self.client_loaders:
            u_acc, u_loss = epoch_test(client_loader, unlearned_model)   # returns (acc, loss)
            o_acc, o_loss = epoch_test(client_loader, original_model)
    
            unlearned_acc_list.append(float(u_acc))
            original_acc_list.append(float(o_acc))
            unlearned_loss_list.append(float(u_loss))
            original_loss_list.append(float(o_loss))

        
        delta_acc_list = [u - o for u, o in zip(unlearned_acc_list, original_acc_list)]
        delta_loss_list = [u - o for u, o in zip(unlearned_loss_list, original_loss_list)]
    
        # Averages (equal weights -> simple mean)
        avg_delta_acc = sum(delta_acc_list) / self.client_num if self.client_num > 0 else 0.0
        avg_delta_loss = sum(delta_loss_list) / self.client_num if self.client_num > 0 else 0.0

        fairness_by_acc = (
            sum(abs(d - avg_delta_acc) for d in delta_acc_list) / self.client_num
            if self.client_num > 0 else 0.0
        )
        fairness_by_loss = (
            sum(abs(d - avg_delta_loss) for d in delta_loss_list) / self.client_num
            if self.client_num > 0 else 0.0
        )

        return {
            "fairness_by_acc": fairness_by_acc,
            "fairness_by_loss": fairness_by_loss,
            "avg_delta_acc": avg_delta_acc,
            "avg_delta_loss": avg_delta_loss,
            "unlearned_acc_list": unlearned_acc_list,
            "original_acc_list": original_acc_list,
            "delta_acc_list": delta_acc_list,
            "unlearned_loss_list": unlearned_loss_list,
            "original_loss_list": original_loss_list,
            "delta_loss_list": delta_loss_list,
        }
        
        