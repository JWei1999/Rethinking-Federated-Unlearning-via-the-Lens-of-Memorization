
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
import pickle


def pickle_save(path,obj):
    with open(path,'wb') as f: pickle.dump(obj, f)

def pickle_load(path):
    with open(path,'rb') as f: 
        obj = pickle.load(f)
        return obj

def make_name(word_list, suffix, ifprint=True):
    name = ""
    for word in word_list:
        name += "{}_" 
    name = name.format(*word_list)[:-1]
    name += suffix
    if ifprint:
        print(name)
    return name

def make_path(root_path,work_path,name, ifprint=True):
    path = os.path.join(root_path,work_path)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path,name)
    if ifprint:
        print(path)
    return path

def get_logger(filename, name=None, verbosity=1,filemode="w"):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
      "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    remove_logger_handlers(logger)
    
    fh = logging.FileHandler(filename, filemode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

def remove_logger_handlers(logger):
    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)
    
