import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#------ Variables ------#
DEVICE = 'cuda'
# DEVICE = 'cpu'
SEED = 0
CLASS_SIZE = 10
DEG_LABEL_SIZE = 4
BATCH_SIZE = 256
ZDIM = 16
NUM_EPOCHS = 1000
LEARNING_RATIO = 1e-3
NUM_WORKERS = 4

#------ Paths -------#
path_mnist_dataset = "./data/"

#------ Flags -------#
flag_mnist_dataset_dl = True
flag_mnist_dataset_train = True
flag_shuffle_trainloader_= True