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


# DEVICE = 'cuda'
DEVICE = 'cpu'
SEED = 0
CLASS_SIZE = 10
DEG_LABEL_SIZE = 4
BATCH_SIZE = 256
ZDIM = 16
NUM_EPOCHS = 20