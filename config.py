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
DEVICE = 'cuda'                         # use device "GPU"
# DEVICE = 'cpu'                        # use device "CPU"
SEED = 0                                # 乱数生成器を初期化する値
CLASS_SIZE = 10                         # MNISTの数字ラベル「0」〜「9」の10個
DEG_LABEL_SIZE = 4                      # MNISTの回転角度ラベル「0」〜「3」の4個
BATCH_SIZE = 256                        # バッチ処理用
ZDIM = 64                               # 潜在空間の次元, 大きいほど精度は上がる？が、...
NUM_EPOCHS = 500                       # エポック数。１エポックでデータセットの全てのデータを見たことになる。
LEARNING_RATIO = 1e-3                   # 学習率
NUM_WORKERS = 8                         # ミニバッチを作成する際の並列実行数

#------ Paths -------#
path_mnist_dataset = "./data/"

#------ Flags -------#
flag_mnist_dataset_dl = True
flag_mnist_dataset_train = True
flag_shuffle_trainloader_= True