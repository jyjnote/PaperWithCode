import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
