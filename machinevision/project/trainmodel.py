import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
####################################################################################################
#Parameters
#################################################
path = ''
datasetDirectory = path+'machinevision/project/dataset/'
definedLabels = path+'machinevision/project/labels.csv'
# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
#model
model = models.resnet50(pretrained=True)  # Load pretrained ResNet50
num_ftrs = model.fc.in_features  # Get number of input features for the final layer

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
trainset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Load the test dataset
testset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
