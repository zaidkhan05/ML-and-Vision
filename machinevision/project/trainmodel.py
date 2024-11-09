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
numEpochs = 10
learningRate = 0.001
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available


# Image transformations (resize, convert to tensor, normalize) for training and testing data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create PyTorch Dataset
class GTSRBDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)



#CNN model
model = models.resnet50(pretrained=True)  # Load pretrained ResNet50
num_ftrs = model.fc.in_features  # Get number of input features for the final layer
model.fc = nn.Linear(num_ftrs, 43)  # Replace final layer for classification of the 43 classes *do i use 43 or 42 cuz the thing could be 0 indexed*
model.to(device)  # Move model to GPU if available

#training
def trainModel(model, criterion, optimizer, trainLoader, epochs, accumulationSteps=4):
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valLoader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print('Epoch: {}, Loss: {:.5f}, Accuracy: {:.5f}%'.format(epoch+1, val_loss/total, (correct/total)*100))

    return model

# Load the training dataset
trainSet = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
trainLoader = DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2)

# Load the test dataset
testSet = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
testLoader = DataLoader(testSet, batch_size=4, shuffle=False, num_workers=2)
