import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Parameters
path = ''
datasetDirectory = path + 'machinevision/project/dataset/'
definedLabels = path + 'machinevision/project/labels.csv'

# Hyperparameters
batch_size = 32
numEpochs = 10
learningRate = 0.001

# Image transformations for training and testing data
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

# CNN model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 43)  # 43 classes in GTSRB dataset

# Training function
def trainModel(model, criterion, optimizer, trainLoader, valLoader, epochs):
    model.train()
    total_training_start = time.time()
    for epoch in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valLoader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch: {epoch+1}, Loss: {val_loss/total:.5f}, Accuracy: {(correct/total)*100:.5f}%')
    total_training_end = time.time()  # End total training timer
    print("Total Training Time: {:.2f} seconds".format(total_training_end - total_training_start))
    print("Testing Time: {:.2f} minutes".format((total_training_end - total_training_start)/60))

    return model

def testModel(model, testLoader):
    model.eval()
    correct = 0
    total = 0
    test_start = time.time()  # Start testing timer
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_end = time.time()  # End testing timer
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print("Testing Time: {:.2f} seconds".format(test_end - test_start))
    #in minutes
    print("Testing Time: {:.2f} minutes".format((test_end - test_start)/60))

# Main script execution
if __name__ == "__main__":
    startTime = time.time()
    # Device configuration
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the training dataset
    trainSet = datasets.GTSRB(root=datasetDirectory, split='train', download=True, transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load the test dataset
    testSet = datasets.GTSRB(root=datasetDirectory, split='test', download=True, transform=transform)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=2)

    # Train the model
    model = trainModel(model, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=learningRate), trainLoader, testLoader, numEpochs)

    # Save the model checkpoint
    torch.save(model.state_dict(), path + 'machinevision/project/model.pth')

    # Test the model
    testModel(model, testLoader)
    endTime = time.time()
    print(f'Total execution time: {endTime - startTime:.2f} seconds')
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testLoader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')

    # Calculate class-wise accuracy
    class_correct = list(0. for i in range(43))
    class_total = list(0. for i in range(43))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Handling variable batch sizes
            for i in range(len(labels)):  # Use len(labels) to handle variable batch sizes
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Display class-wise accuracy and save to a CSV file
    for i in range(43):
        if class_total[i] > 0:  # Avoid division by zero
            # print(f'Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f} %')
            df = pd.DataFrame({'Class': i, 'Accuracy': 100 * class_correct[i] / class_total[i]}, index=[0])
            df.to_csv(path + 'machinevision/project/class_wise_accuracy.csv', mode='a', header=False, index=False)

        # else:
        #     # print(f'Accuracy of class {i} : No samples')



    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
