import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc

# Parameters
path = 'D:/ML-and-Vision/'
datasetDirectory = path + 'machinevision/project/dataset/'
definedLabels = path + 'machinevision/project/labels.csv'

# Hyperparameters
batch_size = 32
numEpochs = 50
learningRate = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN model (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 43)  # 43 classes in GTSRB dataset
# model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 43)
model.to(device)

# Image transformations for training and testing data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# DataLoader Setup with optimized parameters
trainSet = datasets.GTSRB(root=datasetDirectory, split='train', download=True, transform=transform)
trainLoader = DataLoader(
    trainSet, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=1,  # Lowering num_workers to reduce disk I/O
    persistent_workers=True  # Keeps workers alive across epochs
)

testSet = datasets.GTSRB(root=datasetDirectory, split='test', download=True, transform=transform)
testLoader = DataLoader(
    testSet, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=1,  # Lowering num_workers to reduce disk I/O
    persistent_workers=True  # Keeps workers alive across epochs
)

# Training function
def trainModel(model, criterion, optimizer, trainLoader, valLoader, epochs):
    model.train()
    total_training_start = time.time()

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in valLoader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        accuracy = (correct / total) * 100
        print(f'Epoch: {epoch+1}, Loss: {val_loss/total:.5f}, Accuracy: {accuracy:.5f}%')

    total_training_end = time.time()
    print(f"Total Training Time: {(total_training_end - total_training_start):.2f} seconds")
    print("Testing Time: {:.2f} minutes".format((total_training_end - total_training_start)/60))

    return model

# Test function
def testModel(model, testLoader):
    model.eval()
    correct, total = 0, 0
    test_start = time.time()
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_end = time.time()
    accuracy = (correct / total) * 100
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    print(f"Testing Time: {(test_end - test_start):.2f} seconds")

    return all_labels, all_preds

# Save model function
def saveModel(model, filepath):
    torch.save(model.state_dict(), filepath)

# ROC Curve and AUC Plotting
def plot_roc_curve(all_labels, all_preds):
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Main script execution
if __name__ == "__main__":
    startTime = time.time()

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model = trainModel(model, criterion, optimizer, trainLoader, testLoader, numEpochs)

    # Save the trained model after training completes
    saveModel(model, path + 'machinevision/project/model.pth')

    # Test the model
    all_labels, all_preds = testModel(model, testLoader)

    # Plot ROC curve and AUC only once, after final testing
    plot_roc_curve(all_labels, all_preds)

    endTime = time.time()
    print(f'Total execution time: {endTime - startTime:.2f} seconds')
    print("Testing Time: {:.2f} minutes".format((endTime - startTime)/60))
