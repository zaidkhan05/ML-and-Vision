import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Directories
fullDir = 'D:/ML-and-Vision/'
#directories for training and testing images
trainDir = fullDir + 'machinevision/assignment4/images/Train/'
testDir = fullDir + 'machinevision/assignment4/images/Test/'

#Load Data: Get image names and filter out images without labels
def getImageName(dir):
    # Returns a series of image file names from the directory
    return pd.Series(os.listdir(dir), name='imageNames')

def loadDiabeticRetinopathyLabels(names):
    # Assigns labels based on filename patterns ('-0' for NonDR, '-3'/'-4' for DR, others are unknown)
    labels = ['NonDR' if '-0.jpg' in name else 'DR' if '-3.jpg' in name or '-4.jpg' in name else 'unknown' for name in names]
    labeledNames = pd.DataFrame({'imageNames': names, 'labels': labels})
    # Returns only the images that have known labels
    return labeledNames[labeledNames['labels'] != 'unknown']

# Custom Dataset Class for handling image loading and preprocessing
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_dir, labeled_names, transform=None):
        # Initialize with image directory, labels, and optional transformations
        self.image_dir = image_dir
        self.labeled_names = labeled_names
        self.transform = transform

    def __len__(self):
        # Returns the total number of images
        return len(self.labeled_names)

    def __getitem__(self, idx):
        # Retrieves image and its label at a given index
        img_name = os.path.join(self.image_dir, self.labeled_names.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB format
        label = 1 if self.labeled_names.iloc[idx, 1] == 'DR' else 0  # Label DR as 1, NonDR as 0
        if self.transform:
            image = self.transform(image)  # Apply transformations (resize, normalize)
        return image, label

# Image transformations (resize, convert to tensor, normalize) for training and testing data
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with mean and std
])

# Load training image names and labels, then save them to CSV for reference
trainingNames = getImageName(trainDir)
labeledNamesForTraining = loadDiabeticRetinopathyLabels(trainingNames)
labeledNamesForTraining.to_csv(fullDir + 'machinevision/assignment4/labeledNamesForTraining.csv', index=False)

# Load test image names and labels, then save them to CSV for reference
testingNames = getImageName(testDir)
labeledNamesForTesting = loadDiabeticRetinopathyLabels(testingNames)
labeledNamesForTesting.to_csv(fullDir + 'machinevision/assignment4/labeledNamesForTesting.csv', index=False)

# Create datasets for training and testing using the custom dataset class
train_dataset = DiabeticRetinopathyDataset(trainDir, labeledNamesForTraining, transform=data_transforms)
test_dataset = DiabeticRetinopathyDataset(testDir, labeledNamesForTesting, transform=data_transforms)

# Create DataLoaders to load the data in batches, with multiprocessing enabled
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

#Load Pretrained Model (ResNet50) and modify for binary classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

model = models.resnet50(pretrained=True)  # Load pretrained ResNet50
num_ftrs = model.fc.in_features  # Get number of input features for the final layer
model.fc = nn.Linear(num_ftrs, 1)  # Replace final layer for binary classification (DR vs NonDR)
model.to(device)  # Move model to GPU if available

#Define Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001
scaler = torch.cuda.amp.GradScaler()  # Gradient scaler for mixed-precision training

#Train the Model with Gradient Accumulation (to save memory)
def train_model(model, criterion, optimizer, dataloader, num_epochs, accumulation_steps=4):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()  # Zero gradients at the start of each epoch

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.amp.autocast('cuda'):  # Use mixed-precision training for faster computation
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss

            scaler.scale(loss).backward()  # Backward pass with scaled loss

            # Perform optimization step after accumulation of gradients
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients for next iteration

            # Calculate predictions and update accuracy
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total number of labels
            running_loss += loss.item()  # Accumulate loss

        # Print loss and accuracy at the end of each epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Train the model for 10 epochs
train_model(model, criterion, optimizer, train_loader, num_epochs=10)

# Calculate and print confusion matrix and metrics
def calculate_confusion_matrix(cm):
    # Extract true positives, true negatives, false positives, and false negatives
    TP, TN = cm[1, 1], cm[0, 0]
    FP, FN = cm[0, 1], cm[1, 0]
    # Compute accuracy, precision, recall, and F1 score
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonDR', 'DR'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(fullDir + 'machinevision/assignment4/confusion_matrix.png')
    plt.show()

#Evaluate and Save Predictions to CSV, Generate ROC Curve and Confusion Matrix
def evaluate_and_plot_roc(model, dataloader, output_csv_path, image_filenames):
    model.eval()  # Set model to evaluation mode

    all_labels = []
    all_probs = []
    predicted_labels = []
    filenames = []

    with torch.no_grad():  # Disable gradient calculations for inference
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.amp.autocast('cuda'):  # Use mixed precision for inference
                outputs = model(inputs)

            probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            preds = (probs > 0.5).astype(int)  # Convert probabilities to binary predictions

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            predicted_labels.extend(preds.flatten())
            filenames.extend(image_filenames[i * dataloader.batch_size: (i + 1) * dataloader.batch_size])

    # Save predictions to CSV file
    results = [[filenames[i], 'DR' if predicted_labels[i] == 1 else 'NonDR'] for i in range(len(predicted_labels))]
    pd.DataFrame(results, columns=['imageNames', 'predictedLabels']).to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    # Calculate confusion matrix
    cm = confusion_matrix(np.array(all_labels), np.array(predicted_labels))
    calculate_confusion_matrix(cm)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(np.array(all_labels), np.array(all_probs))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Call evaluation function to generate predictions and ROC curve
image_filenames = labeledNamesForTesting['imageNames'].values
evaluate_and_plot_roc(model, test_loader, fullDir + 'machinevision/assignment4/predictions.csv', image_filenames)
