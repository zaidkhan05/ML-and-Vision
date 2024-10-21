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
trainDir = fullDir + 'machinevision/assignment4/images/Train/'
testDir = fullDir + 'machinevision/assignment4/images/Test/'

# 1. Load Data: Labels and Images
def getImageName(dir):
    return pd.Series(os.listdir(dir), name='imageNames')

def loadDiabeticRetinopathyLabels(names):
    labels = ['NonDR' if '-0.jpg' in name else 'DR' if '-3.jpg' in name or '-4.jpg' in name else 'unknown' for name in names]
    labeledNames = pd.DataFrame({'imageNames': names, 'labels': labels})
    return labeledNames[labeledNames['labels'] != 'unknown']

# Custom Dataset Class
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_dir, labeled_names, transform=None):
        self.image_dir = image_dir
        self.labeled_names = labeled_names
        self.transform = transform

    def __len__(self):
        return len(self.labeled_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labeled_names.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = 1 if self.labeled_names.iloc[idx, 1] == 'DR' else 0
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Training and Test Labels
trainingNames = getImageName(trainDir)
labeledNamesForTraining = loadDiabeticRetinopathyLabels(trainingNames)
labeledNamesForTraining.to_csv(fullDir + 'machinevision/assignment4/labeledNamesForTraining.csv', index=False)

testingNames = getImageName(testDir)
labeledNamesForTesting = loadDiabeticRetinopathyLabels(testingNames)
labeledNamesForTesting.to_csv(fullDir + 'machinevision/assignment4/labeledNamesForTesting.csv', index=False)

# Create DataLoaders
train_dataset = DiabeticRetinopathyDataset(trainDir, labeledNamesForTraining, transform=data_transforms)
test_dataset = DiabeticRetinopathyDataset(testDir, labeledNamesForTesting, transform=data_transforms)

# Create DataLoaders with multiprocessing enabled
# Create DataLoaders without prefetch_factor
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)



# 2. Load Pretrained Model (Transfer Learning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Modify final layer for binary classification
model.to(device)

# 3. Define Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()  # For mixed-precision training

# 4. Train the Model with Gradient Accumulation
def train_model(model, criterion, optimizer, dataloader, num_epochs, accumulation_steps=4):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.amp.autocast('cuda'):  # Mixed-precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Train the model
train_model(model, criterion, optimizer, train_loader, num_epochs=1000)

def calculate_confusion_matrix(cm):
    TP, TN = cm[1, 1], cm[0, 0]
    FP, FN = cm[0, 1], cm[1, 0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

# 5. Evaluate and Save Predictions to CSV, and Generate ROC and Confusion Matrix
def evaluate_and_plot_roc(model, dataloader, output_csv_path, image_filenames):
    model.eval()

    all_labels = []
    all_probs = []
    predicted_labels = []
    filenames = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.amp.autocast('cuda'):  # Mixed precision inference
                outputs = model(inputs)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            predicted_labels.extend(preds.flatten())
            filenames.extend(image_filenames[i * dataloader.batch_size: (i + 1) * dataloader.batch_size])

    # Save Predictions to CSV
    results = [[filenames[i], 'DR' if predicted_labels[i] == 1 else 'NonDR'] for i in range(len(predicted_labels))]
    pd.DataFrame(results, columns=['imageNames', 'predictedLabels']).to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    cm = confusion_matrix(np.array(all_labels), np.array(predicted_labels))
    calculate_confusion_matrix(cm)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(np.array(all_labels), np.array(all_probs))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(fullDir + 'machinevision/assignment4/roc_curve.png')
    plt.show()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonDR', 'DR'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(fullDir + 'machinevision/assignment4/confusion_matrix.png')
    plt.show()

    # Metrics
    # TP, TN = cm[1, 1], cm[0, 0]
    # FP, FN = cm[0, 1], cm[1, 0]
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    # print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

# Evaluate and plot
output_csv_path = fullDir + 'machinevision/assignment4/predicted_labels.csv'
evaluate_and_plot_roc(model, test_loader, output_csv_path, labeledNamesForTesting['imageNames'].values)
