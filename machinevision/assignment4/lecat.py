import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Directories
fullDir = 'D:/ML-and-Vision/'
trainDir = fullDir + 'machinevision/assignment4/images/Train/'
testDir = fullDir + 'machinevision/assignment4/images/Test/'

# 1. Load Data: Labels and Images
def getImageName(dir):
    filelist = os.listdir(dir)
    imageNames = pd.Series(filelist, name='imageNames')
    return imageNames

def loadDiabeticRetinopathyLabels(names):
    labels = []
    for i in range(len(names)):
        if '-0.jpg' in names[i]:
            labels.append('NonDR')
        elif '-3.jpg' in names[i] or '-4.jpg' in names[i]:
            labels.append('DR')
        else:
            labels.append('unknown')
    labeledNames = pd.DataFrame({'imageNames': names, 'labels': labels})
    labeledNames = labeledNames[labeledNames['labels'] != 'unknown']  # Filter out unknown labels
    return labeledNames

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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 2. Load Pretrained Model (Transfer Learning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Modify final layer for binary classification
model.to(device)

# 3. Define Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the Model
def train_model(model, criterion, optimizer, dataloader, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Train the model
train_model(model, criterion, optimizer, train_loader, num_epochs=10)

# 5. Evaluate and Save Predictions to CSV
def evaluate_model_and_save_predictions(model, dataloader, output_csv_path, image_filenames):
    model.eval()

    correct = 0
    total = 0
    predicted_labels = []
    filenames = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            predicted_labels.extend(preds.cpu().numpy().astype(int).flatten())
            filenames.extend(image_filenames[i * dataloader.batch_size: (i + 1) * dataloader.batch_size])

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save to CSV
    results_df = pd.DataFrame({
        'imageNames': filenames,
        'predictedLabels': ['DR' if label == 1 else 'NonDR' for label in predicted_labels]
    })
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Evaluate the model and save predictions
output_csv_path = fullDir + 'machinevision/assignment4/predicted_labels.csv'

evaluate_model_and_save_predictions(
    model,
    test_loader,
    output_csv_path,
    labeledNamesForTesting['imageNames'].values
)
