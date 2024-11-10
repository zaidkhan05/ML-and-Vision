import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd

# Set paths and model configuration
path = 'D:/ML-and-Vision/'
modelDirectory = path + 'machinevision/project/model.pth'
videoPath = path + 'machinevision/project/dataset/inputvideo.mp4'

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.40
font = cv2.FONT_HERSHEY_SIMPLEX

# Load video
cap = cv2.VideoCapture(videoPath)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(25, brightness)

# Load ResNet50 model with 43 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 43)
model.load_state_dict(torch.load(modelDirectory, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def getClassName(classNo):
    labels = pd.read_csv(path + 'machinevision/project/labels.csv').values
    return labels[classNo][1]

while cap.isOpened():
    success, imgOriginal = cap.read()
    if not success:
        print("End of video or failed to read the video.")
        break

    # Ignore the bottom half of the frame for classification
    imgOriginal = imgOriginal[:frameHeight // 2, :, :]

    # Convert and preprocess the frame
    img = transform(imgOriginal).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img)
        probabilityValue, classIndex = F.softmax(predictions, dim=1).max(1)
        probabilityValue = probabilityValue.item()
        classIndex = classIndex.item()

    if probabilityValue > threshold:
        className = getClassName(classIndex)
        label = f"{className} ({round(probabilityValue * 100, 2)}%)"
        
        # Draw a bounding box around the entire region of interest
        cv2.rectangle(imgOriginal, (50, 50), (frameWidth - 50, frameHeight // 2 - 50), (0, 255, 0), 2)
        cv2.putText(imgOriginal, label, (60, 60), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(imgOriginal, "No Signal Detected", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", imgOriginal)

    # Adjust video speed (e.g., to play at 2x speed, delay for 500ms per frame)
    k = cv2.waitKey(15)  # Adjust the delay as needed for speed control
    if k == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
