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
videoPath = path + 'machinevision/project/dataset/inputvideo.mp4'  # Path to your input video

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.40
font = cv2.FONT_HERSHEY_SIMPLEX
frame_skip = 2  # Number of frames to skip for increased playback speed

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load video
cap = cv2.VideoCapture(videoPath)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(25, brightness)

# Load ResNet model with 43 classes
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 43)  # Adjust final layer for 43 classes
# model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 43)  # 43 classes in GTSRB dataset
# model.to(device)

# Load the model weights and move the model to GPU if available
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

frame_count = 0
while cap.isOpened():
    success, imgOriginal = cap.read()
    if not success:
        print("End of video or failed to read the video.")
        break

    # Skip frames for faster playback
    if frame_count % (frame_skip + 1) != 0:
        frame_count += 1
        continue

    # Crop the top half of the frame
    imgTopHalf = imgOriginal[:frameHeight // 2, :]

    # Convert and preprocess the cropped frame
    img = transform(imgTopHalf).unsqueeze(0).to(device)  # Move image tensor to GPU

    with torch.no_grad():
        predictions = model(img)
        probabilityValue, classIndex = F.softmax(predictions, dim=1).max(1)
        probabilityValue = probabilityValue.item()
        classIndex = classIndex.item()

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"{classIndex} {getClassName(classIndex)}", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue*100, 2)}%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(imgOriginal, "No Signal Detected", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", imgOriginal)

    k = cv2.waitKey(1)
    if k == ord('q'):  # Press 'q' to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
