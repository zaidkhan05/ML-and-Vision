import pandas as pd
# from pandas import DataFrame as df
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

trainDir = 'machinevision/assignment4/images/Train/'
testDir = 'machinevision/assignment4/images/Test/'



def getImageName(dir):
    filepaths=[]
    imageNames=[]
    filelist=os.listdir(dir)
    imageNames=pd.Series(filelist, name='imageNames')
    return imageNames

def loadDiabeticRetinopathyLabels(names):
    labels = []
    for name in names:
        if '-0' in name:
            labels.append('NonDR')
        elif '-3' or '-4' in name:
            labels.append('DR')

        else:
            labels.append('Unknown')
    return pd.Series(labels, name='labels')


def loadDiabeticRetinopathyImages():
    train_images = []
    test_images = []
    for filename in os.listdir(trainDir):
        img = cv2.imread(trainDir + filename)
        train_images.append(img)
    for filename in os.listdir(testDir):
        img = cv2.imread(testDir + filename)
        test_images.append(img)
    return np.array(train_images), np.array(test_images)

names = getImageName(trainDir)
labels = loadDiabeticRetinopathyLabels(names)

print(names.head())
print(labels.head())




#0    IDRiD_001_-3.jpg
# 1    IDRiD_002_-3.jpg
# 2    IDRiD_003_-2.jpg
# 3    IDRiD_004_-3.jpg
# 4    IDRiD_005_-4.jpg

