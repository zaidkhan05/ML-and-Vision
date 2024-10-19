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
    for i in range(len(names)):
        if '-0.jpg' in names[i]:
            labels.append('NonDR')
        elif '-3.jpg' in names[i] or '-4.jpg' in names[i]:
            labels.append('DR')
        else:
            labels.append('unknown')
        
        # print(names[i], labels[i])
    labeledNames = pd.DataFrame({'imageNames': names, 'labels': labels})
    #get rid of unknowns
    labeledNames = labeledNames[labeledNames['labels'] != 'unknown']
    return labeledNames


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

trainingNames = getImageName(trainDir)
labeledNamesForTraining = loadDiabeticRetinopathyLabels(trainingNames)
#save the labeled names to a csv file
labeledNamesForTraining.to_csv('machinevision/assignment4/labeledNamesForTraining.csv', index=False)
print(labeledNamesForTraining.head())

testingNames = getImageName(testDir)
labeledNamesForTesting = loadDiabeticRetinopathyLabels(testingNames)
#save the labeled names to a csv file
labeledNamesForTesting.to_csv('machinevision/assignment4/labeledNamesForTesting.csv', index=False)
print(labeledNamesForTesting.head())





#0    IDRiD_001_-3.jpg
# 1    IDRiD_002_-3.jpg
# 2    IDRiD_003_-2.jpg
# 3    IDRiD_004_-3.jpg
# 4    IDRiD_005_-4.jpg

