import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

malignant = 1
benign = -1

def loadBreastCancerData():
    return pd.read_csv('machinelearning/assignment2/given/wdbc.data.mb.csv', header=None).values
    # return pd.read_csv('machinelearning/assignment2/given/test.csv', header=None).values

    
x = loadBreastCancerData()

# Split the data into training and test sets
def splitData(x):
    split = int(0.7 * len(x))
    return x[:split], x[split:]


#split data into 30 features and 1 label
def splitFeaturesAndLabels(data):
    return data[:, :30], data[:, 30:]

def normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def KNN(train_features, train_labels, test_features, k):

    distances = np.zeros(len(train_features))
    for i in range(len(train_features)):
        distances[i] = np.linalg.norm(train_features[i] - test_features)
    #sort the distances
    sorted_indices = np.argsort(distances)
    #get the k nearest neighbors
    nearest_neighbors = sorted_indices[:k]
    #get the labels of the k nearest neighbors
    nearest_labels = train_labels[nearest_neighbors]
    #count the number of malignant and benign labels
    malignant_count = 0
    benign_count = 0
    for i in range(k):
        if nearest_labels[i] == malignant:
            malignant_count += 1
        else:
            benign_count += 1
    if malignant_count > benign_count:
        return 1
    else:
        return -1

def testKNN(train_features, train_labels, test_features, test_labels, k):
    predicted_labels = np.zeros(len(test_features))
    for i in range(len(test_features)):
        predicted_labels[i] = KNN(train_features, train_labels, test_features[i], k)
    predictions = pd.DataFrame(predicted_labels)
    return predictions

def confusionMatrix(test_labels, predicted_labels):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    cm = confusion_matrix(test_labels, predicted_labels)
    truePos, trueNeg = cm[1, 1], cm[0, 0]
    falsePos, falseNeg = cm[0, 1], cm[1, 0]
    return truePos, trueNeg, falsePos, falseNeg


train, test = splitData(x)
x = train.shape[0]
y = test.shape[0]
print('percent of data used for testing:',f'{x/(x+y)*100:.0f}')

train_features, train_labels = splitFeaturesAndLabels(train)
train_features = normalize(train_features)
test_features, test_labels = splitFeaturesAndLabels(test)
test_features = normalize(test_features)

#classify the points

# predicted_labels = np.zeros(len(test_features))
k = [1, 3, 5, 7, 9]
predictions = []
df = pd.DataFrame()
for i in range(len(k)):
    predictions.append(testKNN(train_features, train_labels, test_features, test_labels, k[i]))
df = {'kValue': k}
TN = []
FP = []
FN = []
TP = []
Accuracy = []
Precision = []
Recall = []
F1_Score = []
for i in range(len(predictions)):
    prediction = np.array(predictions[i])
    truePositive, trueNegative, falsePositive, falseNegative = confusionMatrix(test_labels, prediction)
    TN.append(trueNegative)
    FP.append(falsePositive)
    FN.append(falseNegative)
    TP.append(truePositive)
    accuracy = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
    precision = TP[i] / (TP[i] + FP[i])
    recall = TP[i] / (TP[i] + FN[i])
    f1_score = 2 * TP[i] / (2 * TP[i] + FP[i] + FN[i])
    Accuracy.append(accuracy)
    Precision.append(precision)
    Recall.append(recall)
    F1_Score.append(f1_score)

df = pd.DataFrame(df)
df['TP'] = TP
df['TN'] = TN
df['FP'] = FP
df['FN'] = FN
df['Accuracy'] = Accuracy
df['Precision'] = Precision
df['Recall'] = Recall
df['F1_Score'] = F1_Score
print(df)