import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

malignant = 1
benign = -1

def loadBreastCancerData():
    return pd.read_csv('machinelearning/assignment2/given/wdbc.data.mb.csv', header=None).values
    # return pd.read_csv('machinelearning/assignment2/given/test.csv', header=None).values


x = loadBreastCancerData()
print(x.shape)

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
    # print(train_features.shape, train_labels.shape)
    # print(test_features.shape)
    distances = np.zeros(len(train_features))
    for i in range(len(train_features)):
        distances[i] = np.linalg.norm(train_features[i] - test_features)
    # print(distances)
    #sort the distances
    sorted_indices = np.argsort(distances)
    # print(sorted_indices)
    #get the k nearest neighbors
    nearest_neighbors = sorted_indices[:k]
    # print(nearest_neighbors)
    #get the labels of the k nearest neighbors
    nearest_labels = train_labels[nearest_neighbors]
    # print(nearest_labels)
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
    print(k)
    for i in range(len(test_features)):
        predicted_labels[i] = KNN(train_features, train_labels, test_features[i], k)
    # for i in range(len(predictions)): #why this no work properly
    TP, TN, FP, FN = confusionMatrix(test_labels, predicted_labels)
    print('k =', k)
    print(TP, TN, FP, FN)
    print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
    print('Precision:', TP / (TP + FP))
    print('Recall:', TP / (TP + FN))
    print('F1 Score:', 2 * TP / (2 * TP + FP + FN))
    predictions = pd.DataFrame(predicted_labels)
    return predictions

def confusionMatrix(test_labels, predicted_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(test_labels)):
        if test_labels[i] == malignant and predicted_labels[i] == malignant:
            TP += 1
        elif test_labels[i] == benign and predicted_labels[i] == benign:
            TN += 1
        elif test_labels[i] == benign and predicted_labels[i] == malignant:
            FP += 1
        else:
            FN += 1
    return TP, TN, FP, FN


# row = x[0]


train, test = splitData(x)
print(train.shape, test.shape)

train_features, train_labels = splitFeaturesAndLabels(train)
train_features = normalize(train_features)
test_features, test_labels = splitFeaturesAndLabels(test)
test_features = normalize(test_features)

#classify the points

# predicted_labels = np.zeros(len(test_features))
k = [1, 3, 5, 7, 9]
# predictions = [predicted_labels] *5
# for i in range(len(k)):
#     print(k[i])
#     for j in range(len(test_features)):
#         predictions[i][j] = KNN(train_features, train_labels, test_features[j], k[i])
# # for i in range(len(predictions)): #why this no work properly
#     TP, TN, FP, FN = confusionMatrix(test_labels, predictions[i])
#     print('k =', k[i])
#     print(TP, TN, FP, FN)
#     print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
#     print('Precision:', TP / (TP + FP))
#     print('Recall:', TP / (TP + FN))
#     print('F1 Score:', 2 * TP / (2 * TP + FP + FN))
# predictions = pd.DataFrame(predictions)
predictions = []
for i in range(len(k)):
    predictions.append(testKNN(train_features, train_labels, test_features, test_labels, k[i]))
    #newline
    print('')
for i in range(len(predictions)):
    print("")
    prediction = np.array(predictions[i])
    TP, TN, FP, FN = confusionMatrix(test_labels, prediction)
    print('k =', f'predictions[{i*2+1}]')
    print(TP, TN, FP, FN)
    print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
    print('Precision:', TP / (TP + FP))
    print('Recall:', TP / (TP + FN))
    print('F1 Score:', 2 * TP / (2 * TP + FP + FN))
    
# print(predictions)
# predictions = pd.DataFrame(predictions)
print(predictions)



# for i in range(len(test_features)):
#     predicted_labels[i] = KNN(train_features, train_labels, test_features[i], 1)
# print(predicted_labels)




# TP, TN, FP, FN = confusionMatrix(test_labels, predicted_labels)

# print(TP, TN, FP, FN)
# print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
# print('Precision:', TP / (TP + FP))
# print('Recall:', TP / (TP + FN))
# print('F1 Score:', 2 * TP / (2 * TP + FP + FN))
# print(predictions)



# print(train_features.shape, train_labels.shape)
# print(train_labels[0])
# print(train_features[0])#row549
# print(train[0])
# print(x[0])