import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def classifyAPoint(points,p,k=3):
    '''
     This function finds the classification of p using
     k nearest neighbor algorithm. It assumes only two
     groups and returns 0 if p belongs to group 0, else
      1 (belongs to group 1).

      Parameters -
          points: Dictionary of training points having two keys - 0 and 1
                   Each key have a list of training data points belong to that

          p : A tuple, test data point of the form (x,y)

          k : number of nearest neighbour to consider, default is 3
    '''
    # replace Underweight in the 3rd column of data with 1 and Normal with 0
    t = "Underweight"

    points['Class'].replace({'Underweight': 1, 'Normal': 2})


    distance=[]
    print(points)
    for group in points:
        print(points['Weight(x2)'].values)
        for feature in points[group]:
            print(points['Height(y2)'].values)

            #calculate the euclidean distance of p from training points
            euclidean_distance = math.sqrt((feature[0]-p[0])**2 +(feature[1]-p[1])**2)

            # Add a tuple of form (distance,group) in the distance list
            distance.append((euclidean_distance,group))

    # sort the distance list in ascending order
    # and select first k distances
    distance = sorted(distance)[:k]

    freq1 = 0 #frequency of group 0
    freq2 = 0 #frequency og group 1

    for d in distance:
        if d[1] == 0:
            freq1 += 1
        elif d[1] == 1:
            freq2 += 1

    return 0 if freq1>freq2 else 1
data = pd.read_csv('weight_height_class.csv')
print(data)
# print(data['Class'].values[1])
x=classifyAPoint(data, (51,172))

print(x)
# Using a K- nearest neighbor(KNN) classifier with K=3 and K=5, classify the following test data ( weight=51,Height=172). Use Manhattan distance measures.
