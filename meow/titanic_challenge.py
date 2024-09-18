# -*- coding: utf-8 -*-
"""titanic_challenge.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FrbdUb7scKPzQYgm6owP7x8851XKkqjM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

train = pd.read_csv('meow/train.csv')
test = pd.read_csv('meow/test.csv')

train.head()

test.head()

train.shape

test.shape

survival_percentages = train.groupby('Sex')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))
plt.bar(survival_percentages.index, survival_percentages, color=['blue', 'salmon'])
plt.xlabel('Gender')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Percentage by Gender')
plt.ylim(0, 100)
plt.show()

survival_percentages_by_class = train.groupby('Pclass')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))

plt.bar(survival_percentages_by_class.index, survival_percentages_by_class, color=['blue','green','pink'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Percentage by Passenger Class')
plt.ylim(0, 100)
plt.xticks(ticks=survival_percentages_by_class.index, labels=[f'Class {i}' for i in survival_percentages_by_class.index])

plt.show()

women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
