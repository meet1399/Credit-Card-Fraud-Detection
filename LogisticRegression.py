import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import sklearn as sk
from sklearn.preprocessing import scale
import random

# Load and preprocess data
df = pd.read_csv('creditcard.csv', low_memory=False)
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# Separate fraud and non-fraud data
frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]

print("We have", len(frauds), "fraud data points and", len(non_frauds), "nonfraudulent data points.")

# Plotting the data
ax = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
non_frauds.plot.scatter(x='Amount', y='Class', color='Blue', label='Normal', ax=ax)
plt.show()

print("This feature looks important based on their distribution with respect to class.")
print("We will now zoom in onto the fraud data to see the ranges of amount just for fun.")


bx = frauds.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud') 
plt.show()

ax = frauds.plot.scatter(x='V22', y='Class', color='Orange', label='Fraud') 
non_frauds.plot.scatter(x='V22', y='Class', color='Blue', label='Normal', ax=ax) 
plt.show() 
print("This feature may not be very important because of the similar distribution.")
