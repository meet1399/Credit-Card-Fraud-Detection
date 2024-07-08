import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv', low_memory=False)
X = df.iloc[:, :-1]
y = df['Class']

X_scaled = scale(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.33, random_state=500)
