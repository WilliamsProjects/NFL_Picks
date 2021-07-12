import sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 


df = pd.read_csv("DrawnData1.csv")

X = df[['x','y']].values

y = []

for i in range(df.shape[0]):
    if df['z'][i] == 'a':
        y.append(1)
    else:
        y.append(0)



X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
plt.show()


pipe = Pipeline([
    ("scale", QuantileTransformer()),
    ("model", KNeighborsClassifier(n_neighbors=20, weights='distance'))
])

pred = pipe.fit(X, y).predict(X)

plt.scatter(X_new[:, 0], X_new[:, 1], c=pred)
plt.show()
print(2)