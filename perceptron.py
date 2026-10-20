import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])
        self.errors_ =  []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                upd = self.eta*(target-self.predict(xi))
                self.w_[1:] += upd*xi
                self.w_[0] += upd
                errors += (upd != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
print('URL:', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
df.head()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0: 100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'green', marker = '*', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=2)
plt.show()


ppn = Perceptron(eta=0.1, n_iter=10, random_state = 10**9+7)
ppn.fit(X, y)
plt.plot(range(len(ppn.errors_)), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('No of upds')
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resol = 0.02):
    markers = ('x', 'o', '*', '^', '+', '$')
    colors = ('red', 'green', 'blue', 'grey', 'yellow', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resol), np.arange(x2_min, x2_max, resol))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)
    

print("Decision region seperation: ")
plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal len(cm)')
plt.ylabel('petal len(cm)')
plt.legend(loc =2)
plt.show()
