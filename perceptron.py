import numpy as np
import pandas as pd
from numpy.random import rand
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta, n_iter, random_state):
        self.eta = eta
        self.n_iter = n_iter
        self.weight = []
        rng = np.random.default_rng(random_state)
        self.b = rng.normal(0.0, 0.01)

    def predict(self, X):
        if (self.weight.dot(X)+self.b >= 0):
            return 1
        else:
            return -1

    def train(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        sz = len(X[0])
        self.weight = np.random.randn(sz)
        epoches = np.zeros(self.n_iter)
        for _ in range(self.n_iter):
            for x,y in zip(X, Y):
                x=np.array(x)
                y=np.array(y)
                pred = self.predict(x)
                duf = (y-pred)
                pred=duf
                if (pred > 0):
                    self.weight = self.weight + self.eta * x
                    self.b += self.eta
                    epoches[_] += 1
                elif (pred < 0):
                    self.b -= self.eta
                    self.weight = self.weight - self.eta * x
                    epoches[_] += 1
        return epoches

    def test(self, X, Y):
        errors = 0
        err_list = []
        for x,y in zip(X, Y):
            pred = self.predict(x)
            duf = (y-pred)
            if (duf != 0):
                errors += 1
            err_list.append((duf!=0))

        return (errors/len(X), err_list)


#Testing with actual data

import os
path = "iris.data"
if not os.path.exists(path):
    pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                header=None).to_csv(path, header=False, index=False)
df = pd.read_csv(path, header=None)
Y = df.iloc[:100, 4].values
Y=np.where(Y == 'Iris-setosa', 1, -1)
X = df.iloc[:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='*', label='Iris-setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='+', label='Iris-versicolour')

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend()
plt.show()

ppn = Perceptron(eta=0.01, n_iter=10, random_state=101)
epoch = ppn.train(X[25:76], Y[25:76])
(err, el) = ppn.test(X, Y)
print(err)

plt.plot(range(1, len(epoch)+1), epoch)
plt.xlabel('Iterations')
plt.ylabel('Error(s)')
plt.show()

