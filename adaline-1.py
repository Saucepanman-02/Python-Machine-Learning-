import numpy as np
import pandas as pd
from numpy.random import rand
import matplotlib.pyplot as plt

class ADALine:
    def __init__(self, eta, n_iter, random_state):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def train(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        cost = np.zeros(self.n_iter)
        rng = np.random.default_rng(self.random_state)
        self.weight = rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        for _ in range(self.n_iter):
            for x, y in zip(X, Y):
                cur_error = y-self.activate(self.predict(x))
                error = cur_error**2/2
                cost[_] += error.sum()
                delta = -(y-self.activate(self.predict(x)))*x
                self.weight = self.weight - self.eta * delta
                self.b += self.eta*(y-self.activate(self.predict(x)))
        return cost

    def activate(self, X):
        return X

    def predict(self, X):
        return (self.weight.dot(X)+self.b)

    def test(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        pred = []
        for x in X:
            if (self.predict(x) >= 0):
                pred.append(1)
            else:
                pred.append(-1)
        errors = 0
        for (y, p) in zip(Y, pred):
            errors += (y != p)
        return errors/len(pred)

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

'''
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='*', label='Iris-setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='+', label='Iris-versicolour')

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend()
#plt.show()
'''

ppn = ADALine(eta=0.01, n_iter=10, random_state=101)
epoch = ppn.train(X[25:76], Y[25:76])
err = ppn.test(X[np.r_[0:25, 76:100]], Y[np.r_[0:25, 76:100]])
print(f"{err:.5f}")

plt.plot(range(1, len(epoch)+1), epoch)
plt.xlabel('Iterations')
plt.ylabel('Error(s)')
plt.show()

