import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance

def split(data):
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values
    return X, y

def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))

def cost(X, y, w, C):
    sum = 0
    n = X.shape[0]
    m = X.shape[1]
    for i in xrange(n):
        sum += np.log(1 + np.exp(-y[i] * np.dot(X[i], w)))
    reg = C * (w ** 2).sum() / m
    cost = sum / np.double(n) + reg
    return cost

def train(X, y, k, C):
    n = X.shape[0]
    m = X.shape[1]
    w = np.zeros(m)
    c = cost(X, y, w, C)
    threshold = 1e-5
    for iteration in xrange(10000):
        new_w = np.zeros(m)
        for j in xrange(m):
            sum = 0
            for i in xrange(n):
                sum += y[i] * X[i, j] * (1 - 1 / (1 + np.exp(-y[i] * np.dot(X[i], w))))
            new_w[j] = w[j] + k * sum / np.double(n) - k * C * w[j]
        new_cost = cost(X, y, new_w, C)
        if distance.euclidean(w, new_w) <= threshold:
            return new_w
        c = new_cost
        w = new_w
    return w

data = pandas.read_csv('data-logistic.csv', header=None)
X, y = split(data)
k = 0.1
score = roc_auc_score(y, sigmoid(X, train(X, y, k, C = 0)))
score_reg = roc_auc_score(y, sigmoid(X, train(X, y, k, C = 10)))

with open("q1.txt", "w") as output:
    output.write('%.3f %.3f' % (score, score_reg))
