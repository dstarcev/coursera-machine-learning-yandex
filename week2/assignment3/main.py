import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = pandas.read_csv('perceptron-train.csv', header=None)
test = pandas.read_csv('perceptron-test.csv', header=None)

def split(data):
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values
    return X, y

def estimate(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    prediciton = clf.predict(X_test)
    return accuracy_score(y_test, prediciton)

def scale(train, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    return train_scaled, test_scaled

X_train, y_train = split(train)
X_test, y_test = split(test)

accuracy = estimate(X_train, y_train, X_test, y_test)

X_train_scaled, X_test_scaled = scale(X_train, X_test)

scaled_accuracy = estimate(X_train_scaled, y_train, X_test_scaled, y_test)

result = scaled_accuracy - accuracy

with open("q1.txt", "w") as output:
    output.write('%.3f' % (result))
