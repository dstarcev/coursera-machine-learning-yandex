import pandas
from sklearn.svm import SVC

def split(data):
    X = data.iloc[:,1:].values
    y = data.iloc[:,0].values
    return X, y

def find_support_vectors(data):
    X, y = split(data)
    clf = SVC(C = 100000, kernel = 'linear', random_state = 241)
    clf.fit(X, y)
    return clf.support_

data = pandas.read_csv('svm-data.csv', header=None)
result = find_support_vectors(data) + 1

with open("q1.txt", "w") as output:
    output.write('%s' % (" ".join(map(str, sorted(result)))))
