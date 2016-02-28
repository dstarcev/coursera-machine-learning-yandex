import numpy as np
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

data = read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data.iloc[:,-1]
X = data.iloc[:,:-1]

def calculate(X, y, threshold):
    best_t, best_score = 0, -float('inf')
    kf = KFold(len(y), n_folds=5, random_state=1, shuffle=True)
    for t in xrange(1, 51):
        clf = RandomForestRegressor(n_estimators=t, random_state=1)
        score = np.mean(cross_val_score(clf, X, y, cv=kf, scoring='r2'))
        if score > threshold:
            return t

result = calculate(X, y, 0.52)

with open("q1.txt", "w") as output:
    output.write('%d' % (result))
