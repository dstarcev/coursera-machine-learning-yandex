from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import pandas, numpy

data = load_boston()
X = data.data
y = data.target

def calculate(X, y):
    best_p, best_score = 0, -float('inf')
    kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
    for p in numpy.linspace(1, 10, num=200):
        knr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
        score = max(cross_val_score(knr, X, y, cv=kf, scoring='mean_squared_error'))
        if score > best_score:
            best_score = score
            best_p = p

    return best_p, best_score

p, score = calculate(scale(X), y)

with open("q1.txt", "w") as output:
    output.write('%.1f' % (p))
