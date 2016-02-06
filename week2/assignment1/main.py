from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import pandas
data = pandas.read_csv('wine.data', header=None)

def calculate(X, y):
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=42)
    best_k, best_score = 0, 0
    for k in xrange(1, 51):
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score

X = data.iloc[:,1:].values
y = data.iloc[:,0].values

nk, ns = calculate(scale(X, axis=0), y)
k, s = calculate(X, y)

with open("q1.txt", "w") as output:
    output.write('%d' % (k))

with open("q2.txt", "w") as output:
    output.write('%.2f' % (s))

with open("q3.txt", "w") as output:
    output.write('%d' % (nk))

with open("q4.txt", "w") as output:
    output.write('%.2f' % (ns))
