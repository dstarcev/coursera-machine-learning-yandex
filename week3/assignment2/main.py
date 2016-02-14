import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space'])

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

best_score = 0
best_C = None
for a in gs.grid_scores_:
    if a.mean_validation_score > best_score:
        best_score = a.mean_validation_score
        best_C = a.parameters['C']

clf.set_params(C = best_C)
clf.fit(X, y)

ind = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]

words = [tfidf.get_feature_names()[i] for i in ind]

with open("q1.txt", "w") as output:
    output.write('%s' % (" ".join(sorted(words))))
