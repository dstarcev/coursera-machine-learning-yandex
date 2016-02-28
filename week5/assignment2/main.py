fname = 'gbm-data.csv'
def download_data():
    import os.path
    if os.path.isfile(fname):
        import hashlib
        hash = hashlib.md5(open(fname, 'rb').read()).hexdigest()
        if hash == '675c7a691add0c328b8b27db003a96aa':
            return
    import urllib
    urllib.urlretrieve('https://d3c33hcgiwev3.cloudfront.net/_75fb7a1b6f3431b6217cdbcba2fd30b9_gbm-data.csv?Expires=1456790400&Signature=QVeBlc-wGKB1TjJJg0ciUwE-fifxMGUjrKC-4~Ew6oZOX3MFF2NBftJdLgY16fnxaWE3jWzZkXD1v5vWqatp1nMgGAOVzgZVPUt9Wqg5G-Y~C5Hc87NJEG2bVuIXj6zBiB9X2p4EbyW~wRXnc7yNWQdFihOohVwIL4bdgdPRYck_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A', fname)

download_data()

from pandas import read_csv
data = read_csv(fname)

X = data.iloc[:,1:]
y = data.iloc[:,0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import numpy as np, os
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot(train_loss, test_loss, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # %matplotlib inline
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig(fname)

min_losses = {}
for index, learning_rate in enumerate([1, 0.5, 0.3, 0.2, 0.1], start=1):
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    train_pred_iters = clf.staged_predict_proba(X_train)
    test_pred_iters = clf.staged_predict_proba(X_test)
    train_loss = [ log_loss(y_train, pred) for pred in train_pred_iters]
    test_loss = [ log_loss(y_test, pred) for pred in test_pred_iters]
    best_iter = np.argmin(test_loss)
    min_losses[learning_rate] = (test_loss[best_iter], best_iter)
    plot(train_loss, test_loss, 'plots/%d_%.1f.png' % (index, learning_rate))

# based on plots view
with open('q1.txt', 'w') as output:
    output.write('overfitting')

with open('q2.txt', 'w') as output:
    output.write('%.2f %d' % min_losses[0.2])

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=min_losses[0.2][1], random_state=241)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_score = log_loss(y_test, rf_pred)

with open('q3.txt', 'w') as output:
    output.write('%.2f' % (rf_score))
