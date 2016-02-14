import pandas
import numpy as np
import sklearn.metrics

data = pandas.read_csv('classification.csv')

t = data[data['true'] == data['pred']]['pred']
tp = t.sum()
tn = t.size - tp

f = data[data['true'] != data['pred']]['pred']
fp = f.sum()
fn = f.size - fp

with open("q1.txt", "w") as output:
    output.write('%d %d %d %d' % (tp, fp, fn, tn))

accuracy = sklearn.metrics.accuracy_score(data['true'], data['pred'])
precision = sklearn.metrics.precision_score(data['true'], data['pred'])
recall = sklearn.metrics.recall_score(data['true'], data['pred'])
F = sklearn.metrics.f1_score(data['true'], data['pred'])

with open("q2.txt", "w") as output:
    output.write('%.2f %.2f %.2f %.2f' % (accuracy, precision, recall, F))

scores = pandas.read_csv('scores.csv')

algs = scores.columns[1:]
def find_best_alg(score_func):
    s = algs.map(lambda alg: [score_func(alg), alg])

    return np.sort(s)[::-1][0]

best_roc, best_roc_alg = find_best_alg(lambda alg:
    sklearn.metrics.roc_auc_score(scores['true'], scores[alg]))

with open("q3.txt", "w") as output:
    output.write('%s' % (best_roc_alg))

def best_prc_score(alg):
    prc = sklearn.metrics.precision_recall_curve(scores['true'], scores[alg])
    fr = pandas.DataFrame({ 'precision': prc[0], 'recall': prc[1] })
    return fr[fr['recall'] >= 0.7]['precision'].max()

best_prc, best_prc_alg = find_best_alg(best_prc_score)

with open("q4.txt", "w") as output:
    output.write('%s' % (best_prc_alg))
