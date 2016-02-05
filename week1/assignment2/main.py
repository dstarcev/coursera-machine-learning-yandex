import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

with open("q1.txt", "w") as output:
	data = pd.read_csv('../titanic.csv', index_col='PassengerId')
	data['Sex'] = pd.factorize(data['Sex'])[0]
	complete = data[np.isnan(data.Age) == False]
	X = complete[['Pclass', 'Fare', 'Age', 'Sex']]
	y = complete['Survived']

	clf = DecisionTreeClassifier(random_state = 241)
	clf.fit(X, y)

	result = pd.DataFrame({
		'Name': X.columns,
		'Weight': clf.feature_importances_}).sort('Weight', ascending=False).Name[:2].as_matrix()
	output.write('%s %s' % tuple(result))
