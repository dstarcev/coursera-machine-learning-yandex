def download_data():
    import os.path
    fname = 'salary-train.csv'
    if os.path.isfile(fname):
        import hashlib
        hash = hashlib.md5(open('salary-train.csv', 'rb').read()).hexdigest()
        if hash == '79f04738042518593ea148dcf860dd32':
            return
    import urllib
    urllib.urlretrieve("https://d3c33hcgiwev3.cloudfront.net/_df0abf627c1cd98b7332b285875e7fe9_salary-train.csv?Expires=1456099200&Signature=Gm5cFk3cR0NRh15HygHbQtd4zQ2D9TE4QhuLXOtly2uHEm3xP556O4Z3Kn~nx7Ah6cvHIQCvi51ZtIwr04U1vBPUOjpQ16DNc6BsMx7SKexghGA5v5gA5Hlj8qDUsZ4H~58mRDAQIdvQvB9gonoYEKIZxjayQRONa69K4e-OzII_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A", "salary-train.csv")

def load_data():
    download_data()
    import pandas
    return pandas.read_csv('salary-train.csv'), pandas.read_csv('salary-test-mini.csv')

def normalize_data(data):
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()

data_train, data_test = load_data()

normalize_data(data_train)
normalize_data(data_test)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5)
X_train_words = tfidf.fit_transform(data_train['FullDescription'])
X_test_words = tfidf.transform(data_test['FullDescription'])

from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

from scipy.sparse import hstack
X_train = hstack([X_train_words, X_train_categ])
X_test = hstack([X_test_words, X_test_categ])

y_train = data_train['SalaryNormalized']

from sklearn.linear_model import Ridge
model = Ridge(alpha=1)
model.fit(X_train, y_train)

result = model.predict(X_test)

with open("q1.txt", "w") as output:
    output.write(' '.join(['%.2f' % (i) for i in result]))
