import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

with open("q1.txt", "w") as output:
    result = data.Sex.value_counts()
    output.write('%(male)d %(female)d' % result)

with open("q2.txt", "w") as output:
    result = data.Survived.value_counts() / len(data) * 100
    output.write('%.2f' % result[1])

with open("q3.txt", "w") as output:
    result = data.Pclass.value_counts() / len(data) * 100
    output.write('%.2f' % result[1])

with open("q4.txt", "w") as output:
    mean = data.Age.mean()
    median = data.Age.median()
    output.write('%.2f %.2f' % (mean, median))

with open("q5.txt", "w") as output:
    result = data[['SibSp', 'Parch']].corr()
    output.write('%.2f' % result.iat[0, 1])

with open("q6.txt", "w") as output:
    names = data[data.Sex == 'female'].Name
    top_names = pandas.Series(" ".join(names).split()).value_counts()
    # TODO: separate passengeer from ticket buyer and select name automatically
    result = top_names.index[4]
    output.write(result)
