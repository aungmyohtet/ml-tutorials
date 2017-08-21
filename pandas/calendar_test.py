import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

input_file = 'CalendarMay2014.csv'
names = ['id','r1','r2','r3','s1','s2','s3','t1','t2','t3','f1','f2','f3']

df = pd.read_csv(input_file, names=names)
df = df[0:500]
#print(df.head(1))
#print(df.shape)
#print(df)

dfX = df[['r1','r2', 'r3', 's1', 's2', 's3', 't1', 't2', 't3']]
dfY = df[['f1']]

print(dfX)
print(dfY)

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(dfX, dfY.values.ravel())
print(clf.predict(dfX[0:5]))

#from sklearn.svm import SVC
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.preprocessing import LabelBinarizer

#X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
#y = [0, 0, 1, 1, 2]

#classif = OneVsRestClassifier(estimator=SVC(random_state=0))
#classif.fit(X, y).predict(X)

