# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import urllib.request
from sklearn import svm

url = "file:calendar.csv"
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,1:9]
y = dataset[:,12]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X,y)
print(clf.predict(X[0:10]))