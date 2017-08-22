import numpy as np
import urllib.request
from sklearn import svm

url = "file:../sklearn/calendar.csv"
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
# separate the data from the target attributes
calendarX = dataset[:,1:10]
calendarY = dataset[:,11:12]

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 9])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([9,1]))
b = tf.Variable(tf.zeros([1]))
sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    train_step.run(feed_dict={x: calendarX, y_: calendarY})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('correct_prediction')
print(correct_prediction)
print('accuracy')
print(accuracy)
print(accuracy.eval(feed_dict={x: calendarX, y_: calendarY}))

feed_dict = {x: calendarX[0:10]}
classification = sess.run(tf.argmax(y, 1), feed_dict=feed_dict)
print('classification')
print(classification)

prediction=tf.argmax(y,1)
print(prediction.eval(feed_dict={x: calendarX[0:10]}, session=sess))
