import tensorflow as tf #importing the tensorflow library
T, F = 1., -1.   #True has the +1 value and False has -1, it's important to note that you can assign any value to them
bias = 1.
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]
training_output = [
    [T],
    [F],
    [F],
    [F],
]
print('training_input')
print(training_input)
print('training_output')
print(training_output)

w = tf.Variable(tf.random_normal([3, 1]))
# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.mul(as_float, 2)
    return tf.sub(doubled, 1)

output = step(tf.matmul(training_input, w))
error = tf.sub(training_output, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(training_input, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err, target = 1, 0
epoch, max_epochs = 0, 10

while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)