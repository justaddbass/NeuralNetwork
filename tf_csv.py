#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

from sklearn import datasets
from sklearn.cross_validation import train_test_split
import sklearn
'''
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-l", "--layers", dest="layers")
parser.add_option("-h", "--hidden", dest="hidden")
(options, args) = parser.parse_args()

options.hidden = options.hidden.split(',')
weights = ['']*(len(options.hidden)+1)
biases = ['']*(len(options.hidden)+1)
weights[0] = tf.Variable(tf.random_normal([205, options.hidden[0]]))
for i in range(1, len(options.hidden)-2):
    weights[i] = tf.Variable(tf.random_normal([options.hidden[i], options.hidden[i+1]]))
weights[len(options.hidden)-1] = tf.Variable(tf.random_normal([options.hidden[len(options.hidden)-1], 2]))
for i in range(0, len(options.hidden)-2):
    biases[i] = tf.Variable(tf.random_normal([options.hidden[i]]))
biases[len(option.hidden)-1] = tf.Variable(tf.random_normal([2]))
'''

weights = {
    'h1': tf.Variable(tf.random_normal([205, 16])),
    'h2': tf.Variable(tf.random_normal([16, 8])),
    'h3': tf.Variable(tf.random_normal([8, 4])),
    'out': tf.Variable(tf.random_normal([4, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([16])),
    'b2': tf.Variable(tf.random_normal([8])),
    'b3': tf.Variable(tf.random_normal([4])),
    'out': tf.Variable(tf.random_normal([2]))
}


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer RELU
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def read_data(filename):
    data = open(filename, 'r')
    data = data.readlines()
    keys = ['']*len(data)
    for i, line in enumerate(data):
        data[i] = data[i].strip('\n')
        data[i] = data[i].split(' ')
        del data[i][-1]
        if data[i][-1] != '?':
            keys[i] = int(data[i][-1])
        del data[i][-1]
        for j, _ in enumerate(data[i]):
            data[i][j] = float(data[i][j])
    return (np.array(data), np.array(keys))

# Convert to one hot
def convertOneHot(data):
    y = np.array(data)
    onehot = np.zeros((len(y), 2))
    onehot[np.arange(len(y)), y] = 1
    return (y, onehot)

def next_batch(data, data2, batch_size):
    batch = data[next_batch.counter:next_batch.counter+batch_size]
    batch2 = data2[next_batch.counter:next_batch.counter+batch_size]
    next_batch.counter += batch_size
    return (batch, batch2)
next_batch.counter = 0

learning_rate = .01
batch_size = 1
display_step = 2

x_train, y_train = read_data("train.nmv.txt")
test_data, _ = read_data("prelim-nmv-noclass.txt")
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

training_epochs = len(x_train)

y_train,y_train_onehot = convertOneHot(y_train)
y_validate,y_validate_onehot = convertOneHot(y_validate)

x = tf.placeholder(tf.float32, shape=[None,205], name="n_input")
y = tf.placeholder(tf.float32, shape=[None, 2], name="n_class")

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(n_samples/batch_size)
        total_batch = 1
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(x_train, y_train_onehot, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Training Accuracy: ", accuracy.eval(feed_dict={x: x_train, y: y_train_onehot}))
    print("Test Accuracy: ", accuracy.eval(feed_dict={x: x_validate, y: y_validate_onehot}))

    prediction = tf.argmax(pred, 1)
    out = prediction.eval(feed_dict={x: test_data})
    output = open("output.txt", 'w')
    for i in out:
        output.write(repr(i) + '\n')
    output.close()
