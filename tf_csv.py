#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

# Build Example Data is CSV format, but use Iris data
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import sklearn

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

weights = {
    'h1': tf.Variable(tf.random_normal([205, 128])),
    'h2': tf.Variable(tf.random_normal([128, 128])),
    'out': tf.Variable(tf.random_normal([128, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([128])),
    'b2': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([2]))
}

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def next_batch(data, data2, batch_size):
    batch = data[next_batch.counter:next_batch.counter+batch_size]
    batch2 = data2[next_batch.counter:next_batch.counter+batch_size]
    next_batch.counter += batch_size
    return (batch, batch2)
next_batch.counter = 0

learning_rate = .01
training_epochs = 9000
batch_size = 1
display_step = 1

n_input = 206
n_classes = 2
n_samples = 9000
n_test_samples = 4000

x_train, target = read_data("train.nmv.txt")
test_data, _ = read_data("prelim-nmv-noclass.txt")
#x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

#data = genfromtxt('cs-training.csv',delimiter=',')  # Training data
#test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data

#x_train=np.array([ i[:-1:] for i in x_train])
y_train,y_train_onehot = convertOneHot(target)
#print(x_train)
#print(y_train_onehot)
#x_test=np.array([ i[:-1:] for i in x_test])
#y_test,y_test_onehot = convertOneHot(y_test)

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
    prediction = tf.argmax(pred, 1)
    out = prediction.eval(feed_dict={x: test_data})
    output = open("output.txt", 'w')
    for i in out:
        output.write(repr(i) + '\n')
    output.close()
