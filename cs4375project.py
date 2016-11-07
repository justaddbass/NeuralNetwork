import numpy as np
import sys
import tensorflow as tf

learning_rate = 1
training_epochs = 4
batch_size = 1
display_step = 1

n_input = 2
n_classes = 2
n_samples = 4

x = tf.placeholder(tf.float32, shape=[batch_size,n_input], name="n_input")
y = tf.placeholder(tf.float32, shape=[batch_size,n_classes], name="n_class")

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

def train(layer1, layer2):
    records, classes = read_data(sys.argv[1])

    n_hidden_1 = layer1
    n_hidden_2 = layer2

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #cost = tf.reduce_sum((y - pred)**2/n_samples)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            #total_batch = int(n_samples/batch_size)
            total_batch = 1
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(records, classes, batch_size)
                #batch_y = batch_y.reshape([1,])
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        #print("Accuracy:", test_accuracy)
        #return accuracy

def read_data(filename):
    data = open(filename, 'r')
    data = data.readlines()
    key = ['']*n_samples
    for i, line in enumerate(data):
        data[i] = data[i].strip('\n')
        data[i] = data[i].split(' ')
        del data[i][-1]
        for j, _ in enumerate(data[i]):
            data[i][j] = int(data[i][j])
    val = np.array(data)
    key = val[:,-1].tolist()
    print(val[:-1].tolist())
    for i, line in enumerate(key):
        if line == 0:
            key[i] = [0,1]
        else:
            key[i] = [1,0]
    key = np.array(key)
    print(key)
    #print(val)
    val = np.delete(val, len(val[0])-1, 1)
    return (val, key)

def next_batch(data, data2, batch_size):
    batch = data[next_batch.counter % n_samples:(next_batch.counter+batch_size) % n_samples:1, ]
    batch2 = data2[next_batch.counter:next_batch.counter+batch_size:1, ]
    #batch2 = np.reshape(batch2, [n_classes,])
    next_batch.counter += batch_size
    return (batch, batch2)
next_batch.counter = 0

train(1,1)
