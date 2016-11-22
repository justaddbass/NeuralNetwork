import numpy as np
import sys
import tensorflow as tf

learning_rate = .01
training_epochs = 135
batch_size = 1
display_step = 10

n_input = 4
n_classes = 3
n_samples = 135
n_test_samples = 15

x = tf.placeholder(tf.float32, shape=[None,n_input], name="n_input")
y = tf.placeholder(tf.float32, shape=[None,n_classes], name="n_class")

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
        prediction = tf.argmax(pred, 1)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_data, test_classes = read_data(sys.argv[2])
        print(prediction.eval(feed_dict={x: test_data}))
        test_accuracy = accuracy.eval({x: test_data,
                                       y: test_classes})
        print("Accuracy:", test_accuracy)
        return accuracy

def read_data(filename):
    data = open(filename, 'r')
    data = data.readlines()
    keys = ['']*len(data)
    for i, line in enumerate(data):
        data[i] = data[i].strip('\n')
        data[i] = data[i].split(',')
        keys[i] = data[i][-1]
        del data[i][-1]
        for j, _ in enumerate(data[i]):
            data[i][j] = float(data[i][j])
    classes = set(keys)
    out = [0]*len(classes)
    out[0] = 1
    out = np.array(out)
    class_dict = dict()
    for i, line in enumerate(classes):
        class_dict[line] = np.roll(out, i).tolist()
    for i, line in enumerate(keys):
        keys[i] = class_dict[line]
    keys = np.array(keys)
    data = np.array(data)
    #print(keys)
    print(class_dict)
    return (data, keys)

def next_batch(data, data2, batch_size):
    batch = data[next_batch.counter:next_batch.counter+batch_size:1, ]
    batch2 = data2[next_batch.counter:next_batch.counter+batch_size:1, ]
    next_batch.counter += batch_size
    return (batch, batch2)
next_batch.counter = 0

if __name__ == "__main__":
    #train(8,8)
    read_data("bezdekiris.data")
