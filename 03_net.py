import numpy as np
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

# Fetch the test/training data from the provided tensroflow library
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
training_input = mnist.train.images
test_input = mnist.test.images
training_output = mnist.train.labels
test_output = mnist.test.labels

# 28x28 images => 784 INPUT nodes.
# None indicates that we can feed multiple inputs at once, thus doing (mini-)batch training
X = tf.placeholder(tf.float32, [None, 784]) # This palceholer is the input image(s) in flattened format 1x784 (that we will feed into our network later)
# 10 classes represent each number (0..9)
Y = tf.placeholder(tf.float32, [None, 10]) # This placeholder is the true label of the given X input in one hot encoding

# 625 nodes in the HIDDEN layer
w_input_hidden = tf.Variable(tf.random_normal([784, 625], stddev=0.01)) # A 784x625 matrix represents the connections between the INPUT and the HIDDEN layer
w_hidden_output = tf.Variable(tf.random_normal([625, 10], stddev=0.01)) # A 625x10 matrix represents the connections between the HIDDEN and the OUTPUT layer

hidden_layer = tf.nn.sigmoid(tf.matmul(X, w_input_hidden)) # We create the operation-node(tf term...kind of) for calculating the hidden layer values+sigmoid
output_layer =  tf.matmul(hidden_layer, w_hidden_output) # We create the operation-node for calculating the output layer values (no sigmoid here)
predicted_labels = tf.argmax(output_layer, 1) # Takes the biggest value from every inference. (The biggest "probability")

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, Y)) # Calculate error using built in cross-entropy
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost_function) # With a learning rate of 0.05 try to minimize the cost we've defined

# Create a session that's going to use the tensorflow operation we've defined above
with tf.Session() as sess:
    # Initializes every variable (e.g. w_input_hidden)
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(training_input), 128), range(128, len(training_input)+1, 128)):
            sess.run(train, feed_dict={X: training_input[start:end], Y: training_output[start:end]})
        # Calculates how many of the predicted labels match the true labels, divides by number of labels(np.mean)
        print(i, np.mean(np.argmax(test_output, axis=1) == sess.run(predicted_labels, feed_dict={X: test_input})))
