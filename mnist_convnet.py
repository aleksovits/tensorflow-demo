import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class ConvNetMNIST:

    def __init__(self, session):
        # Fetch the test/training data from the provided tensorflow library
        data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.training_input = data.train.images.reshape(-1, 28, 28, 1) # To have proper form for convolutions Nx28x28x1
        self.test_input = data.test.images.reshape(-1, 28, 28, 1) # To have proper form for convolutions Nx28x28x1
        self.training_output = data.train.labels
        self.test_output = data.test.labels
        self.session = session
        self.build_network_graph()

    def build_network_graph(self):
        self.create_layers()
        self.create_graph(self.w_layer)

    def create_layers(self):
        w_layer = {}
        # Build up the tensorflow operation graph for our conv. net.
        w_layer['conv_0'] = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # Having 32 3x3 convolution windows (We have only 1 layer in the input)
        w_layer['conv_1'] = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # Having 64 3x3 convolution windows (We have only 32 layers from each window of the last layer)
        w_layer['conv_2'] = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)) # Having 128 3x3 convolution windows (We have only 64 layers from each window of the last layer)

        # Adding a fully connected layer that processes the low-level features
        w_layer['ff_0'] = tf.Variable(tf.random_normal([128*4*4, 625], stddev=0.01))
        w_layer['ff_1'] = tf.Variable(tf.random_normal([625, 10], stddev=0.01))
        self.w_layer = w_layer;

    def create_graph(self, w_layer):
        network_layer = {}
        final_layer = {}

        input_image = tf.placeholder(tf.float32, name='input_image')
        output_label = tf.placeholder(tf.float32, name='output_label')
        prob_dropout = tf.placeholder(tf.float32, name='prob_dropout')

        # BULDING THE INFERENCE
        # Input -> Convolutions + ReLU -> 2x2 Max Pooling
        network_layer[0] = tf.nn.relu(tf.nn.conv2d(input_image, w_layer['conv_0'], strides=[1, 1, 1, 1], padding='SAME')) # Takes 28x28x1 -> produces (the SAME) 28x28x*Number of Conv Windows (NCW)
        final_layer[0] = tf.nn.dropout(tf.nn.max_pool(network_layer[0], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), prob_dropout) # Takes 28x28xNCW1 -> produces 14x14xNCW1

        # Layer 1 -> Convolutions + ReLU -> 2x2 Max Pooling
        network_layer[1] = tf.nn.relu(tf.nn.conv2d(final_layer[0], w_layer['conv_1'], strides=[1, 1, 1, 1], padding='SAME')) # Takes 14x14xNCW1 -> produces (the SAME) 14x14x*NCW2
        final_layer[1] = tf.nn.dropout(tf.nn.max_pool(network_layer[1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), prob_dropout) # Takes 14x14xNCW2 -> produces 7x7xNCW2

        # Layer 2 -> Convolutions + ReLU -> 2x2 Max Pooling
        network_layer[2] = tf.nn.relu(tf.nn.conv2d(final_layer[1], w_layer['conv_2'], strides=[1, 1, 1, 1], padding='SAME')) # Takes 7x7xNCW2 -> produces (the SAME) 7x7xNCW3
        final_layer[2] = tf.nn.dropout(tf.nn.max_pool(network_layer[2], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), prob_dropout) # Takes 7x7xNCW3 -> produces 4x4xNCW3

        # Layer 3 -> Fully-Connected Hidden layer
        #network_layer[3] = tf.nn.relu(tf.matmul(tf.reshape(final_layer[2],[1,2048]), w_layer['ff_0']))
        network_layer[3] = tf.nn.relu(tf.matmul(tf.reshape(final_layer[2],[-1,w_layer['ff_0'].get_shape().as_list()[0]]), w_layer['ff_0']))
        final_layer[3] = tf.nn.dropout(network_layer[3], prob_dropout)

        # Layer 4 ->  output (No dropout here)
        final_layer[4] = tf.nn.relu(tf.matmul(final_layer[3], w_layer['ff_1']))
        # Probably add dropout
        # The predicted output label is the max of the Softmax distribution
        self.pred_label = tf.argmax(final_layer[4], 1)

        # BUILDING THE BACKPROPAGATION
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_layer[4], output_label))
        # try others
        self.train = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.error)

    def train_model(self):
        # Initializes every variable (e.g. w_layer variable array)
        tf.global_variables_initializer().run()

        for i in range(100):
            for start, end in zip(range(0, len(self.training_input), 128), range(128, len(self.training_input)+1, 128)):
                self.session.run(self.train, feed_dict={'input_image:0': self.training_input[start:end], 'output_label:0': self.training_output[start:end], 'prob_dropout:0':0.75})
            # Calculates how many of the predicted labels match the true labels, divides by number of labels(np.mean)
            print(i, np.mean(np.argmax(self.test_output, axis=1) == sess.run(self.pred_label, feed_dict={'input_image:0': self.test_input, 'prob_dropout:0':1})))


if __name__ == "__main__":
    print "Building graph"
    with tf.Session() as sess:
        x = ConvNetMNIST(sess)
        x.train_model()
