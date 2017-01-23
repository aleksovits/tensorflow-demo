# Created by Szabo Sandor-Zsolt
import numpy as np
import tensorflow as tf

vec = [1,2,3]
# Must be Variable
out = tf.Variable(tf.zeros((3,3)))
change_row_op = tf.scatter_update(out,[1],[[1,2,3]])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(change_row_op))
