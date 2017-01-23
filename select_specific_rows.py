import tensorflow as tf
import numpy as np
'''
#row = tf.gather(tf.constant([[1, 2],[3, 4]]), row_indices)
input = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
function_to_map = lambda x: (x ** 2)
final_result = tf.map_fn(function_to_map, input)
'''

x = tf.Variable([
    [1,2,3,1],
    [0,0,0,0],
    [1,3,5,7],
    [0,0,0,0],
    [3,5,7,8]])

y = tf.Variable([0,0,0,0])
z = tf.Variable([0,0,0,1])
sim = tf.equal(x,z)
condition = tf.equal(x, y)
res = tf.where(condition)

row_wise_sum = tf.reduce_sum(tf.abs(x),1)
select_zero_sum = tf.where(tf.equal(row_wise_sum,0))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(select_zero_sum))
