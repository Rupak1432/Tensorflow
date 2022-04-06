# from aymericdamien

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution() #or tfe.enable_eager_execution()

print(tf.executing_eagerly())

a = tf.constant(2)
print(a)
b = tf.constant(3)
print(b)

#Run the operation without the need for tf.Session
c = a+b
print(c)
d = a*b
print(d)

#Full Compatibiltiy with Numpy
#Mixing operations with Tensors and Numpy
a = tf.constant([[2., 1.],
                [1., 0.]], dtype=tf.float32)
print(a)

b = np.array([[3., 0.],
                [5., 1.]], dtype=np.float32)
print(b)


c  = a+b
print(c)

d = tf.matmul(a,b)
print(d)

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
