import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

c = tf.multiply(a,b)

x = tf.constant([[2,2]])
y = tf.constant([[3],
                 [3]])

z = tf.matmul(x,y)

print(c)

with tf.Session() as sess:
    print("Addition",sess.run(a+b))
    print("Multiplication",sess.run(a*b))
    print("c",sess.run(c))
    print("Matrix Multiplication",sess.run(z))

