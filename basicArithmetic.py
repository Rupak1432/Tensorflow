import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

c = tf.multiply(a,b)

print(c)

with tf.Session() as sess:
    print("Addition",sess.run(a+b))
    print("Multiplication",sess.run(a*b))
    print("c",sess.run(c))

sess.close()