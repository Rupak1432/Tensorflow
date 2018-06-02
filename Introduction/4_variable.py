import tensorflow as tf

x = tf.Variable(0)

add_operation = tf.add(x,1)
update_operation = tf.assign(x, add_operation)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(3):
        sess.run(update_operation)
        print(x)
        print(sess.run(x))
