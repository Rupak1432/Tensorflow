from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

learning_rate = 0.01
n_epochs = 1000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
7.042,10.791,5.313,7.997,5.654,9.27,3.1])

train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32, name=None)
Y = tf.placeholder(tf.float32, name=None)

W = tf.Variable(rng.randn(), name="weight")
tf.summary.histogram("Weight", W)

b = tf.Variable(rng.randn(), name="bias")
tf.summary.histogram("Bias", b)

pred = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)  # or tf.reduce_mean(tf.square(pred-Y))
#tf.summary.histogram("cost",cost)

training_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter('./logs/LR/train', sess.graph)
    counter = 0

    for epoch in range(n_epochs):
        for (x,y) in zip(train_X, train_Y):
            counter += 1
            merge = tf.summary.merge_all()
            summary = sess.run(merge)

            sess.run(training_function, feed_dict={X:x, Y:y})

            train_writer.add_summary(summary, counter)
        
        if (epoch+1)%display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch: ",epoch+1, "cost: ",c, "W: ",sess.run(W), "b: ",sess.run(b))
    
    print("\ntraiing finished\n")
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print("Training cost: ",training_cost, "W: ",sess.run(W), "b: ",sess.run(b), "\n")

    plt.plot(train_X,train_Y, 'ro', label = 'Original Data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b), label='fitted line')
    plt.legend()
    plt.show()

    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    testing_cost = sess.run(tf.reduce_mean(tf.square(pred-Y)), feed_dict={X:test_X, Y:test_Y})
    print("Testing Cost: ", testing_cost)
    print("Absolute mean square loss difference: ", abs(training_cost - testing_cost))

    plt.plot(test_X,test_Y, 'bo', label='Ori Testing data')
    plt.plot(test_X,sess.run(W)*test_X + sess.run(b), label = 'fitted line' )
    plt.legend()
    plt.show()

  