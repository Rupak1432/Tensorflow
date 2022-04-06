import tensorflow as tf
import numpy as np
from glob import glob
import pickle
from random import shuffle

trData = glob("/home/rupak/Ariviyal/Dataset/cifar-10-batches-py/data*")
teData = glob("/home/rupak/Ariviyal/Dataset/cifar-10-batches-py/test_batch")
print(trData)
print(teData)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

xtr = []
ytr = []
xte = []
yte = []
xtr = np.concatenate( [unpickle(i)[b'data'] for i in trData] )
ytr = np.concatenate( [np.asarray(unpickle(i)[b'labels']) for i in trData] )
xte = unpickle(teData[0])[b'data']
yte = np.asarray( unpickle(teData[0])[b'labels'] )
ypr = np.zeros(10000)

# s = list(zip(xtr,ytr))
# shuffle(s)
# xtr,ytr = zip(*s)
# t = list(zip(xte,yte))
# shuffle(t)
# xte,yte = zip(*t)
# xtr = np.asarray(list(xtr[:5000]))
# ytr = np.asarray(list(ytr[:5000]))
# xte = np.asarray(list(xte[:500]))
# yte = np.asarray(list(yte[:500]))

print(np.shape(xtr))
print(np.shape(ytr))
print(np.shape(xte))
print(np.shape(yte))

print(xtr[1:5])
print(ytr[1:5])

#txtr = tf.convert_to_tensor(xtr, dtype=tf.float32)
#tytr = tf.convert_to_tensor(ytr, dtype=tf.float32)
#txte = tf.convert_to_tensor(xte, dtype=tf.float32)
#tyte = tf.convert_to_tensor(yte, dtype=tf.float32)
txtr = tf.placeholder(tf.float32, [None, 3072])
txte = tf.placeholder(tf.float32, [3072])

dist = tf.reduce_sum( tf.abs(tf.subtract(txtr,txte)), 1 )
pred = tf.argmin(dist)
accuracy = 0

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(np.shape(xte)[0]):
        index = sess.run(pred, feed_dict = { txtr : xtr, txte : xte[i,:]})
        ypr[i] = ytr[index]
        if i%100 == 0:
            print(i)
        
accuracy = (np.sum(np.asarray(ypr) == yte))/(np.shape(xte)[0])
print(accuracy)
