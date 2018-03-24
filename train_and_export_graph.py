#!/usr/bin/env python3

import data
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Define a trivial model; some convolutions, dense connections, and a binary prediction
imgs = tf.placeholder(dtype=tf.float16, shape=(1, 64, 64, 3), name='imgs')
model = slim.conv2d(imgs, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e1')  # (1, 32, 32, 5)
model = slim.conv2d(model, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e2')  # (1, 16, 16, 5)
model = slim.conv2d(model, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e3')  # (1, 8, 8, 5)
model = slim.flatten(model)  # (1, 320)
logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)  # (1, 1)
output = tf.nn.sigmoid(logits, name='output')  # (1, 1)

# Train it to madly overfit two specific known examples (described in data.py)
label = tf.placeholder(dtype=tf.float16, shape=(1, 1), name='label')
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train_op = optimiser.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(500):
  sess.run(train_op, feed_dict={imgs: [data.NEG_TENSOR], label: [[0]]})
  sess.run(train_op, feed_dict={imgs: [data.POS_TENSOR], label: [[1]]})

# save model ckpt and export model graph definition
saver = tf.train.Saver()
if not os.path.exists("ckpt"):
  os.makedirs("ckpt")
saver.save(sess, "ckpt/dummy_ckpt")
tf.train.write_graph(sess.graph_def, ".", "graph.pbtxt")
