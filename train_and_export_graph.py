#!/usr/bin/env python3

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import data

# Define a trivial model using convolutions and some dense connections to a binary prediction
imgs = tf.placeholder(dtype=tf.float16, shape=(2, 64, 64, 3), name='imgs')
model = slim.conv2d(imgs, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e1')  # (2, 32, 32, 5)
model = slim.conv2d(model, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e2')  # (2, 16, 16, 5)
model = slim.conv2d(model, num_outputs=5, kernel_size=3, stride=2, padding='SAME', scope='e3')  # (2, 8, 8, 5)
model = slim.flatten(model)  # (2, 320)
logits = slim.fully_connected(model, num_outputs=1, activation_fn=None)  # (2, 1)
output = tf.nn.sigmoid(logits, name='output')  # (2, 1)

# Train it to madly overfit two specific known examples (described in data.py)
labels = tf.placeholder(dtype=tf.float32, shape=(2, 1), name='labels')
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
optimiser = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimiser.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(100):
  sess.run(train_op,
           feeddict={imgs:
                     labels:[


saver = tf.train.Saver()
if not os.path.exists("ckpt"):
  os.makedirs("ckpt")
saver.save(sess, "ckpt/dummy_ckpt")

tf.train.write_graph(sess.graph_def, ".", "graph.pbtxt")
