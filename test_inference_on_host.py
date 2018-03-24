#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import data

graph_def = tf.GraphDef()
with open('graph.frozen.pb', "rb") as f:
  graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name=None)

imgs = tf.get_default_graph().get_tensor_by_name('import/imgs:0')
model_output = tf.get_default_graph().get_tensor_by_name('import/output:0')

with tf.Session() as sess:
  print("-ve prediction", sess.run(model_output, feed_dict={imgs: [data.NEG_TENSOR]}))
  print("+ve prediction", sess.run(model_output, feed_dict={imgs: [data.POS_TENSOR]}))
  print("zeros prediction", sess.run(model_output, feed_dict={imgs: np.zeros((1, 64, 64, 3))}))
  print("ones prediction", sess.run(model_output, feed_dict={imgs: np.ones((1, 64, 64, 3))}))
