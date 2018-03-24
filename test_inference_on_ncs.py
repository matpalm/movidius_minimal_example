#!/usr/bin/env python3

import mvnc.mvncapi as mvnc
import numpy as np
import data

# enable full verbose debugging
mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)

# open handle to NCS
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.OpenDevice()

# load graph onto device
binary_graph = open('graph.mv', 'rb' ).read()
graph = device.AllocateGraph(binary_graph)

# run -ve example through NCS
for t, tag in [(data.NEG_TENSOR, 'neg'),
               (data.POS_TENSOR, 'pos'),
               (np.zeros((64, 64, 3)).astype(np.float16), 'zeros'),
               (np.ones((64, 64, 3)).astype(np.float16), 'ones')]:
  graph.LoadTensor(t, '')
  output, _user_object = graph.GetResult()
  print(tag, output)
#  print("debug", graph.GetGraphOption(mvnc.GraphOption.DEBUG_INFO))
#  print("time taken", graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN))

# cleanup
graph.DeallocateGraph()
device.CloseDevice()
