#!/usr/bin/env python3

import mvnc.mvncapi as mvnc
import numpy as np

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

# make some dummy data. note: non batched
dummy_data = np.random.random(size=(64, 64, 5, 3)).astype(np.float16)

# run through NCS
graph.LoadTensor(dummy_data, 'arbitrary_user_tag')
output, user_object = graph.GetResult()
assert user_object == 'arbitrary_user_tag'
print(output.shape, output[:10])

print("debug", graph.GetGraphOption(mvnc.GraphOption.DEBUG_INFO))
print("time taken", graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN))

# cleanup
graph.DeallocateGraph()
device.CloseDevice()
