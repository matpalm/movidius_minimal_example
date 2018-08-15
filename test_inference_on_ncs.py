#!/usr/bin/env python3

import mvnc.mvncapi as mvnc
import numpy as np
import data

# enable full verbose debugging
#mvnc.global_set_option(mvnc.global_set_option.LOG_LEVEL, 2)

# open handle to NCS
devices = mvnc.enumerate_devices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.open()

# load graph onto device
binary_graph = open('graph.mv', 'rb' ).read()
graph = mvnc.Graph('g')
input_fifo, output_fifo = graph.allocate_with_fifos(device, binary_graph)

def run_on_ncs(input):
  graph.queue_inference_with_fifo_elem(input_fifo, output_fifo,
                                       np.float32(input), None)
  output, _user_object = output_fifo.read_elem()
  return output

ncs_positive_prediction = run_on_ncs(data.POS_TENSOR)
ncs_negative_prediction = run_on_ncs(data.NEG_TENSOR)
print("ncs_positive_prediction", ncs_positive_prediction.shape, ncs_positive_prediction)
print("ncs_negative_prediction", ncs_negative_prediction.shape, ncs_negative_prediction)

input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()

