# minimal tensorflow conv net -> movidius neural compute stick example

1. train a simple model, overfitting like crazy on two examples.
this generates a single ckpt in `ckpt` directory and exports
pbtxt version of graph to `graph.pbtxt`

```
./train_and_export_graph.py
```

2. freeze graph; compiles graph def and checkpoint into frozen graph def single file `graph.frozen.pb`

```
python3 -m tensorflow.python.tools.freeze_graph \
--clear_devices \
--input_graph graph.pbtxt \
--input_checkpoint ckpt/dummy_ckpt \
--output_node_names "output" \
--output_graph graph.frozen.pb
```

3. check the behaviour of the frozen graph by loading it up and running, on host machine,
the two examples we trained on as well as a tensor of just zeros and
a tensor of just ones.

```
./test_inference_on_host.py
-ve prediction [[ 0.00260864]]
+ve prediction [[ 0.99734974]]
zeros prediction [[ 0.50239927]]
ones prediction [[ 0.34015185]]
```

4. convert the tensorflow frozen graph into one loadable onto compute stick (`graph.mv`)

```
mvNCCompile graph.frozen.pb -in imgs -on output -o graph.mv
```

5. test inference on compute stick by running same two examples we used for training the model

```
./test_inference_on_ncs.py
Device 0 Address: 2 - VID/PID 03e7:2150
Starting wait for connect with 2000ms timeout
Found Address: 2 - VID/PID 03e7:2150
Found EP 0x81 : max packet size is 512 bytes
Found EP 0x01 : max packet size is 512 bytes
Found and opened device
Performing bulk write of 865724 bytes...
Successfully sent 865724 bytes of data in 75.076813 ms (10.996987 MB/s)
Boot successful, device address 2
Found Address: 2 - VID/PID 03e7:f63b
done
Booted 2 -> VSC
neg [ 0.00258064]
pos [ 0.99707031]
zeros [ 0.50146484]
ones [ 0.33764648]
```

# graph layout

```
python3 -m tensorflow.python.tools.freeze_graph --clear_devices --input_graph graph.pbtxt --input_checkpoint ckpt/dummy_ckpt --output_node_names output --output_graph graph.frozen.pb
2018-03-24 22:41:11.327754: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX FMA
Converted 8 variables to const ops.
+ tee graph.frozen.summary
+ /home/mat/dev/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=graph.frozen.pb --print_structure
Found 1 possible inputs: (name=imgs, type=float(1), shape=[1,64,64,3])
No variables spotted.
Found 1 possible outputs: (name=output, op=Sigmoid)
Found 1913 (1.91k) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 13 Const, 8 Identity, 4 BiasAdd, 3 Conv2D, 3 Relu, 1 MatMul, 1 Pack, 1 Placeholder, 1 Reshape, 1 Sigmoid, 1 StridedSlice
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=graph.frozen.pb --show_flops --input_layer=imgs --input_layer_type=float --input_layer_shape=1,64,64,3 --output_layer=output
fully_connected/biases (Const): [], value=Tensor<type: float shape: [1] values: 0.0185636394>
fully_connected/biases/read (Identity): [fully_connected/biases]
fully_connected/weights (Const): [], value=Tensor<type: float shape: [512,1] values: [-0.0283197872][-0.0623537377][0.0835486799]...>
fully_connected/weights/read (Identity): [fully_connected/weights]
Flatten/flatten/Reshape/shape/1 (Const): [], value=Tensor<type: int32 shape: [] values: -1>
Flatten/flatten/strided_slice/stack_2 (Const): [], value=Tensor<type: int32 shape: [1] values: 1>
Flatten/flatten/strided_slice/stack_1 (Const): [], value=Tensor<type: int32 shape: [1] values: 1>
Flatten/flatten/strided_slice/stack (Const): [], value=Tensor<type: int32 shape: [1] values: 0>
Flatten/flatten/Shape (Const): [], value=Tensor<type: int32 shape: [4] values: 1 8 8...>
Flatten/flatten/strided_slice (StridedSlice): [Flatten/flatten/Shape, Flatten/flatten/strided_slice/stack, Flatten/flatten/strided_slice/stack_1, Flatten/flatten/strided_slice/stack_2]
Flatten/flatten/Reshape/shape (Pack): [Flatten/flatten/strided_slice, Flatten/flatten/Reshape/shape/1]
e3/biases (Const): [], value=Tensor<type: float shape: [8] values: -0.0414280668 0.0214476679 -0.000389760884...>
e3/biases/read (Identity): [e3/biases]
e3/weights (Const): [], value=Tensor<type: float shape: [3,3,8,8] values: [[[-0.0453788526 0.215599 0.116338238]]]...>
e3/weights/read (Identity): [e3/weights]
e2/biases (Const): [], value=Tensor<type: float shape: [8] values: 0.00499854609 0.0222236235 -0.00933255721...>
e2/biases/read (Identity): [e2/biases]
e2/weights (Const): [], value=Tensor<type: float shape: [3,3,8,8] values: [[[-0.0821640491 -0.122335374 -0.102570482]]]...>
e2/weights/read (Identity): [e2/weights]
e1/biases (Const): [], value=Tensor<type: float shape: [8] values: 0.00932196 0.0351668708 0.0270004775...>
e1/biases/read (Identity): [e1/biases]
e1/weights (Const): [], value=Tensor<type: float shape: [3,3,3,8] values: [[[0.0848274603 -0.0166886337 -0.159113795]]]...>
e1/weights/read (Identity): [e1/weights]
imgs (Placeholder): []
e1/Conv2D (Conv2D): [imgs, e1/weights/read]
e1/BiasAdd (BiasAdd): [e1/Conv2D, e1/biases/read]
e1/Relu (Relu): [e1/BiasAdd]
e2/Conv2D (Conv2D): [e1/Relu, e2/weights/read]
e2/BiasAdd (BiasAdd): [e2/Conv2D, e2/biases/read]
e2/Relu (Relu): [e2/BiasAdd]
e3/Conv2D (Conv2D): [e2/Relu, e3/weights/read]
e3/BiasAdd (BiasAdd): [e3/Conv2D, e3/biases/read]
e3/Relu (Relu): [e3/BiasAdd]
Flatten/flatten/Reshape (Reshape): [e3/Relu, Flatten/flatten/Reshape/shape]
fully_connected/MatMul (MatMul): [Flatten/flatten/Reshape, fully_connected/weights/read]
fully_connected/BiasAdd (BiasAdd): [fully_connected/MatMul, fully_connected/biases/read]
output (Sigmoid): [fully_connected/BiasAdd]
```

# gotchas

noticed so far...

* export frozen graph as batch_size=1; NCS runs without batching
* must have `padding='VALID'` (i.e. SAME will give lots of problems)
* incorrect shape/size is set for model output tensor if 2d. see [this forum post](https://ncsforum.movidius.com/discussion/1128/incorrect-shape-size-set-for-2d-output-tensor)
