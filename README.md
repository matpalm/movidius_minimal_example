dump graph def (pbtxt) and single ckpt

```
./train_and_export_graph.py
```

freeze graph

```
python3 -m tensorflow.python.tools.freeze_graph \
--clear_devices \
--input_graph graph.pbtxt \
--input_checkpoint ckpt/dummy_ckpt \
--output_node_names "output/Sigmoid" \
--output_graph graph.frozen.pb
```


```
~/dev/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph="graph.frozen.pb" --print_structure
Found 1 possible inputs: (name=imgs, type=float(1), shape=[1,50,50,3])
No variables spotted.
Found 1 possible outputs: (name=output, op=Sigmoid)
Found 140 (140) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 2 Const, 2 Identity, 1 BiasAdd, 1 Conv2D, 1 Placeholder, 1 Relu, 1 Sigmoid
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=graph.frozen.pb --show_flops --input_layer=imgs --input_layer_type=float --input_layer_shape=1,50,50,3 --output_layer=output
e1/biases (Const): [], value=Tensor<type: float shape: [5] values: 0 0 0...>
e1/biases/read (Identity): [e1/biases]
e1/weights (Const): [], value=Tensor<type: float shape: [3,3,3,5] values: [[[0.153000504 0.0609570444 -0.0584387183]]]...>
e1/weights/read (Identity): [e1/weights]
imgs (Placeholder): []
e1/Conv2D (Conv2D): [imgs, e1/weights/read]
e1/BiasAdd (BiasAdd): [e1/Conv2D, e1/biases/read]
e1/Relu (Relu): [e1/BiasAdd]
output (Sigmoid): [e1/Relu]
```

does it work against this trivial graph?

```
mvNCCompile graph.frozen.pb -in imgs -on "output/Sigmoid" -o graph.mv
```


