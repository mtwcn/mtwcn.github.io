---
"title": "TensorflowLite Model Structure",
"tags": ["ai", "tensorflowlite", "model"],
"date": "2018-12-01",
"author": "gerry"
---
### <font color="LightSeaGreen"> Intruduction </font>

This article descript about the structure of a [tensorflowlite](https://www.tensorflow.org/lite/) model file(.tflite) and how it was interpreted by tensorflowlite framework. 

### <font color="LightSeaGreen">File Format </font>

The tflite model user [flatbuffer](https://google.github.io/flatbuffers/) which will not covered by thie article.  Let's look at the model structure

#### **Model**
* **version** : the version of tflite scheme
* **operator_codes** : the list of operator_code that used by this model (both builtin & custom)
* **[subgraph](#Subgraph)** : the graph for tensors, inputs and outputs
* **description**
* **buffers**
* **metadata_buffer**
~~~
struct ModelT : public flatbuffers::NativeTable {
  typedef Model TableType;
  uint32_t version;
  std::vector<std::unique_ptr<OperatorCodeT>> operator_codes;
  std::vector<std::unique_ptr<SubGraphT>> subgraphs;
  std::string description;
  std::vector<std::unique_ptr<BufferT>> buffers;
  std::vector<int32_t> metadata_buffer;
  ModelT()
      : version(0) {
  }
};
~~~
#### **OperatorCode**
~~~
struct OperatorCodeT : public flatbuffers::NativeTable {
  typedef OperatorCode TableType;
  BuiltinOperator builtin_code;
  std::string custom_code;
  int32_t version;
  OperatorCodeT()
      : builtin_code(BuiltinOperator_ADD),
        version(1) {
  }
};
~~~
#### Subgraph

Till now tensorflowlite only support **1** subgraph. When get the subgraph info from flatbuffer, interpreter parse each operator to a node, which will be set to a execution_plan and push to a execution queue. That means **one operator -> one node -> one excution plan**;


~~~
struct SubGraphT : public flatbuffers::NativeTable {
  typedef SubGraph TableType;
  std::vector<std::unique_ptr<TensorT>> tensors;
  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  std::vector<std::unique_ptr<OperatorT>> operators;
  std::string name;
  SubGraphT() {
  }
};
~~~
#### Operator
~~~
struct OperatorT : public flatbuffers::NativeTable {
  typedef Operator TableType;
  uint32_t opcode_index;
  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  BuiltinOptionsUnion builtin_options;
  std::vector<uint8_t> custom_options;
  CustomOptionsFormat custom_options_format;
  std::vector<bool> mutating_variable_inputs;
  OperatorT()
      : opcode_index(0),
        custom_options_format(CustomOptionsFormat_FLEXBUFFERS) {
  }
};

~~~
#### Tensor
The interpreter parse Tensor info from flatbuffer,  do quantization, set tensor with parameters. there are two different methods to set parameter for ro and rw for tenors with buffer and tensors without buffer.

~~~
struct TensorT : public flatbuffers::NativeTable {
  typedef Tensor TableType;
  std::vector<int32_t> shape;
  TensorType type; 
  uint32_t buffer;
  std::string name; 
  std::unique_ptr<QuantizationParametersT> quantization;
  bool is_variable;
  TensorT()
      : type(TensorType_FLOAT32),
        buffer(0),
        is_variable(false) {
  }
};
~~~
### <font color="LightSeaGreen"> Diagram </font>
![](static/img/workstuff/ai_tensorflowlite-model-structure_1.jpg)
### <font color="LightSeaGreen"> Interpreter </font>
The tflite model file is mmap to memory and interprete to the class [tflite::Interpreter](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/lite/interpreter.h) . there are four major members in this class, they are
* **inputs_** : the input node for the model
* **outputs_**: the output node for the model
* **tensors_**: the tensor list 
* **nodes_and_registration_**: the node and registration list

here is a dump of tensors and node for interpreter of mobilenet_v1_1.0_224.tflite

~~~
Interpreter has 105 tensors and 31 nodes
Inputs: 88
Outputs: 87

Tensor   0 MobilenetV1/Conv2d_0/weights kTfLiteFloat32   kTfLiteMmapRo       3456 bytes ( 0.0 MB)   is_variable = 0 32 3 3 3
Tensor   1 MobilenetV1/Conv2d_10_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor   2 MobilenetV1/Conv2d_10_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)   is_variable = 0 512 1 1 512
Tensor   3 MobilenetV1/Conv2d_11_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor   4 MobilenetV1/Conv2d_11_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)   is_variable = 0 512 1 1 512
Tensor   5 MobilenetV1/Conv2d_12_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor   6 MobilenetV1/Conv2d_12_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    2097152 bytes ( 2.0 MB)   is_variable = 0 1024 1 1 512
Tensor   7 MobilenetV1/Conv2d_13_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)   is_variable = 0 1 3 3 1024
Tensor   8 MobilenetV1/Conv2d_13_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    4194304 bytes ( 4.0 MB)   is_variable = 0 1024 1 1 1024
Tensor   9 MobilenetV1/Conv2d_1_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       1152 bytes ( 0.0 MB)   is_variable = 0 1 3 3 32
Tensor  10 MobilenetV1/Conv2d_1_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo       8192 bytes ( 0.0 MB)   is_variable = 0 64 1 1 32
Tensor  11 MobilenetV1/Conv2d_2_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       2304 bytes ( 0.0 MB)   is_variable = 0 1 3 3 64
Tensor  12 MobilenetV1/Conv2d_2_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo      32768 bytes ( 0.0 MB)   is_variable = 0 128 1 1 64
Tensor  13 MobilenetV1/Conv2d_3_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)   is_variable = 0 1 3 3 128
Tensor  14 MobilenetV1/Conv2d_3_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo      65536 bytes ( 0.1 MB)   is_variable = 0 128 1 1 128
Tensor  15 MobilenetV1/Conv2d_4_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)   is_variable = 0 1 3 3 128
Tensor  16 MobilenetV1/Conv2d_4_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo     131072 bytes ( 0.1 MB)   is_variable = 0 256 1 1 128
Tensor  17 MobilenetV1/Conv2d_5_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)   is_variable = 0 1 3 3 256
Tensor  18 MobilenetV1/Conv2d_5_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo     262144 bytes ( 0.2 MB)   is_variable = 0 256 1 1 256
Tensor  19 MobilenetV1/Conv2d_6_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)   is_variable = 0 1 3 3 256
Tensor  20 MobilenetV1/Conv2d_6_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo     524288 bytes ( 0.5 MB)   is_variable = 0 512 1 1 256
Tensor  21 MobilenetV1/Conv2d_7_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor  22 MobilenetV1/Conv2d_7_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)   is_variable = 0 512 1 1 512
Tensor  23 MobilenetV1/Conv2d_8_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor  24 MobilenetV1/Conv2d_8_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)   is_variable = 0 512 1 1 512
Tensor  25 MobilenetV1/Conv2d_9_depthwise/depthwise_weights kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)   is_variable = 0 1 3 3 512
Tensor  26 MobilenetV1/Conv2d_9_pointwise/weights kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)   is_variable = 0 512 1 1 512
Tensor  27 MobilenetV1/Logits/AvgPool_1a/AvgPool kTfLiteFloat32  kTfLiteArenaRw       4096 bytes ( 0.0 MB)   is_variable = 0 1 1 1 1024
Tensor  28 MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd kTfLiteFloat32  kTfLiteArenaRw       4004 bytes ( 0.0 MB)   is_variable = 0 1 1 1 1001
Tensor  29 MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       4004 bytes ( 0.0 MB)   is_variable = 0 1001
Tensor  30 MobilenetV1/Logits/Conv2d_1c_1x1/weights kTfLiteFloat32   kTfLiteMmapRo    4100096 bytes ( 3.9 MB)   is_variable = 0 1001 1 1 1024
Tensor  31 MobilenetV1/Logits/SpatialSqueeze kTfLiteFloat32  kTfLiteArenaRw       4004 bytes ( 0.0 MB)   is_variable = 0 1 1001
Tensor  32 MobilenetV1/Logits/SpatialSqueeze_shape kTfLiteInt32   kTfLiteMmapRo          8 bytes ( 0.0 MB)   is_variable = 0 2
Tensor  33 MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)   is_variable = 0 32
Tensor  34 MobilenetV1/MobilenetV1/Conv2d_0/Relu6 kTfLiteFloat32  kTfLiteArenaRw    1605632 bytes ( 1.5 MB)   is_variable = 0 1 112 112 32
Tensor  35 MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  36 MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  37 MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  38 MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  39 MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  40 MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  41 MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  42 MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  43 MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     100352 bytes ( 0.1 MB)   is_variable = 0 1 7 7 512
Tensor  44 MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  45 MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)   is_variable = 0 1024
Tensor  46 MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     200704 bytes ( 0.2 MB)   is_variable = 0 1 7 7 1024
Tensor  47 MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     200704 bytes ( 0.2 MB)   is_variable = 0 1 7 7 1024
Tensor  48 MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)   is_variable = 0 1024
Tensor  49 MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)   is_variable = 0 1024
Tensor  50 MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     200704 bytes ( 0.2 MB)   is_variable = 0 1 7 7 1024
Tensor  51 MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw    1605632 bytes ( 1.5 MB)   is_variable = 0 1 112 112 32
Tensor  52 MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)   is_variable = 0 32
Tensor  53 MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)   is_variable = 0 64
Tensor  54 MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw    3211264 bytes ( 3.1 MB)   is_variable = 0 1 112 112 64
Tensor  55 MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     802816 bytes ( 0.8 MB)   is_variable = 0 1 56 56 64
Tensor  56 MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)   is_variable = 0 64
Tensor  57 MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)   is_variable = 0 128
Tensor  58 MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw    1605632 bytes ( 1.5 MB)   is_variable = 0 1 56 56 128
Tensor  59 MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw    1605632 bytes ( 1.5 MB)   is_variable = 0 1 56 56 128
Tensor  60 MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)   is_variable = 0 128
Tensor  61 MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)   is_variable = 0 128
Tensor  62 MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw    1605632 bytes ( 1.5 MB)   is_variable = 0 1 56 56 128
Tensor  63 MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 28 28 128
Tensor  64 MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)   is_variable = 0 128
Tensor  65 MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)   is_variable = 0 256
Tensor  66 MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     802816 bytes ( 0.8 MB)   is_variable = 0 1 28 28 256
Tensor  67 MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     802816 bytes ( 0.8 MB)   is_variable = 0 1 28 28 256
Tensor  68 MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)   is_variable = 0 256
Tensor  69 MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)   is_variable = 0 256
Tensor  70 MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     802816 bytes ( 0.8 MB)   is_variable = 0 1 28 28 256
Tensor  71 MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     200704 bytes ( 0.2 MB)   is_variable = 0 1 14 14 256
Tensor  72 MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)   is_variable = 0 256
Tensor  73 MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  74 MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  75 MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  76 MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  77 MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  78 MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  79 MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  80 MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  81 MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  82 MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  83 MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  84 MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  85 MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_bias kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)   is_variable = 0 512
Tensor  86 MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6 kTfLiteFloat32  kTfLiteArenaRw     401408 bytes ( 0.4 MB)   is_variable = 0 1 14 14 512
Tensor  87 MobilenetV1/Predictions/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw       4004 bytes ( 0.0 MB)   is_variable = 0 1 1001
Tensor  88 input                kTfLiteFloat32  kTfLiteArenaRw     602112 bytes ( 0.6 MB)   is_variable = 0 1 224 224 3
Tensor  89 (null)               kTfLiteFloat32  kTfLiteArenaRw    1354752 bytes ( 1.3 MB)   is_variable = 0 1 112 112 27
Tensor  90 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent       3456 bytes ( 0.0 MB)   is_variable = 0 27 32
Tensor  91 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent       8192 bytes ( 0.0 MB)   is_variable = 0 32 64
Tensor  92 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent      32768 bytes ( 0.0 MB)   is_variable = 0 64 128
Tensor  93 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent      65536 bytes ( 0.1 MB)   is_variable = 0 128 128
Tensor  94 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent     131072 bytes ( 0.1 MB)   is_variable = 0 128 256
Tensor  95 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent     262144 bytes ( 0.2 MB)   is_variable = 0 256 256
Tensor  96 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent     524288 bytes ( 0.5 MB)   is_variable = 0 256 512
Tensor  97 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    1048576 bytes ( 1.0 MB)   is_variable = 0 512 512
Tensor  98 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    1048576 bytes ( 1.0 MB)   is_variable = 0 512 512
Tensor  99 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    1048576 bytes ( 1.0 MB)   is_variable = 0 512 512
Tensor 100 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    1048576 bytes ( 1.0 MB)   is_variable = 0 512 512
Tensor 101 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    1048576 bytes ( 1.0 MB)   is_variable = 0 512 512
Tensor 102 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    2097152 bytes ( 2.0 MB)   is_variable = 0 512 1024
Tensor 103 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    4194304 bytes ( 4.0 MB)   is_variable = 0 1024 1024
Tensor 104 (null)               kTfLiteFloat32 kTfLiteArenaRwPersistent    4100096 bytes ( 3.9 MB)   is_variable = 0 1024 1001

Node   0 Operator Builtin Code   3
  Inputs: 88 0 33
  Outputs: 34
Node   1 Operator Builtin Code   4
  Inputs: 34 9 52
  Outputs: 51
Node   2 Operator Builtin Code   3
  Inputs: 51 10 53
  Outputs: 54
Node   3 Operator Builtin Code   4
  Inputs: 54 11 56
  Outputs: 55
Node   4 Operator Builtin Code   3
  Inputs: 55 12 57
  Outputs: 58
Node   5 Operator Builtin Code   4
  Inputs: 58 13 60
  Outputs: 59
Node   6 Operator Builtin Code   3
  Inputs: 59 14 61
  Outputs: 62
Node   7 Operator Builtin Code   4
  Inputs: 62 15 64
  Outputs: 63
Node   8 Operator Builtin Code   3
  Inputs: 63 16 65
  Outputs: 66
Node   9 Operator Builtin Code   4
  Inputs: 66 17 68
  Outputs: 67
Node  10 Operator Builtin Code   3
  Inputs: 67 18 69
  Outputs: 70
Node  11 Operator Builtin Code   4
  Inputs: 70 19 72
  Outputs: 71
Node  12 Operator Builtin Code   3
  Inputs: 71 20 73
  Outputs: 74
Node  13 Operator Builtin Code   4
  Inputs: 74 21 76
  Outputs: 75
Node  14 Operator Builtin Code   3
  Inputs: 75 22 77
  Outputs: 78
Node  15 Operator Builtin Code   4
  Inputs: 78 23 80
  Outputs: 79
Node  16 Operator Builtin Code   3
  Inputs: 79 24 81
  Outputs: 82
Node  17 Operator Builtin Code   4
  Inputs: 82 25 84
  Outputs: 83
Node  18 Operator Builtin Code   3
  Inputs: 83 26 85
  Outputs: 86
Node  19 Operator Builtin Code   4
  Inputs: 86 1 36
  Outputs: 35
Node  20 Operator Builtin Code   3
  Inputs: 35 2 37
  Outputs: 38
Node  21 Operator Builtin Code   4
  Inputs: 38 3 40
  Outputs: 39
Node  22 Operator Builtin Code   3
  Inputs: 39 4 41
  Outputs: 42
Node  23 Operator Builtin Code   4
  Inputs: 42 5 44
  Outputs: 43
Node  24 Operator Builtin Code   3
  Inputs: 43 6 45
  Outputs: 46
Node  25 Operator Builtin Code   4
  Inputs: 46 7 48
  Outputs: 47
Node  26 Operator Builtin Code   3
  Inputs: 47 8 49
  Outputs: 50
Node  27 Operator Builtin Code   1
  Inputs: 50
  Outputs: 27
Node  28 Operator Builtin Code   3
  Inputs: 27 30 29
  Outputs: 28
Node  29 Operator Builtin Code  22
  Inputs: 28 32
  Outputs: 31
Node  30 Operator Builtin Code  25
  Inputs: 31
  Outputs: 87

~~~
Can compare the interpreter info with model file diagram

![](static/img/workstuff/ai_tensorflowlite-model-structure_2.jpg)




