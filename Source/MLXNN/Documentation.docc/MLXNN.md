# ``MLXNN``

Neural Networks support for MLX

## Overview

Writing arbitrarily complex neural networks in MLX can be done using only 
`MLXArray` and `valueAndGrad()`. However, this requires the user to write 
again and again the same simple neural network operations as well as handle 
all the parameter state and initialization manually and explicitly.

The `MLXNN` package solves this problem by providing an intuitive way 
of composing neural network layers, initializing their parameters, freezing
them for finetuning and more.

## Modules

The workhorse of any neural network library is the ``Module`` class. In MLX 
the ``Module`` class is a container of `MLXArray` or ``Module`` instances. Its 
main function is to provide a way to recursively access and update its
parameters and those of its submodules.

- ``Module``
- <doc:custom-layers>

### Parameters

A parameter of a module is any member of type `MLXArray` (its name should 
not start with `_`). It can be nested in other ``Module`` instances 
or `Array` and `Dictionary`.

``Module/parameters()`` can be used to extract a `NestedDictionary` 
(``ModuleParameters``) with all the parameters of a module and its submodules.

A Module can also keep track of “frozen” parameters. See the
``Module/freeze(recursive:keys:strict:)`` method for more details.
``valueAndGrad(model:_:)-12a2c`` the gradients returned will be with
respect to these trainable parameters.

### Training

See <doc:training>

## Other MLX Packages

- [MLX](mlx)
- [MLXOptimizers](mlxoptimizers)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)

## Topics

### Articles

- <doc:custom-layers>
- <doc:training>

### Base Classes and Interfaces

- ``Module``
- ``UnaryLayer``
- ``Quantizable``

- ``ModuleInfo``
- ``ParameterInfo``

- ``ModuleParameters``
- ``ModuleChildren``
- ``ModuleItem``
- ``ModuleItems``
- ``ModuleValue``

### Unary Layers

Layers that provide an interface that takes a single MLXArray and produces a single MLXArray.
These can be used with ``Sequential``.

- ``AvgPool1d``
- ``AvgPool2d``
- ``Conv1d``
- ``Conv2d``
- ``Dropout``
- ``Dropout2d``
- ``Dropout3d``
- ``Embedding``
- ``Identity``
- ``Linear``
- ``MaxPool1d``
- ``MaxPool2d``
- ``QuantizedLinear``
- ``RoPE``
- ``RMSNorm``
- ``Sequential``

### Sampling

- ``Upsample``

### Recurrent

- ``RNN``
- ``GRU``
- ``LSTM``

### Other Layers

- ``Bilinear``
- ``MultiHeadAttention``

### Activation Free Functions

- ``celu(_:alpha:)``
- ``elu(_:alpha:)``
- ``gelu(_:)``
- ``geluApproximate(_:)``
- ``geluFastApproximate(_:)``
- ``glu(_:axis:)``
- ``hardSwish(_:)``
- ``leakyRelu(_:negativeSlope:)``
- ``logSigmoid(_:)``
- ``logSoftmax(_:axis:)``
- ``mish(_:)``
- ``prelu(_:alpha:)``
- ``relu(_:)``
- ``relu6(_:)``
- ``reluSquared(_:)``
- ``selu(_:)``
- ``silu(_:)``
- ``sigmoid(_:)``
- ``softplus(_:)``
- ``softsign(_:)``
- ``step(_:threshold:)``

### Activation Modules

- ``CELU``
- ``GELU``
- ``GLU``
- ``HardSwish``
- ``LeakyReLU``
- ``LogSigmoid``
- ``LogSoftmax``
- ``Mish``
- ``PReLU``
- ``ReLU``
- ``ReLU6``
- ``ReLUSquared``
- ``SELU``
- ``SiLU``
- ``Sigmoid``
- ``Softmax``
- ``Softplus``
- ``Softsign``
- ``Step``
- ``Tanh``

### Loss Functions

- ``binaryCrossEntropy(logits:targets:weights:withLogits:reduction:)``
- ``cosineSimilarityLoss(x1:x2:axis:eps:reduction:)``
- ``crossEntropy(logits:targets:weights:axis:labelSmoothing:reduction:)``
- ``hingeLoss(inputs:targets:reduction:)``
- ``huberLoss(inputs:targets:delta:reduction:)``
- ``klDivLoss(inputs:targets:axis:reduction:)``
- ``l1Loss(predictions:targets:reduction:)``
- ``logCoshLoss(inputs:targets:reduction:)``
- ``mseLoss(predictions:targets:reduction:)``
- ``nllLoss(inputs:targets:axis:reduction:)``
- ``smoothL1Loss(predictions:targets:beta:reduction:)``
- ``tripletLoss(anchors:positives:negatives:axis:p:margin:eps:reduction:)``

### Normalization Layers

- ``InstanceNorm``
- ``LayerNorm``
- ``RMSNorm``
- ``GroupNorm``
- ``BatchNorm``

### Positional Encoding Layers

- ``RoPE``
- ``SinusoidalPositionalEncoding``
- ``ALiBi``

### Transformer Layers

- ``MultiHeadAttention``
- ``Transformer``

### Value and Grad

- ``valueAndGrad(model:_:)-12a2c``
- ``valueAndGrad(model:_:)-548r7``
- ``valueAndGrad(model:_:)-45dg5``
