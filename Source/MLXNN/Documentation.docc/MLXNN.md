# ``MLXNN``

## Overview

Some places to read to get started are:

- ``Module``
- <doc:custom-layers>
- <doc:training>

## Other MLX Packages

- [MLX](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/)
- [MLXRandom](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/)
- [MLXOptimizers](https://ml-explore.github.io/mlx-swift/MLXOptimizers/documentation/mlxoptimizers/)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)

## Topics

### Articles

- <doc:custom-layers>
- <doc:training>
- <doc:module-filters>

### Base Classes and Interfaces

- ``Module``
- ``UnaryLayer``

- ``ModuleParameters``
- ``ModuleChilren``
- ``ModuleItem``
- ``ModuleItems``
- ``ModuleValue``

### Unary Layers

Layers that provide an interface that takes a single MLXArray and produces a single MLXArray.
These can be used with ``Sequential``.

- ``Conv1d``
- ``Conv2d``
- ``Dropout``
- ``Dropout2d``
- ``Dropout3d``
- ``Embedding``
- ``Identity``
- ``Linear``
- ``QuantizedLinear``
- ``RoPE``
- ``RMSNorm``
- ``Sequential``

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
- ``logSoftMax(_:axis:)``
- ``mish(_:)``
- ``prelu(_:alpha:)``
- ``relu(_:)``
- ``relu6(_:)``
- ``selu(_:)``
- ``silu(_:)``
- ``sigmoid(_:)``
- ``softMax(_:)``
- ``softPlus(_:)``
- ``softSign(_:)``
- ``step(_:threshold:)``

### Activation Modules

- ``CELU``
- ``GELU``
- ``GLU``
- ``HardSwish``
- ``LeakyReLU``
- ``LogSigmoid``
- ``LogSoftMax``
- ``Mish``
- ``PReLU``
- ``ReLU``
- ``Relu6``
- ``SELU``
- ``SiLU``
- ``Sigmoid``
- ``SoftMax``
- ``SoftPlus``
- ``SoftSign``
- ``Step``
- ``Tanh``

### Loss Functions

- ``binaryCrossEntropy(logits:targets:reduction:)``
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

### Transformer Layers

- ``MultiHeadAttention``
- ``Transformer``

### Value and Grad

- ``valueAndGrad(model:_:)-12a2c``
- ``valueAndGrad(model:_:)-1w6x8``
- ``valueAndGrad(model:_:)-548r7``
