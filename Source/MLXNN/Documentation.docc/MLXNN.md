# ``MLXNN``

## Overview

Two places to read to get started are:

- <doc:custom-layers>
- ``Module``

## Other MLX Packages

- [MLX](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/)
- [MLXRandom](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)

## Topics

### Base Classes and Interfaces

- ``Module``
- ``UnaryLayer``

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

### Optimizers

- ``AdaDelta``
- ``Adafactor``
- ``AdaGrad``
- ``AdamW``
- ``Adam``
- ``Adamax``
- ``Lion``
- ``RMSprop``
- ``SGD``


### Optimizer Base Classes and Protocols

- ``Optimizer``
- ``OptimizerBase``
- ``OptimizerBaseArrayState``
