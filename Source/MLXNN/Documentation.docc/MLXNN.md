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

