# Activation Functions and Layers

Built-in activation functions and layers.

`MLXNN` provides a number of activation functions and modules.  The modules
simply wrap the functions, though some like ``GELU`` provide some settings
that select between different functions.  Others, like ``CELU`` encapsulate parameters
such as `alpha`.

## Topics

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
- ``SELU``
- ``SiLU``
- ``Sigmoid``
- ``Softmax-63x8p``
- ``Softplus``
- ``Softsign``
- ``Step``
- ``Tanh``

