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
- ``hardShrink(_:lambda:)``
- ``hardSwish(_:)``
- ``hardTanH(_:min:max:)``
- ``leakyRelu(_:negativeSlope:)``
- ``logSigmoid(_:)``
- ``logSoftmax(_:axis:)``
- ``mish(_:)``
- ``prelu(_:alpha:)``
- ``relu(_:)``
- ``relu6(_:)``
- ``reluSquared(_:)``
- ``selu(_:)``
- ``sigmoid(_:)``
- ``silu(_:)``
- ``softmin(_:axis:)``
- ``softplus(_:)``
- ``softshrink(_:lambda:)``
- ``softsign(_:)``
- ``step(_:threshold:)``

### Activation Modules

- ``CELU``
- ``ELU``
- ``GELU``
- ``GLU``
- ``HardShrink``
- ``HardSwish``
- ``HardTanh``
- ``LeakyReLU``
- ``LogSigmoid``
- ``LogSoftmax``
- ``Mish``
- ``PReLU``
- ``ReLU``
- ``ReLU6``
- ``ReLUSquared``
- ``SELU``
- ``Sigmoid``
- ``SiLU``
- ``Softmax-63x8p``
- ``Softmin``
- ``Softplus``
- ``Softshrink``
- ``Softsign``
- ``Step``
- ``Tanh``

