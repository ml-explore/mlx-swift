# Layers

Built-in layers.

`MLXNN` provides a number of built-in layers that can be used to build models.  
See also <doc:activations> for Activation Layers and <doc:custom-layers> for examples of their use

## Topics

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
- ``RMSNorm``
- ``Sequential``

### Other Layers

- ``Bilinear``
- ``MultiHeadAttention``
