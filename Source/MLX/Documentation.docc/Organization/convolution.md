# Convolution

Convolution operations.

## Topics

### Convolution Functions

MLX has several functions to support convolutions:

- ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
- ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
- ``conv3d(_:_:stride:padding:dilation:groups:stream:)``
- ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-(MLXArray,MLXArray,IntOrArray,IntOrArray,IntOrArray,IntOrArray,Int,Bool,StreamOrDevice)``
- ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-(MLXArray,MLXArray,IntOrArray,(Int,Int),IntOrArray,IntOrArray,Int,Bool,StreamOrDevice)``
- ``convTransposed1d(_:_:stride:padding:dilation:outputPadding:groups:stream:)``
- ``convTransposed2d(_:_:stride:padding:dilation:outputPadding:groups:stream:)``
- ``convTransposed3d(_:_:stride:padding:dilation:outputPadding:groups:stream:)``
- ``convolve(_:_:mode:stream:)``
