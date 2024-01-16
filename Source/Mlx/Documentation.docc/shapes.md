# Shapes

Shape is a term to describe the number and size of the dimensions of an N dimension (ND) array.

``MLXArray`` is an N dimensional array.  The number of dimensions is described by ``MLXArray/ndim``
and the size of each dimension can be examined with ``MLXArray/dim(_:)-ywch`` or ``MLXArray/shape``.

## Topics

### MLXArray Shape Methods (Same Size)

Some methods allow you to manipulate the shape of the array.  These methods change the size
and ``MLXArray/shape`` of the dimensions without changing the number of elements or contents of the array:

- ``MLXArray/flatten(start:end:stream:)``
- ``MLXArray/reshape(_:stream:)``
- ``MLXArray/squeeze(axes:stream:)``

### MLXArray Shape Methods (Change Size)

These methods manipulate the shape and contents of the array:

- ``MLXArray/moveAxis(source:destination:stream:)``
- ``MLXArray/split(parts:axis:stream:)``
- ``MLXArray/split(indices:axis:stream:)``
- ``MLXArray/swapAxes(_:_:stream:)``
- ``MLXArray/transpose(axes:stream:)``

### Free Functions To Manipulate Shapes

TODO

- ``MLXArray/flatten(start:end:stream:)``

