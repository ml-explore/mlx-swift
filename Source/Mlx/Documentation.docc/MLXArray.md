#  ``MLXArray``

bla blah blah

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
