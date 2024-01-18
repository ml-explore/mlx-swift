# Shapes

Shape is a term to describe the number and size of the dimensions of an N dimension (ND) array.

``MLXArray`` is an N dimensional array.  The number of dimensions is described by ``MLXArray/ndim``
and the size of each dimension can be examined with ``MLXArray/dim(_:)-ywch`` or ``MLXArray/shape``.

Some of these functions can manipulate the shape without changing the contents while others 
will moves rows and columns as they modify the shape.

## Topics

### MLXArray Shape Methods (Same Size)

Some methods allow you to manipulate the shape of the array.  These methods change the size
and ``MLXArray/shape`` of the dimensions without changing the number of elements or contents of the array:

- ``MLXArray/flatten(start:end:stream:)``
- ``MLXArray/reshape(_:stream:)-uxps``
- ``MLXArray/squeeze(axes:stream:)``
- ``expandDimensions(_:axes:stream:)``
- ``asStrided(_:_:strides:offset:stream:)``

- ``flatten(_:start:end:stream:)``
- ``reshape(_:_:stream:)-8p51j``
- ``squeeze(_:axes:stream:)``

### MLXArray Shape Methods (Change Size)

These methods manipulate the shape and contents of the array:

- ``MLXArray/moveAxis(source:destination:stream:)``
- ``MLXArray/split(parts:axis:stream:)``
- ``MLXArray/split(indices:axis:stream:)``
- ``MLXArray/swapAxes(_:_:stream:)``
- ``MLXArray/transpose(axes:stream:)``

### Free Functions To Manipulate Shapes

- ``asStrided(_:_:strides:offset:stream:)``
- ``concatenate(_:axis:stream:)``
- ``expandDimensions(_:axes:stream:)``
- ``moveAxis(_:source:destination:stream:)``
- ``pad(_:width:value:stream:)``
- ``pad(_:widths:value:stream:)``
- ``split(_:indices:axis:stream:)``
- ``split(_:parts:axis:stream:)``
- ``stack(_:axis:stream:)``
- ``swapAxes(_:_:_:stream:)``
- ``transpose(_:axes:stream:)``
