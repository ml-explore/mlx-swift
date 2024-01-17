#  Reduction Functions

Reduction or aggregation functions.

MLX has a number of functions to reduce or aggregate data in `MLXArray`.
These functions typically work over one or more axes, though there are
overloads where the axis can be omitted and the work occurs on the entire array.

For example:

```swift
let array = MLXArray(0 ..< 12, [4, 3])

// scalar array with the sum of all the values
let totalSum = array.sum()

// array with the sum of the colums
let columnSum = array.sum(axis: 0)
```

See also <doc:logical> and <doc:cumulative>

## Topics

### MLXArray Logical Reduction Functions

- ``MLXArray/all(axes:keepDims:stream:)``
- ``MLXArray/any(axes:keepDims:stream:)``

### MLXArray Aggregating Reduction Functions

- ``MLXArray/logSumExp(axes:keepDims:stream:)``
- ``MLXArray/product(axis:keepDims:stream:)``
- ``MLXArray/max(axes:keepDims:stream:)``
- ``MLXArray/mean(axes:keepDims:stream:)``
- ``MLXArray/min(axes:keepDims:stream:)``
- ``MLXArray/sum(axes:keepDims:stream:)``
- ``MLXArray/variance(axes:keepDims:ddof:stream:)``

### Logical Reduction Free Functions

- ``all(_:axes:keepDims:stream:)``
- ``any(_:axes:keepDims:stream:)``

### Aggregating Reduction Free Functions

- ``logSumExp(_:axes:keepDims:stream:)``
- ``product(_:axis:keepDims:stream:)``
- ``max(_:axes:keepDims:stream:)``
- ``mean(_:axes:keepDims:stream:)``
- ``min(_:axes:keepDims:stream:)``
- ``sum(_:axes:keepDims:stream:)``
- ``variance(_:axes:keepDims:ddof:stream:)``
