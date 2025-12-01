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

- ``MLXArray/all(keepDims:stream:)``
- ``MLXArray/all(axis:keepDims:stream:)``
- ``MLXArray/all(axes:keepDims:stream:)``
- ``MLXArray/any(keepDims:stream:)``
- ``MLXArray/any(axis:keepDims:stream:)``
- ``MLXArray/any(axes:keepDims:stream:)``

### MLXArray Aggregating Reduction Functions

- ``MLXArray/logSumExp(keepDims:stream:)``
- ``MLXArray/logSumExp(axis:keepDims:stream:)``
- ``MLXArray/logSumExp(axes:keepDims:stream:)``
- ``MLXArray/product(keepDims:stream:)``
- ``MLXArray/product(axis:keepDims:stream:)``
- ``MLXArray/product(axes:keepDims:stream:)``
- ``MLXArray/max(keepDims:stream:)``
- ``MLXArray/max(axis:keepDims:stream:)``
- ``MLXArray/max(axes:keepDims:stream:)``
- ``MLXArray/mean(keepDims:stream:)``
- ``MLXArray/mean(axis:keepDims:stream:)``
- ``MLXArray/mean(axes:keepDims:stream:)``
- ``MLXArray/min(keepDims:stream:)``
- ``MLXArray/min(axis:keepDims:stream:)``
- ``MLXArray/min(axes:keepDims:stream:)``
- ``MLXArray/sum(keepDims:stream:)``
- ``MLXArray/sum(axis:keepDims:stream:)``
- ``MLXArray/sum(axes:keepDims:stream:)``
- ``MLXArray/variance(keepDims:ddof:stream:)``
- ``MLXArray/variance(axis:keepDims:ddof:stream:)``
- ``MLXArray/variance(axes:keepDims:ddof:stream:)``

### Logical Reduction Free Functions

- ``all(_:keepDims:stream:)``
- ``all(_:axis:keepDims:stream:)``
- ``all(_:axes:keepDims:stream:)``
- ``any(_:keepDims:stream:)``
- ``any(_:axis:keepDims:stream:)``
- ``any(_:axes:keepDims:stream:)``

### Aggregating Reduction Free Functions

- ``logSumExp(_:keepDims:stream:)``
- ``logSumExp(_:axis:keepDims:stream:)``
- ``logSumExp(_:axes:keepDims:stream:)``
- ``product(_:keepDims:stream:)``
- ``product(_:axis:keepDims:stream:)``
- ``product(_:axes:keepDims:stream:)``
- ``max(_:keepDims:stream:)``
- ``max(_:axis:keepDims:stream:)``
- ``max(_:axes:keepDims:stream:)``
- ``mean(_:keepDims:stream:)``
- ``mean(_:axis:keepDims:stream:)``
- ``mean(_:axes:keepDims:stream:)``
- ``median(_:keepDims:stream:)``
- ``median(_:axis:keepDims:stream:)``
- ``median(_:axes:keepDims:stream:)``
- ``min(_:keepDims:stream:)``
- ``min(_:axis:keepDims:stream:)``
- ``min(_:axes:keepDims:stream:)``
- ``std(_:axes:keepDims:ddof:stream:)``
- ``std(_:axis:keepDims:ddof:stream:)``
- ``std(_:keepDims:ddof:stream:)``
- ``sum(_:keepDims:stream:)``
- ``sum(_:axis:keepDims:stream:)``
- ``sum(_:axes:keepDims:stream:)``
- ``variance(_:keepDims:ddof:stream:)``
- ``variance(_:axis:keepDims:ddof:stream:)``
- ``variance(_:axes:keepDims:ddof:stream:)``
