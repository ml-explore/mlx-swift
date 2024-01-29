# ``MLX/MLXArray``

An N dimensional array.  The main type in `mlx`.

## Thread Safety

> `MXArray` is not thread safe.

Although `MLXArray` looks like a normal multidimensional array, it is actually far more
sophisticated.  It actually holds a promise for future computations, see <doc:lazy-evaluation>
and is thus not thread safe.  For example:

```swift
let a: MLXArray
let b: MLXArray

let c = a + b
```

`c` is not the result of `a + b` but rather a graph representing `a` and `b` (which in turn
may be large unresolved graphs) and the `addition` operator that combines them.  It is not safe
to create `c` in one thread and consume/evaluate it in another.

## Topics

### MLXArray

- <doc:arithmetic>
- <doc:convolution>
- <doc:cumulative>
- <doc:indexes>
- <doc:indexing>
- <doc:initialization>
- <doc:logical>
- <doc:reduction>
- <doc:shapes>
- <doc:sorting>

### Arithmetic Operators

- ``+(_:_:)-1rv98``
- ``+(_:_:)-2vili``
- ``+(_:_:)-1jn5i``
- ``-(_:)``
- ``*(_:_:)-1z2ck``
- ``*(_:_:)-sw3w``
- ``*(_:_:)-7441r``
- ``**(_:_:)-8xxt3``
- ``**(_:_:)-6ve5u``
- ``**(_:_:)-4lp4b``

### Element-wise Arithmetic Functions

- ``abs(stream:)``
- ``cos(stream:)``
- ``exp(stream:)``
- ``floor(stream:)``
- ``floorDivide(_:stream:)``
- ``log(stream:)``
- ``log2(stream:)``
- ``log10(stream:)``
- ``log1p(stream:)``
- ``pow(_:stream:)``
- ``reciprocal(stream:)``
- ``rsqrt(stream:)``
- ``round(decimals:stream:)``
- ``sin(stream:)``
- ``sqrt(stream:)``
- ``square(stream:)``

### Shape Methods (Same Size)

Some methods allow you to manipulate the shape of the array.  These methods change the size
and ``shape`` of the dimensions without changing the number of elements or contents of the array:

- ``flattened(start:end:stream:)``
- ``reshaped(_:stream:)-19x5z``
- ``squeezed(axes:stream:)``

### Shape Methods (Change Size)

These methods manipulate the shape and contents of the array:

- ``movedAxis(source:destination:stream:)``
- ``split(parts:axis:stream:)``
- ``split(indices:axis:stream:)``
- ``swappedAxes(_:_:stream:)``
- ``transposed(axes:stream:)``

### Cumulative Methods

- ``cummax(axis:reverse:inclusive:stream:)``
- ``cummax(reverse:inclusive:stream:)``
- ``cummin(axis:reverse:inclusive:stream:)``
- ``cummin(reverse:inclusive:stream:)``
- ``cumprod(axis:reverse:inclusive:stream:)``
- ``cumprod(reverse:inclusive:stream:)``
- ``cumsum(axis:reverse:inclusive:stream:)``
- ``cumsum(reverse:inclusive:stream:)``

### Indexes

- ``argMax(axis:keepDims:stream:)``
- ``argMin(axis:keepDims:stream:)``
- ``subscript(_:stream:)-82jwt``
- ``take(_:axis:stream:)``

### Logical Functions

- ``all(axes:keepDims:stream:)``
- ``all(keepDims:stream:)``
- ``all(axis:keepDims:stream:)``
- ``any(axes:keepDims:stream:)``
- ``any(keepDims:stream:)``
- ``any(axis:keepDims:stream:)``
- ``allClose(_:rtol:atol:stream:)``
- ``allClose(_:_:rtol:atol:stream:)``
- ``arrayEqual(_:equalNAN:stream:)``
- ``.!(_:)``
- ``.==(_:_:)-56m0a``
- ``.==(_:_:)-79hbc``
- ``.!=(_:_:)-mbw0``
- ``.!=(_:_:)-gkdj``
- ``.<(_:_:)-9rzup``
- ``.<(_:_:)-54ivt``
- ``.<=(_:_:)-2a0s9``
- ``.<=(_:_:)-6vb92``
- ``.>(_:_:)-fwi1``
- ``.>(_:_:)-2v86b``
- ``.>=(_:_:)-2gqml``
- ``.>=(_:_:)-6zxj9``
- ``.&&(_:_:)``
- ``.||(_:_:)``

### Aggregating Reduction Functions

- ``logSumExp(axes:keepDims:stream:)``
- ``product(axis:keepDims:stream:)``
- ``max(axes:keepDims:stream:)``
- ``mean(axes:keepDims:stream:)``
- ``min(axes:keepDims:stream:)``
- ``sum(axes:keepDims:stream:)``
- ``variance(axes:keepDims:ddof:stream:)``
