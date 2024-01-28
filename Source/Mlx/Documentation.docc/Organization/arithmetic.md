# Arithmetic

MLX supports a wide range of binary arithmetic operators.

Many of the operations are avilable as infix operators (e.g. '+') or
as functions, either on MLXArray itself or as free functions.

```swift
let a = MLXArray(0 ..< 12, [4, 3])
let b = MLXArray([4, 5, 6])

// these are equivalent
let r1 = a + b + 7
let r2 = add(add(a, b), 7)
```

There are also a wide variety of element-wise math functions:

```swift
let a = MLXArray(0 ..< 12, [4, 3])

let r = log(a)
```

> There are two operators from python that are not supported (as operators) in
swift: `@` (``matmul(_:_:stream:)``) and `//` (``floorDivide(_:_:stream:)``).  Please
use the methods on `MLXArray` or the free functions.

## Topics

### MLXArray Operators

- ``MLXArray/+(_:_:)``
- ``MLXArray/-(_:_:)``
- ``MLXArray/-(_:)``
- ``MLXArray/!(_:)``
- ``MLXArray/*(_:_:)``
- ``MLXArray/**(_:_:)``
- ``MLXArray/.==(_:_:)``
- ``MLXArray/.!=(_:_:)``
- ``MLXArray/.<(_:_:)``
- ``MLXArray/.<=(_:_:)``
- ``MLXArray/.>(_:_:)``
- ``MLXArray/.>=(_:_:)``
- ``MLXArray/.&&(_:_:)``
- ``MLXArray/.||(_:_:)``

### MLXArray Element-wise Arithmetic Functions

- ``MLXArray/abs(stream:)``
- ``MLXArray/cos(stream:)``
- ``MLXArray/exp(stream:)``
- ``MLXArray/floor(stream:)``
- ``MLXArray/floorDivide(_:stream:)``
- ``MLXArray/log(stream:)``
- ``MLXArray/log2(stream:)``
- ``MLXArray/log10(stream:)``
- ``MLXArray/log1p(stream:)``
- ``MLXArray/pow(_:stream:)``
- ``MLXArray/reciprocal(stream:)``
- ``MLXArray/rsqrt(stream:)``
- ``MLXArray/round(decimals:stream:)``
- ``MLXArray/sin(stream:)``
- ``MLXArray/sqrt(stream:)``
- ``MLXArray/square(stream:)``

### Element-wise Arithmetic Free Functions

- ``abs(_:stream:)``
- ``acos(_:stream:)``
- ``acosh(_:stream:)``
- ``add(_:_:stream:)``
- ``asin(_:stream:)``
- ``asinh(_:stream:)``
- ``atan(_:stream:)``
- ``atanh(_:stream:)``
- ``ceil(_:stream:)``
- ``clip(_:min:max:stream:)``
- ``cos(_:stream:)``
- ``cosh(_:stream:)``
- ``divide(_:_:stream:)``
- ``erf(_:stream:)``
- ``erfInverse(_:stream:)``
- ``exp(_:stream:)``
- ``floor(_:stream:)``
- ``floorDivide(_:_:stream:)``
- ``log(_:stream:)``
- ``log10(_:stream:)``
- ``log1p(_:stream:)``
- ``log2(_:stream:)``
- ``logAddExp(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``matmul(_:_:stream:)``
- ``maximum(_:_:stream:)``
- ``minimum(_:_:stream:)``
- ``multiply(_:_:stream:)``
- ``negative(_:stream:)``
- ``notEqual(_:_:stream:)``
- ``pow(_:_:stream:)-7pe7j``
- ``pow(_:_:stream:)-49xi0``
- ``pow(_:_:stream:)-8ie9c``
- ``reciprocal(_:stream:)``
- ``remainder(_:_:stream:)``
- ``round(_:decimals:stream:)``
- ``rsqrt(_:stream:)``
- ``sigmoid(_:stream:)``
- ``sign(_:stream:)``
- ``sin(_:stream:)``
- ``sinh(_:stream:)``
- ``softMax(_:axes:stream:)``
- ``sqrt(_:stream:)``
- ``square(_:stream:)``
- ``subtract(_:_:stream:)``
- ``tan(_:stream:)``
- ``tanh(_:stream:)``
- ``where(_:_:_:stream:)``

