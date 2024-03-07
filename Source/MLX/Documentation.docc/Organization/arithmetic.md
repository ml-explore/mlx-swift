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

## Swift Naming

Note that the element-wise logical operations such as:

- ``MLXArray/.==(_:_:)-56m0a``
- ``MLXArray/.==(_:_:)-79hbc``

are named using the Swift convention for SIMD operations, e.g. `.==`, `.<`, etc.  These
operators produce a new ``MLXArray`` with `true`/`false` values for the elementwise comparison.

## Operations With Scalars

Many functions and operators that work on ``MLXArray`` take a ``ScalarOrArray`` argument or have
an overload that does.  A sampling:

- ``MLXArray/+(_:_:)-2vili``
- ``MLXArray/+(_:_:)-1jn5i``
- ``MLX/minimum(_:_:stream:)``
- ``MLX/pow(_:_:stream:)-7pe7j``
- ``MLX/pow(_:_:stream:)-49xi0``

``ScalarOrArray`` is a protocol that various numeric types (`Int`, `Float`, etc.) implement and it
provides a method to convert the scalar to an ``MLXArray`` using a suggested ``DType``.  This allows:

```swift
let values: [Float16] = [ 0.5, 1.0, 2.5 ]

// a has dtype .float16
let a = MLXArray(values)

// b also has dtype .float16 because this translates (roughly) to:
// t = Int(3).asMLXArray(dtype: .float16)
// let b = a + t
let b = a + 3
```

Scalars will not promote results to `float32` using these functions.

## Topics

### MLXArray Operators

Note: the `-` and `/` operators are not able to be linked here.

- ``MLXArray/+(_:_:)-1rv98``
- ``MLXArray/+(_:_:)-2vili``
- ``MLXArray/+(_:_:)-1jn5i``
- ``MLXArray/-(_:)``
- ``MLXArray/*(_:_:)-1z2ck``
- ``MLXArray/*(_:_:)-sw3w``
- ``MLXArray/*(_:_:)-7441r``
- ``MLXArray/**(_:_:)-8xxt3``
- ``MLXArray/**(_:_:)-6ve5u``
- ``MLXArray/**(_:_:)-4lp4b``
- ``MLXArray/%(_:_:)-3ubwd``
- ``MLXArray/%(_:_:)-516wd``
- ``MLXArray/%(_:_:)-8az7l``
- ``MLXArray/.!(_:)``
- ``MLXArray/.==(_:_:)-56m0a``
- ``MLXArray/.==(_:_:)-79hbc``
- ``MLXArray/.!=(_:_:)-mbw0``
- ``MLXArray/.!=(_:_:)-gkdj``
- ``MLXArray/.<(_:_:)-9rzup``
- ``MLXArray/.<(_:_:)-54ivt``
- ``MLXArray/.<=(_:_:)-2a0s9``
- ``MLXArray/.<=(_:_:)-6vb92``
- ``MLXArray/.>(_:_:)-fwi1``
- ``MLXArray/.>(_:_:)-2v86b``
- ``MLXArray/.>=(_:_:)-2gqml``
- ``MLXArray/.>=(_:_:)-6zxj9``
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
- ``addmm(_:_:_:alpha:beta:stream:)``
- ``asin(_:stream:)``
- ``asinh(_:stream:)``
- ``atan(_:stream:)``
- ``atanh(_:stream:)``
- ``ceil(_:stream:)``
- ``clip(_:min:max:stream:)``
- ``cos(_:stream:)``
- ``cosh(_:stream:)``
- ``divide(_:_:stream:)``
- ``divmod(_:_:stream:)``
- ``erf(_:stream:)``
- ``erfInverse(_:stream:)``
- ``exp(_:stream:)``
- ``floor(_:stream:)``
- ``floorDivide(_:_:stream:)``
- ``isNaN(_:stream:)``
- ``isInf(_:stream:)``
- ``isPosInf(_:stream:)``
- ``isNegInf(_:stream:)``
- ``log(_:stream:)``
- ``log10(_:stream:)``
- ``log1p(_:stream:)``
- ``log2(_:stream:)``
- ``logAddExp(_:_:stream:)``
- ``logicalAnd(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``logicalOr(_:_:stream:)``
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
- ``which(_:_:_:stream:)``

### Vector, Matrix, and Tensor Products

- ``MLXArray/matmul(_:stream:)``
- ``matmul(_:_:stream:)``
- ``inner(_:_:stream:)``
- ``outer(_:_:stream:)``
- ``tensordot(_:_:axes:stream:)-3qkgq``
- ``tensordot(_:_:axes:stream:)-8yqyi``
