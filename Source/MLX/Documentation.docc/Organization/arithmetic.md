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

- ``MLXArray/.==(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.==(_:_:)-(MLXArray,ScalarOrArray)``

are named using the Swift convention for SIMD operations, e.g. `.==`, `.<`, etc.  These
operators produce a new ``MLXArray`` with `true`/`false` values for the elementwise comparison.

## Operations With Scalars

Many functions and operators that work on ``MLXArray`` take a ``ScalarOrArray`` argument or have
an overload that does.  A sampling:

- ``MLXArray/+(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/+(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLX/minimum(_:_:stream:)``
- ``MLX/pow(_:_:stream:)-(MLXArray,ScalarOrArray,_)``
- ``MLX/pow(_:_:stream:)-(ScalarOrArray,MLXArray,_)``

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

- ``MLXArray/+(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/+(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/+(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/-(_:)``
- ``MLXArray/*(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/*(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/*(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/**(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/**(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/**(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/%(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/%(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/%(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.!(_:)``
- ``MLXArray/.==(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.==(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.!=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.!=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.<(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.<(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.<=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.<=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.>(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.>(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.>=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.>=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.&&(_:_:)``
- ``MLXArray/.||(_:_:)``
- ``MLXArray/~(_:)``
- ``MLXArray/&(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/&(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/&(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/|(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/|(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/|(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/^(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/^(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/^(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/<<(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/<<(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/<<(_:_:)-(ScalarOrArray,MLXArray)``
- ``MLXArray/>>(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/>>(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/>>(_:_:)-(ScalarOrArray,MLXArray)``

### MLXArray Element-wise Arithmetic Functions

- ``MLXArray/abs(stream:)``
- ``MLXArray/conjugate(stream:)``
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
- ``atan2(_:_:stream:)``
- ``atanh(_:stream:)``
- ``bitwiseAnd(_:_:stream:)``
- ``bitwiseInvert(_:stream:)``
- ``bitwiseOr(_:_:stream:)``
- ``bitwiseXOr(_:_:stream:)``
- ``ceil(_:stream:)``
- ``clip(_:min:max:stream:)``
- ``conjugate(_:stream:)``
- ``cos(_:stream:)``
- ``cosh(_:stream:)``
- ``degrees(_:stream:)``
- ``divide(_:_:stream:)``
- ``divmod(_:_:stream:)``
- ``erf(_:stream:)``
- ``erfInverse(_:stream:)``
- ``exp(_:stream:)``
- ``expm1(_:stream:)``
- ``floor(_:stream:)``
- ``floorDivide(_:_:stream:)``
- ``isNaN(_:stream:)``
- ``isInf(_:stream:)``
- ``isFinite(_:stream:)``
- ``isPosInf(_:stream:)``
- ``isNegInf(_:stream:)``
- ``leftShift(_:_:stream:)``
- ``log(_:stream:)``
- ``log10(_:stream:)``
- ``log1p(_:stream:)``
- ``log2(_:stream:)``
- ``logAddExp(_:_:stream:)``
- ``logicalAnd(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``logicalOr(_:_:stream:)``
- ``maximum(_:_:stream:)``
- ``minimum(_:_:stream:)``
- ``multiply(_:_:stream:)``
- ``nanToNum(_:nan:posInf:negInf:stream:)``
- ``negative(_:stream:)``
- ``notEqual(_:_:stream:)``
- ``pow(_:_:stream:)-(MLXArray,ScalarOrArray,_)``
- ``pow(_:_:stream:)-(ScalarOrArray,MLXArray,_)``
- ``pow(_:_:stream:)-(MLXArray,MLXArray,_)``
- ``radians(_:stream:)``
- ``reciprocal(_:stream:)``
- ``remainder(_:_:stream:)``
- ``rightShift(_:_:stream:)``
- ``round(_:decimals:stream:)``
- ``rsqrt(_:stream:)``
- ``sigmoid(_:stream:)``
- ``sign(_:stream:)``
- ``sin(_:stream:)``
- ``sinh(_:stream:)``
- ``softmax(_:axes:precise:stream:)``
- ``sqrt(_:stream:)``
- ``square(_:stream:)``
- ``subtract(_:_:stream:)``
- ``tan(_:stream:)``
- ``tanh(_:stream:)``
- ``trace(_:offset:axis1:axis2:dtype:stream:)``
- ``which(_:_:_:stream:)``

### Vector, Matrix, and Tensor Products

- ``MLXArray/matmul(_:stream:)``
- ``matmul(_:_:stream:)``
- ``gatherMM(_:_:lhsIndices:rhsIndices:sortedIndices:stream:)``
- ``blockMaskedMM(_:_:blockSize:maskOut:maskLHS:maskRHS:stream:)``
- ``addMM(_:_:_:alpha:beta:stream:)``
- ``quantizedMM(_:_:scales:biases:transpose:groupSize:bits:mode:stream:)``
- ``gatherQuantizedMM(_:_:scales:biases:lhsIndices:rhsIndices:transpose:groupSize:bits:mode:sortedIndices:stream:)``
- ``quantizedQuantizedMM(_:_:scales:groupSize:bits:mode:stream:)``
- ``inner(_:_:stream:)``
- ``outer(_:_:stream:)``
- ``tensordot(_:_:axes:stream:)-(MLXArray,MLXArray,Int,StreamOrDevice)``
- ``tensordot(_:_:axes:stream:)-(MLXArray,MLXArray,((Int,Int),(Int,Int)),StreamOrDevice)``
