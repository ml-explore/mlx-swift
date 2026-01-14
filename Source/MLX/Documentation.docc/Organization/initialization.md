#  Initialization

Creating MLXArrays.

### Scalar Arrays

A scalar ``MLXArray`` is created from a scalar and has zero dimensions:

```swift
let v1 = MLXArray(true)
let v2 = MLXArray(7)
let v3 = MLXArray(8.5)
```

If an `MLXArray` of a different type is needed there is an initializer:

```swift
// dtype is .float32
let v4 = MLXArray(8.5)

// dtype is .float16
let v5 = MLXArray(Float16(8.5))

// dtype is .float16
let v6 = MLXArray(8.5, dtype: .float16)
```

Sometimes scalars can be used in place of arrays (no need to explicitly create them).
Some functions and operators that work on ``MLXArray`` take a ``ScalarOrArray`` argument or have
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

### Int vs Int32 vs Int64

In swift an Int is a 64 bit value (aka Int64).  You can get 32 bit values by using:

```swift
let i = Int32(10)
```

In MLX the preferred integer type is ``DType/int32`` or smaller.  You can create
an ``MLXArray`` with an Int32 like this:

```swift
let a = MLXArray(Int32(10))
```

but as a convenience you can also create them like this:

```swift
// also int32!
let a = MLXArray(10)
```

If the value is out of range you will get an error pointing
you to the alternate initializer:

```swift
// array creation with Int -- we want it to produce .int32
let a1 = MLXArray(500)
XCTAssertEqual(a1.dtype, .int32)

// eplicit int64
let a2 = MLXArray(int64: 500)
XCTAssertEqual(a2.dtype, .int64)
```

All of the `Int` initializers (e.g. `[Int]` and `Sequence<Int>`) work
the same way and all have the `int64:` variant.

### Double

If you have a `Double` array, you have to convert it as `MLXArray` does not support `Double`:

```swift
// this converts to a Float array behind the scenes
let v1 = MLXArray(converting: [0.1, 0.5])
```


### Multi Value Arrays

Typically MLXArrays are created with many values and potentially many dimensions.  You can create
an MLXArray from another array (literal in this case, but swift `Array` variables work as well):

```swift
// create an array of Int64 with shape [3]
let v1 = MLXArray([1, 2, 3])
```

You can also create an array from a swift `Sequence`:

```swift
// create an array of shape [12] from a sequence
let v1 = MLXArray(0 ..< 12)

// this works with various types of sequences
let v2 = MLXArray(stride(from: Float(0.5), to: Float(1.5), by: Float(0.1)))
```

If you have `Data` or a `UnsafePointer` (of various kinds) you can also create an `MLXArray`
from that:

```swift
let data = Data([1, 2, 3, 4])

// directly from Data
let v1 = MLXArray(data, type: UInt8.self)

// or via a pointer
let v2 = data.withUnsafeBytes { ptr in
    MLXArray(ptr, type: UInt8.self)
}
```

When creating using an array or sequence you can also control the shape:

```swift
let v1 = MLXArray(0 ..< 12, [3, 4])
```

### Random Value Arrays

See also `MLXRandom` for creating arrays with random data.

### Other Arrays

There are a number of factory methods to create common array patterns.  For example:

```swift
// an array full of zeros
let zeros = MLXArray.zeros([5, 5])

// 2-d identity array
let identity = MLXArray.identity(5)
```

### Complex Values

``MLXArray`` supports complex numbers, specifically a real and imaginary `Float32` 
as ``DType/complex64``.
MLX uses [swift-numerics](https://github.com/apple/swift-numerics/tree/main)
to represent the `Complex` type, though there are a few functions for manipulating
the individual pieces, see <doc:conversion>.

To create a complex scalar there are a few approaches:

```swift
let c1 = MLXArray(Complex(0, 1))
let c2 = MLXArray(real: 0, imaginary: 1)
```

You can use `Complex` to create an array of complex as well:

```swift
let c3 = MLXArray([Complex(2, 7), Complex(3, 8), Complex(4, 9)])
```

If you have two arrays that you want to combine you can use this pattern:

```swift
let r = MLXRandom.uniform(0.0 ..< 1.0, [100, 100])
let i = MLXRandom.uniform(0.0 ..< 1.0, [100, 100])

// dtype is .complex64
let c = r + i.asImaginary()
```

## Topics

### MLXArray Literal Initializers

- ``MLXArray/init(arrayLiteral:)``

### MLXArray Scalar Initializers

- ``MLXArray/init(_:)-(Int32)``
- ``MLXArray/init(_:)-(Bool)``
- ``MLXArray/init(_:)-(Float)``
- ``MLXArray/init(_:)-(Int)``
- ``MLXArray/init(_:)-(T)``
- ``MLXArray/init(_:dtype:)``
- ``MLXArray/init(bfloat16:)``

### MLXArray Int Overrides

Creating an ``MLXArray`` from `Int` will produce ``DType/int32`` rather
than ``DType/int64`` (`Int` is really `Int64`).  If you need ``DType/int64``
there are specific initializers to request it:

- ``MLXArray/init(_:)-(Int)``
- ``MLXArray/init(_:_:)-([Int],_)``
- ``MLXArray/init(int64:)``
- ``MLXArray/init(int64:_:)-([Int],_)``
- ``MLXArray/init(int64:_:)-(Sequence<Int>,_)``

### MLXArray Array Initializers

- ``MLXArray/init(_:_:)-([T],_)``
- ``MLXArray/init(_:_:)-(Sequence,_)``
- ``MLXArray/init(_:_:)-([Int],_)``
- ``MLXArray/init(converting:_:)``
- ``MLXArray/init(_:_:type:)-(UnsafeRawBufferPointer,_,_)``
- ``MLXArray/init(_:_:type:)-(Data,_,_)``

### MLXArray Complex Initializers

- ``MLXArray/init(real:imaginary:)``
- ``MLXArray/init(_:)-(Complex<Float>)``

### MLXArray Factory Methods

- ``MLXArray/zeros(_:type:stream:)``
- ``MLXArray/zeros(like:stream:)``
- ``MLXArray/zeros(_:dtype:stream:)``
- ``MLXArray/ones(_:type:stream:)``
- ``MLXArray/ones(like:stream:)``
- ``MLXArray/ones(_:dtype:stream:)``
- ``MLXArray/eye(_:m:k:type:stream:)``
- ``MLXArray/full(_:values:type:stream:)``
- ``MLXArray/full(_:values:stream:)``
- ``MLXArray/identity(_:type:stream:)``
- ``MLXArray/linspace(_:_:count:stream:)-(Int,Int,Int,StreamOrDevice)``
- ``MLXArray/linspace(_:_:count:stream:)-(Double,Double,Int,StreamOrDevice)``
- ``MLXArray/repeated(_:count:axis:stream:)``
- ``MLXArray/repeated(_:count:stream:)``
- ``MLXArray/repeat(_:count:axis:stream:)``
- ``MLXArray/repeat(_:count:stream:)``
- ``MLXArray/tri(_:m:k:type:stream:)``

### MLXArray Factory Free Methods

- ``MLX/zeros(_:type:stream:)``
- ``MLX/zeros(like:stream:)``
- ``MLX/ones(_:type:stream:)``
- ``MLX/ones(like:stream:)``
- ``MLX/eye(_:m:k:type:stream:)``
- ``MLX/full(_:values:type:stream:)``
- ``MLX/full(_:values:stream:)``
- ``MLX/identity(_:type:stream:)``
- ``MLX/linspace(_:_:count:stream:)-7vj0o``
- ``MLX/linspace(_:_:count:stream:)-6w959``
- ``MLXArray/repeated(_:count:axis:stream:)``
- ``MLXArray/repeated(_:count:stream:)``
- ``MLX/repeat(_:count:axis:stream:)``
- ``MLX/repeat(_:count:stream:)``
- ``MLX/tri(_:m:k:type:stream:)``
