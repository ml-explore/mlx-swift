#  Initialization

Creating MLXArrays.

### Implicit Arrays

MLXArrays can be created implicitly from literals.  You can create scalar
arrays by using a scalar literal in a context where an MLXArray is needed:

```swift
let array = MLXArray(0 ..< 12, [4, 3])

// equivalent to array + MLXArray(3)
let r = array + 3

// or as a parameter
let r2 = divide(array, 3.0)
```

You can do the same with a scalar array of integers (`Int32`).  This does
not work for `Float` arrays -- use ``MLXArray/init(_:_:)-1nwrn`` for those.

```swift
let array = MLXArray(0 ..< 12, [4, 3])

// equivalent to array + MLXArray([1, 2, 3])
let r = array + [1, 2, 3]
```

### Scalar Arrays

A scalar MLXArray is created from a scalar and has zero dimensions:

```swift
let v1 = MLXArray(true)
let v2 = MLXArray(7)
let v3 = MLXArray(8.5)
```

Typically you can create these through the scalar literal conversions (previous section).

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

If you have a `Double` array, you have to convert it as `MLXArray` does not support `Double`:

```swift
// this converts to a Float array behind the scenes
let v1 = MLXArray(converting: [0.1, 0.5])
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

See also ``MLXRandom`` for creating arrays with random data.

### Other Arrays

There are a number of factory methods to create common array patterns.  For example:

```swift
// an array full of zeros
let zeros = MLXArray.zeros([5, 5])

// 2-d identity array
let identity = MLXArray.identity(5)
```

## Topics

### MLXArray Literal Initializers

- ``MLXArray/init(floatLiteral:)``
- ``MLXArray/init(booleanLiteral:)``
- ``MLXArray/init(integerLiteral:)``
- ``MLXArray/init(arrayLiteral:)``

### MLXArray Scalar Initializers

- ``MLXArray/init(_:)-20ctj``
- ``MLXArray/init(_:)-9h4g2``
- ``MLXArray/init(_:)-7t81v``
- ``MLXArray/init(_:)-9jjm5``

### MLXArray Array Initializers

- ``MLXArray/init(_:_:)-9z9ix``
- ``MLXArray/init(_:_:)-3roe8``
- ``MLXArray/init(_:_:)-1nwrn``
- ``MLXArray/init(converting:_:)``
- ``MLXArray/init(_:_:type:)-22a1g``
- ``MLXArray/init(_:_:type:)-7rglc``

### MLXArray Factory Methods

- ``MLXArray/zeros(_:type:stream:)``
- ``MLXArray/zeros(like:stream:)``
- ``MLXArray/ones(_:type:stream:)``
- ``MLXArray/ones(like:stream:)``
- ``MLXArray/eye(_:m:k:type:stream:)``
- ``MLXArray/full(_:values:type:stream:)``
- ``MLXArray/full(_:values:stream:)``
- ``MLXArray/identity(_:type:stream:)``
- ``MLXArray/linspace(_:_:count:stream:)-32sbl``
- ``MLXArray/linspace(_:_:count:stream:)-1m270``
- ``MLXArray/repeat(_:count:axis:stream:)``
- ``MLXArray/repeat(_:count:stream:)``
- ``MLXArray/triangle(_:m:k:type:stream:)``
