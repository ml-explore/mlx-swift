# Cumulative Operations

Operations that produce a cumulative result.

There are a number of functions that can produce a cumulative result.  For example:

```swift
// [0, 1, 2, 3, 4]
let array = MLXArray(0 ..< 5)

// [0, 1, 3, 6, 10]
let result = array.cumsum()
```

These are available as both methods on `MLXArray` and free functions.  They each have options to:

- perform the computation along an axis or over the flattened array
- reverse the direction of cumulative computation

## Topics

### MLXArray Methods

- ``MLXArray/cummax(axis:reverse:inclusive:stream:)``
- ``MLXArray/cummax(reverse:inclusive:stream:)``
- ``MLXArray/cummin(axis:reverse:inclusive:stream:)``
- ``MLXArray/cummin(reverse:inclusive:stream:)``
- ``MLXArray/cumprod(axis:reverse:inclusive:stream:)``
- ``MLXArray/cumprod(reverse:inclusive:stream:)``
- ``MLXArray/cumsum(axis:reverse:inclusive:stream:)``
- ``MLXArray/cumsum(reverse:inclusive:stream:)``

### Free Functions

- ``cummax(_:axis:reverse:inclusive:stream:)``
- ``cummax(_:reverse:inclusive:stream:)``
- ``cummin(_:axis:reverse:inclusive:stream:)``
- ``cummin(_:reverse:inclusive:stream:)``
- ``cumprod(_:axis:reverse:inclusive:stream:)``
- ``cumprod(_:reverse:inclusive:stream:)``
- ``cumsum(_:axis:reverse:inclusive:stream:)``
- ``cumsum(_:reverse:inclusive:stream:)``
