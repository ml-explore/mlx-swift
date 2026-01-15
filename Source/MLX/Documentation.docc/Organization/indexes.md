# Indexes

Functions to produce and consume array indexes.

MLX has a number of functions (typically named `argX()`) that can produce array 
indices and a few functions that can consume them.

```swift
// array with values in random order
let array = MLXRandom.randInt(0 ..< 100, [10])

let sortIndexes = argSort(array, axis: -1)

// the array in sorted order
let sorted = array[sortIndexes]
```

## Topics

### Index Producing Functions

- ``MLXArray/argMax(keepDims:stream:)``
- ``MLXArray/argMax(axis:keepDims:stream:)``
- ``MLXArray/argMin(keepDims:stream:)``
- ``MLXArray/argMin(axis:keepDims:stream:)``
- ``argMax(_:keepDims:stream:)``
- ``argMax(_:axis:keepDims:stream:)``
- ``argMin(_:keepDims:stream:)``
- ``argMin(_:axis:keepDims:stream:)``
- ``argPartition(_:kth:stream:)``
- ``argPartition(_:kth:axis:stream:)``
- ``argSort(_:stream:)``
- ``argSort(_:axis:stream:)``

### Index Consuming Functions

- ``MLXArray/subscript(_:stream:)-(MLXArrayIndex,StreamOrDevice)``
- ``MLXArray/take(_:axis:stream:)``
- ``takeAlong(_:_:axis:stream:)``
