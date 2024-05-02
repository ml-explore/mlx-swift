# Sorting

Sorting and partitioning values and indices.

MLX has a number of methods that produce an array of indexes, including for sorting
and partitioning:

```swift
// array with values in random order
let array = MLXRandom.randInt(0 ..< 100, [10])

let sortIndexes = argSort(array, axis: -1)

// the array in sorted order
let sorted = array[sortIndexes]
```

There are other methods that produce a new sorted or partitioned array:

```swift
// array with values in random order
let array = MLXRandom.randInt(0 ..< 100, [10])

// the array in sorted order
let sorted = sort(array)
```

See related items in <doc:indexes>.

## Topics

### Index Producing Functions

These functions produce indexes for sorting and partitioning.  
``MLXArray/subscript(_:stream:)-375a0`` or ``MLXArray/take(_:stream:)`` must
be used to apply them to an array (if needed).

- ``argSort(_:axis:stream:)``
- ``argPartition(_:kth:axis:stream:)``

### Sorting Functions

These sort or partition the data directly (producing a new array).

- ``sorted(_:axis:stream:)``
- ``partitioned(_:kth:axis:stream:)``
