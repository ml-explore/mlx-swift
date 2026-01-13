# Indexing

Array subscripts.

``MLXArray`` supports all the same indexing (see <doc:indexing>) as 
the python `mx.array`, though in some cases they are written differently.
In all cases both `MLXArray` and `mx.array` indexing strive to match
[numpy indexing](https://numpy.org/doc/stable/user/basics.indexing.html).

The full range of operations and their equivalent to python are:

```swift
let array = MLXArray(0 ..< 512, [8, 8, 8])

// index by integer
array[1]

// index by multiple integers
array[1, 3]

// index by range expression
// python: [1:5]
array[1 ..< 5]

// full range slice
// python: [:]
array[0 ...]

// slice with stride of 2
// python: [::2]
array[.stride(by: 2)]

// ellipsis operator (consume all remaining axes)
// python: [..., 3]
array[.ellipsis, 3]

// newaxis operator (insert a new axis of size 1)
// python: [None]
array[.newAxis]

// using another MLXArray as an index
let i = MLXArray([1, 2])
array[i]
```

These can be combined in any way with the following restrictions:

- `.ellipsis` can only be used once in an indexing operation
- `.newAxis` cannot be used in a set operation, e.g. `array[.newAxis] = MLXArray(1)` is invalid
- the number of axes given must be equal or less than the number of axes in the source array

### Set

The same operation that reads an array can also be used to update it:

```swift
let array = MLXArray(0 ..< 512, [8, 8, 8])

let a2 = array[5]
array[5] = MLXArray(100)
```

The assignment is performed using <doc:broadcasting>:

```swift
var a = MLXArray(0 ..< 512, [8, 8, 8])

// sets an [8, 8] area to 7 (broadcasting)
a[1] = MLXArray(7)
```

### Advanced Indexing

[Numpy advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing)
is also usable.  The simplest form allows using an `MLXArray` as indices (see <doc:indexes>):

```swift
// array with values in random order
let array = MLXRandom.randInt(0 ..< 100, [10])

let sortIndexes = argSort(array, axis: -1)

// the array in sorted order
let sorted = array[sortIndexes]
```

More complex forms are also available if multiple arrays are passed.

## Idiomatic Expressions From Python

### Negative Indexes

There are a number of idiomatic indexing expressions from python and numpy that are supported.  <doc:converting-python>
gives a table of common expressions.  This document seeks to explain some of them.

In python a negative index is used to mean "from end of array".  Thus `array[-1]` reads the last element of the array.
This does not work generally in swift but it does work with ``MLXArray`` in the following cases:

- `array[-1]` -- reads the last element of the first axis of the array
- `array.dim(-1)` -- reads the last dimension of the array's shape.  This is equivalent to `array.shape[-1]` in python, but that is not allowed here because the `shape` is a normal swift `Array`
- `array[-3 ..< -1]` -- iterates from the third-to-last index up to the last index.  This is equivalent to `array[-3:-1]` in python and follows the normal pattern for translating python slices to swift.  Note that parenthesis may be required in some cases like `array[(-3)...]`
- any function that takes an axis or axes, e.g. ``repeated(_:count:axis:stream:)``

### Slicing

In python a single colon (`:`) or double colon is used as a slicing operator.  In swift the equivalent would be a 
[RangeExpression](https://developer.apple.com/documentation/swift/rangeexpression):

```swift
// python: array[1:5]
array[1 ..< 5]
```

Both python and swift allow open ended range expressions:

```swift
// python: array[:5]
array[..<5]

// python: array[3:]
array[3..<]
```

As mentioned previously, negative indexes are usable with ``MLXArray``, including in slices:

```swift
// 3rd to last and 2nd to last
// python: [-3:-1]
array[-3 ..< -1]

// open ended expressions too -- parenthesis may be required
// python: [-3:]
array[(-3)...]
```

In python the bare `:` gives a full range slice.  The equivalent in swift:

```swift
// python: array[:]
array[0...]
```

There is no python equivalent for the [ClosedRange](https://developer.apple.com/documentation/swift/closedrange):

```swift
// no python equivalent
array[1 ... 4]
```

Python slices have an optional third parameter, the stride.  Swift uses ``MLXArrayIndex/stride(from:to:by:)``:

```swift
// full range, stride by 2
// python: array[::2]
array[.stride(by: 2)]

// start/end, stride by 2
// python: array[1:6:2]
array[.stride(from: 1, to: 6, by: 2)]
```

Negative strides are allowed:

```swift
// reverse the axis
// python: array[::-1]
array[.stride(by: -1)]
```

### None (newaxis) Operator

The [`newaxis` index in numpy](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) is
often written as `None` in idiomatic python code.  The swift equivalent does the same operation:

```swift
// python: array[None]
array[.newAxis]
```

The ``MLXArrayIndex/newAxis`` (`.newAxis`) index inserts a new size one dimension, similar to calling
``MLXArray/expandedDimensions(axis:stream:)`` with the appropriate axis.

### Ellipsis Operator

The ellipsis operator is used in python to consume all available axes with full range slices.
In swift this is written as ``MLXArrayIndex/ellipsis`` (`.ellipsis`):

```swift
// produces an array with the given shape
let array = MLXArray.ones([3, 4, 5, 6])

// the following groups of expressions are all equivalent

// python: array[..., 4]
array[.ellipsis, 4]
array[0..., 0..., 0..., 4]

// python: array[2, ...]
array[2, .ellipsis]
array[2, 0..., 0..., 0...]

// python: array[2, ..., 4]
array[2, .ellipsis, 4]
array[2, 0..., 0..., 4]
```

This can be used, for example, to apply a stride to the last axis:

```swift
// stride the last axis by 2
// python: array[..., ::2]
array[.ellipsis, .stride(by: 2)]

// reverse the last axis
// python: array[..., ::-1]
array[.ellipsis, .stride(by: -1)]
```

or to append a new axis:

```swift
// python array[..., None]
array[.ellipsis, .newAxis]
```

## Topics

### Subscript Functions

- ``MLXArray/subscript(_:stream:)-(MLXArrayIndex...,StreamOrDevice)``
- ``MLXArray/subscript(_:stream:)-([MLXArrayIndex],StreamOrDevice)``
- ``MLXArray/subscript(_:axis:stream:)-(Int,Int,StreamOrDevice)``
- ``MLXArray/subscript(_:axis:stream:)-(RangeExpression<Int>,Int,StreamOrDevice)``
- ``MLXArray/subscript(from:to:stride:axis:stream:)``

### Related Functions

- ``MLXArray/take(_:axis:stream:)``
- ``takeAlong(_:_:axis:stream:)``
- <doc:converting-python>
