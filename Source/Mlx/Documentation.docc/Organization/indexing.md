# Indexing

Array subscripts.

``MLXArray`` provides a number of subscript operators to provide access to the contents
of an array.

Here are several examples that read a region of the array to produce a new array:

```swift
let a = MLXArray(0 ..< 512, [8, 8, 8])

// arrays can be indexed by integer -- this produces an [8, 8] result
let v = a[1]

// this produces an array scalar -- get the value with item(Int.self)
let v = a[1, 2, 3]

// array indexes can be negative -- this means "from the end"
let v = a[-1, -2]

// range expressions can be used.  the first example produces a [2, 8, 8] result
let v = a[1 ..< 3]
let v = a[1 ... 1]

// open ended and multiple ranges are possible
let v = a[1 ..< 2, ..<3, 3...]

// the same negative index rules apply
let v = a[-2 ..< -1, ..<(-3), (-3)...]
```

The same operations can be used to assign with <doc:broadcasting>:

```swift
var a = MLXArray(0 ..< 512, [8, 8, 8])

// sets an [8, 8] area to 7 (broadcasting)
a[1] = 7
```

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

## Topics

### Subscript Functions

- ``MLXArray/subscript(_:stream:)-42bjb``
- ``MLXArray/subscript(_:stream:)-4cvvk``
- ``MLXArray/subscript(_:stream:)-1gwn3``
- ``MLXArray/subscript(_:stream:)-8x0dh``

### Related Functions

- ``MLXArray/take(_:axis:stream:)``
- ``takeAlong(_:_:axis:stream:)``
