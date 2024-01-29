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

In python you can index using slices with strides, e.g. to access only the even
numbered elements:

```python
# access strided contents of array
evens = array[..., ::2]
```

You can accomplish something similar with the swift API:

```
let array = MLXArray(0 ..< (2 * 3 * 4), [2, 3, 4])

// array([[[0, 2],
//         [4, 6],
//         [8, 10]],
//        [[12, 14],
//         [16, 18],
//         [20, 22]]], dtype=int64)
let evens = a[stride: 2, axis: -1]
```

## Topics

### Subscript Functions

- ``MLXArray/subscript(_:stream:)-od5g``
- ``MLXArray/subscript(_:stream:)-7n5nw``
- ``MLXArray/subscript(_:stream:)-82jwt``
- ``MLXArray/subscript(_:stream:)-1a84u``
- ``MLXArray/subscript(_:stream:)-8a2s7``
- ``MLXArray/subscript(_:stream:)-4z56f``
- ``MLXArray/subscript(_:axis:stream:)-1jy5n``
- ``MLXArray/subscript(_:axis:stream:)-79psf``
- ``MLXArray/subscript(from:to:stride:axis:stream:)``

### Related Functions

- ``MLXArray/take(_:axis:stream:)``
- ``takeAlong(_:_:axis:stream:)``
