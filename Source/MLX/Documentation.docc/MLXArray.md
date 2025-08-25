# ``MLX/MLXArray``

An N dimensional array.  The main type in `mlx`.

## Introduction

`MLXArray` is an N dimension array that can contain a variety of data types (``DType``).
`MLXArray` supports a wide range of functions and operations to manipulate it and
is used as the basis for the `MLXNN` module which implements many layers that
are the basis for building machine learning models.

```swift
// create an array from a swift array
let a1 = MLXArray([1, 2, 3])

// this holds Int32
print(a1.dtype)

// and has a shape of [3]
print(a1.shape)

// there are a variety of operators and functions that can be used
let a2 = sqrt(a1 * 3)
```

## Thread Safety

> `MLXArray` is not thread safe.

Although `MLXArray` looks like a normal multidimensional array, it is actually far more
sophisticated.  It actually holds a promise for future computations, see <doc:lazy-evaluation>
and is thus not thread safe.  For example:

```swift
let a: MLXArray
let b: MLXArray

let c = a + b
```

`c` is not the result of `a + b` but rather a graph representing `a` and `b` (which in turn
may be large unresolved graphs) and the `addition` operator that combines them.  It is not safe
to create `c` in one thread and consume/evaluate it in another.

## Memory Safety

> `MLXArray` is not memory safe.

Unlike swift `Array`, `MLXArray` is not memory safe -- use caution when using
indexing operators.

For example:

```swift
let a = MLXArray([0])
print(a[10000])
```

Will print a random value from outside `a`.

## Topics

### Creation

- <doc:initialization>

### Arithmetic Operators and Functions

- <doc:arithmetic>
- <doc:convolution>
- <doc:cumulative>

### Indexing

- <doc:indexes>
- <doc:indexing>
- <doc:sorting>

### Logical Operations

- <doc:logical>

### Shapes

Some methods allow you to manipulate the shape of the array.  These methods change the size
and ``shape`` of the dimensions without changing the number of elements or contents of the array.

- <doc:shapes>
- <doc:reduction>

### Conversion and Data Types

- <doc:conversion>
