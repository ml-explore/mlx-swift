#  Numpy Style Broadcasting

How different size arrays can be used together.

MLX uses Numpy style broadcasting:

- [Numpy Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)

Here is a brief excerpt on how numpy describes this:

> Numpy: The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations. There are, however, cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.

Let's consider some examples using an array like this:

```swift
let array = MLXArray(0 ..< 12, [4, 3])
```

giving us this structure:

```swift
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8],
       [9, 10, 11]], dtype=int32)
```

#### Scalars

To add `1` to each value in the array we can simply write:

```swift
let r = array + 1
```

giving us:

```swift
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
       [10, 11, 12]], dtype=int32)
```

This uses ``MLXArray/+(_:_:)-(MLXArray,ScalarOrArray)`` which uses ``ScalarOrArray`` to automatically convert scalar values
into ``MLXArray``.

A scalar can be broadcast to any shape array:  the scalar is repeated for each element in the array.
Conceptually the scalar is converted into an array of the same shape and then added:

```swift
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]], dtype=int32)
```

Broadcasting allows a much more efficient implementation where the scalar may simply be reused
for ever element in the first array.

### Arrays

Array broadcasting is similar to scalar broadcasting but it requires compatible shapes.  
Going from right to left two shapes are compatible for broadcasting if any of these conditions are true:

- the dimensions are equal
- one of the dimensions is 1 or is missing (fewer dimensions)

If the arrays have different number of dimensions, the result will have the same number of dimensions
as the array with the most dimensions.  The resulting shape will have the max of the two matching dimensions.

For example:

```swift
let a = MLXArray(0 ..< 12, [4, 3])
let b = MLXArray(0 ..< 3, [3])

// compatible because the last dimensions match:
// [4, 3]
// [   3]
let r = a + b

// equivalent to adding these two arrays:
//
// array([[0, 1, 2],
//        [3, 4, 5],
//        [6, 7, 8],
//        [9, 10, 11]], dtype=int32)
//
// array([[0, 1, 2],
//        [0, 1, 2],
//        [0, 1, 2],
//        [0, 1, 2]], dtype=int32)
```
