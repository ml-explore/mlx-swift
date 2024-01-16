#  Numpy Style Broadcasting

How different size arrays can be used together.

MLX uses Numpy style broadcasting:

- [Numpy Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)

Here is a brief excerpt on how numpy describes this:

> Numpy: The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations. There are, however, cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.

Let's consider some examples using an array like this:

```
let array = MLXArray(0 ..< 12, [4, 3])
```

giving us this structure:

```
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8],
       [9, 10, 11]], dtype=int64)
```

#### Scalars

To add `1` to each value in the array we can simply write:

```
let r = array + 1
```

giving us:

```
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
       [10, 11, 12]], dtype=int64)
```

This uses the ``MLXArray/init(_:)-7t81v`` which allows automatic conversion from an `Int32` literal to
a scalar `MLXArray`, e.g. `MLXArray(Int32(1))`.

A scalar can be broadcast to any shape array:  the scalar is repeated for each element in the array.
Conceptually the scalar is converted into an array of the same shape and then added:

```
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]], dtype=int64)
```

Broadcasting allows a much more efficient implementation where the scalar may simply be reused
for ever element in the first array.

