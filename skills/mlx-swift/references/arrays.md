# MLXArray Reference

MLXArray is the fundamental data type in MLX Swift, representing multi-dimensional arrays on Apple Silicon.

## Creating Arrays

### From Swift Values

```swift
// Scalar
let scalar = MLXArray(42)
let floatScalar = MLXArray(3.14)

// From array literal
let a = MLXArray([1, 2, 3, 4])
let b = MLXArray([1.0, 2.0, 3.0])

// From range
let c = MLXArray(0 ..< 10)

// With explicit shape
let d = MLXArray(0 ..< 12, [3, 4])  // 3 rows, 4 columns
let e = MLXArray([1, 2, 3, 4, 5, 6], [2, 3])

// Complex numbers
let complex = MLXArray(real: 1.0, imaginary: 2.0)
```

### Factory Methods

```swift
// Zeros and ones
MLXArray.zeros([3, 4])
MLXArray.zeros([2, 2], dtype: .float16)
MLXArray.ones([4, 4])
MLXArray.ones(like: existingArray)

// Full (constant value)
MLXArray.full([2, 3], values: 7.0)
MLXArray.full([2, 3], values: MLXArray(7.0), dtype: .float32)

// Identity matrix
MLXArray.identity(3)
MLXArray.identity(4, dtype: .float32)

// Ranges
arange(10)                    // [0, 1, 2, ..., 9]
arange(0, 10, 2)              // [0, 2, 4, 6, 8]
arange(0.0, 1.0, 0.1)         // Floating point range

linspace(0.0, 1.0, 5)         // [0.0, 0.25, 0.5, 0.75, 1.0]
linspace(0.0, 10.0, 11)       // Evenly spaced
```

### Random Arrays

Use `MLXRandom` namespace or free functions:

```swift
import MLX

// Uniform distribution
MLXRandom.uniform(0.0 ..< 1.0, [3, 3])
MLXRandom.uniform(low: -1, high: 1, [100])

// Normal distribution
MLXRandom.normal([100])
MLXRandom.normal([3, 3], dtype: .float16)

// Integer random
MLXRandom.randInt(0 ..< 100, [10])

// Bernoulli
MLXRandom.bernoulli(p: 0.5, [10])

// Seeding
MLXRandom.seed(42)

// Key-based RNG
let key = MLXRandom.key(42)
let (newKey, values) = MLXRandom.split(key)
```

## Data Types (DType)

```swift
// Available types
DType.bool
DType.uint8, .uint16, .uint32, .uint64
DType.int8, .int16, .int32, .int64
DType.float16, .float32, .float64   // Note: float64 limited on GPU
DType.bfloat16
DType.complex64

// Type conversion
let floatArray = intArray.asType(.float32)
let f16 = array.asType(Float16.self)

// Check type
array.dtype.isFloatingPoint
array.dtype.isInteger
array.dtype.isComplex
```

## Array Properties

```swift
let array = MLXArray(0 ..< 12, [3, 4])

// Dimensions
array.ndim      // 2 (number of dimensions)
array.shape     // [3, 4]
array.size      // 12 (total elements)
array.count     // 3 (first dimension size)
array.nbytes    // Size in bytes
array.itemSize  // Bytes per element

// Individual dimensions
array.dim(0)    // 3
array.dim(1)    // 4
array.dim(-1)   // 4 (last dimension)

// Tuple accessors for fixed dimensions
let (rows, cols) = array.shape2
let (d0, d1, d2) = array3d.shape3
let (b, h, w, c) = array4d.shape4

// Data type
array.dtype     // .int64
```

## Reading Values

```swift
// Single scalar value
let value: Float = scalarArray.item()
let intValue = array[0, 0].item(Int.self)

// To Swift array (forces evaluation)
let swiftArray: [Float] = array.asArray(Float.self)

// Raw data access
let data = array.asData(access: .copy)  // Safe copy
let ptr = data.withUnsafeBytes { $0.baseAddress }
```

## Indexing and Slicing

### Basic Indexing

```swift
let a = MLXArray(0 ..< 12, [3, 4])

// Single element (returns MLXArray, not scalar)
a[0, 0]         // First element
a[2, 3]         // Last element
a[-1, -1]       // Negative indexing from end

// Extract scalar
let value: Int = a[0, 0].item()
```

### Range Slicing

```swift
let a = MLXArray(0 ..< 12, [3, 4])

// Row slicing
a[0]            // First row [0, 1, 2, 3]
a[1...]         // Rows 1 onwards
a[..<2]         // First 2 rows
a[1..<3]        // Rows 1 and 2

// Column slicing
a[0..., 0]      // First column
a[0..., 1...]   // Columns 1 onwards

// Combined
a[1..., 2...]   // Subarray from row 1, col 2
a[0..<2, 1..<3] // 2x2 subarray
```

### Advanced Indexing

```swift
// Strided access
a[.stride(by: 2)]     // Every other element along first axis

// Ellipsis (all remaining dimensions)
a[.ellipsis, 0]       // First element of last dimension
a[0, .ellipsis]       // First element of first dimension

// New axis (add dimension)
a[.newAxis, .ellipsis]  // Add dimension at front: [1, 3, 4]
a[.ellipsis, .newAxis]  // Add dimension at end: [3, 4, 1]

// Boolean masking
let mask = a .> 5
let filtered = a[mask]
```

### Array Indexing

```swift
// Index with another array
let indices = MLXArray([0, 2])
a[indices]            // Rows 0 and 2

// Fancy indexing
let rowIdx = MLXArray([0, 1, 2])
let colIdx = MLXArray([1, 2, 3])
a[rowIdx, colIdx]     // Elements at (0,1), (1,2), (2,3)
```

## Modifying Arrays

### Assignment

```swift
var a = MLXArray(0 ..< 12, [3, 4])

// Assign to slice
a[0] = MLXArray([10, 11, 12, 13])
a[0..., 0] = MLXArray([100, 200, 300])

// Assign scalar to slice
a[1, 1] = MLXArray(999)
```

### The `at` Property for Updates

For repeated updates at the same indices, use `.at`:

```swift
let idx = MLXArray([0, 1, 0, 1])

// Standard assignment: each index updated once
var a1 = MLXArray([0, 0])
a1[idx] += 1
// Result: [1, 1]

// Using .at: accumulates updates
var a2 = MLXArray([0, 0])
a2 = a2.at[idx].add(1)
// Result: [2, 2]
```

Available `at` operations:
- `at[idx].add(value)`
- `at[idx].subtract(value)`
- `at[idx].multiply(value)`
- `at[idx].divide(value)`
- `at[idx].maximum(value)`
- `at[idx].minimum(value)`

## Shape Manipulation

```swift
let a = MLXArray(0 ..< 12, [3, 4])

// Reshape
a.reshaped([4, 3])
a.reshaped([2, 6])
a.reshaped(-1, 6)     // Infer first dim: [2, 6]
a.reshaped([12])      // Flatten

// Transpose
a.T                    // Transpose 2D
a.transposed()         // Reverse all axes
a.transposed(1, 0)     // Explicit axis order

// Swap axes
a.swappedAxes(0, 1)

// Add/remove dimensions
a.expandedDimensions(axis: 0)   // [1, 3, 4]
a.expandedDimensions(axes: [0, 2])
a.squeezed()                     // Remove all size-1 dims
a.squeezed(axis: 0)              // Remove specific dim

// Flatten
a.flattened()
a.flattened(start: 0, end: 1)
```

## Evaluation

MLXArray uses lazy evaluation. Values are computed only when needed:

```swift
let a = MLXArray([1, 2, 3])
let b = MLXArray([4, 5, 6])
let c = a + b  // Not computed yet!

// Force evaluation
c.eval()

// Or evaluate multiple at once (more efficient)
eval(a, b, c)

// Evaluation happens automatically when reading values
let value: Int = c[0].item()  // Forces eval
print(c)                       // Forces eval
```
