# Array Operations Reference

MLX provides NumPy-like operations with automatic broadcasting and lazy evaluation.

## Arithmetic Operations

### Basic Math

```swift
let a = MLXArray([1.0, 2.0, 3.0])
let b = MLXArray([4.0, 5.0, 6.0])

// Operators
a + b           // Element-wise addition
a - b           // Subtraction
a * b           // Multiplication
a / b           // Division
a ** 2          // Power

// Function equivalents
add(a, b)
subtract(a, b)
multiply(a, b)
divide(a, b)
pow(a, 2)

// With scalars
a + 5
a * 2.0
3 - a
```

### More Arithmetic

```swift
abs(a)              // Absolute value
negative(a)         // Negate: -a
reciprocal(a)       // 1/a
square(a)           // a^2
sqrt(a)             // Square root
rsqrt(a)            // 1/sqrt(a)

floor(a)            // Round down
ceil(a)             // Round up
round(a)            // Round to nearest

clip(a, min: 0, max: 1)  // Clamp values
```

### Trigonometric

```swift
sin(a)
cos(a)
tan(a)
asin(a)
acos(a)
atan(a)
atan2(y, x)

sinh(a)
cosh(a)
tanh(a)
asinh(a)
acosh(a)
atanh(a)
```

### Exponential and Logarithmic

```swift
exp(a)              // e^a
expm1(a)            // e^a - 1 (more accurate for small a)
log(a)              // Natural log
log2(a)             // Log base 2
log10(a)            // Log base 10
log1p(a)            // log(1 + a) (more accurate for small a)
logAddExp(a, b)     // log(exp(a) + exp(b))
```

## Comparison Operations

```swift
let a = MLXArray([1, 2, 3])
let b = MLXArray([2, 2, 2])

// Operators (return boolean arrays)
a .== b             // Equal
a .!= b             // Not equal
a .< b              // Less than
a .<= b             // Less or equal
a .> b              // Greater than
a .>= b             // Greater or equal

// Functions
equal(a, b)
notEqual(a, b)
less(a, b)
lessEqual(a, b)
greater(a, b)
greaterEqual(a, b)

// Approximate equality
allClose(a, b, rtol: 1e-5, atol: 1e-8)
isClose(a, b, rtol: 1e-5, atol: 1e-8)
```

### Element-wise Logic

```swift
let mask1 = a .> 1
let mask2 = a .< 3

// Logical operators
mask1 .&& mask2     // AND
mask1 .|| mask2     // OR
!mask1              // NOT (logicalNot)

// Functions
logicalAnd(mask1, mask2)
logicalOr(mask1, mask2)
logicalNot(mask1)
```

### Special Comparisons

```swift
maximum(a, b)       // Element-wise max
minimum(a, b)       // Element-wise min

isNaN(a)            // Check for NaN
isInf(a)            // Check for infinity
isNegInf(a)         // Check for -inf
isPosInf(a)         // Check for +inf

`where`(condition, a, b)  // Ternary selection (backticks required)
which(condition, a, b)   // Alias without backticks
```

## Reduction Operations

### Sum and Product

```swift
let a = MLXArray(0 ..< 12, [3, 4])

// Full reduction
sum(a)              // Sum all elements
prod(a)             // Product of all elements

// Along axis
sum(a, axis: 0)     // Sum columns: [4] array
sum(a, axis: 1)     // Sum rows: [3] array
sum(a, axes: [0, 1])

// Keep dimensions
sum(a, axis: 0, keepDims: true)  // Shape [1, 4]

// Cumulative
cumsum(a, axis: 0)
cumprod(a, axis: 1)
```

### Mean and Variance

```swift
mean(a)                     // Mean of all elements
mean(a, axis: 0)            // Mean along axis

variance(a)                 // Variance
variance(a, axis: 0)
variance(a, ddof: 1)        // Sample variance

std(a)                      // Standard deviation
std(a, axis: 0)
```

### Min and Max

```swift
// Values
min(a)
max(a)
min(a, axis: 0)
max(a, axis: 1)

// Indices
argMin(a)                   // Index of minimum
argMax(a)                   // Index of maximum
argMin(a, axis: 0)
argMax(a, axis: 1)
```

### Logical Reductions

```swift
all(boolArray)              // All true?
any(boolArray)              // Any true?
all(boolArray, axis: 0)
any(boolArray, axis: 1)
```

## Matrix Operations

### Matrix Multiply

```swift
let a = MLXArray.ones([3, 4])
let b = MLXArray.ones([4, 5])

// Matrix multiplication
matmul(a, b)                // [3, 5] result
a.matmul(b)

// Batched matmul
let batchA = MLXArray.ones([2, 3, 4])
let batchB = MLXArray.ones([2, 4, 5])
matmul(batchA, batchB)      // [2, 3, 5]

// Inner product
inner(a, b)

// Outer product
outer(a, b)
```

### Special Matrix Ops

```swift
// Add matrix multiply: alpha * (a @ b) + beta * c
addMM(c, a, b, alpha: 1.0, beta: 1.0)

// Block masked matrix multiply
blockMaskedMM(a, b, blockSize: 64, maskOut: nil)

// Gather matrix multiply
gatherMM(a, b, lhsIndices: idx1, rhsIndices: idx2)
```

### Linear Algebra

Use the `Linalg` namespace (NOT deprecated `MLXLinalg` module):

```swift
import MLX

// Matrix inverse
Linalg.inv(a)

// Matrix norm
Linalg.norm(a)
Linalg.norm(a, ord: 2)
Linalg.norm(a, axis: 0)

// Decompositions
let (q, r) = Linalg.qr(a)
let (u, s, vt) = Linalg.svd(a)
let L = Linalg.cholesky(a)

// Triangular inverse
Linalg.triInv(a, upper: true)

// LU decomposition
let (P, L, U) = Linalg.lu(a)
let (LU, pivots) = Linalg.luFactor(a)

// Solve linear systems
Linalg.solve(A, b)
Linalg.solveTriangular(A, b, upper: true)

// Cross product
Linalg.cross(a, b)
```

### Diagonals

```swift
diagonal(a)                 // Extract diagonal
diagonal(a, offset: 1)      // Above main diagonal
diagonal(a, offset: -1)     // Below main diagonal

diag(vector)                // Create diagonal matrix
trace(a)                    // Sum of diagonal
```

## Broadcasting

MLX follows NumPy broadcasting rules:

```swift
let a = MLXArray(0 ..< 4, [4, 1])   // [4, 1]
let b = MLXArray(0 ..< 3, [1, 3])   // [1, 3]

// Result is [4, 3]
let c = a + b

// Broadcasting rules:
// 1. Dimensions are compared right-to-left
// 2. Dimensions match if equal or one is 1
// 3. Missing dimensions are treated as 1
```

### Explicit Broadcasting

```swift
// Broadcast to specific shape
broadcast(a, to: [4, 4])

// Expand dimensions for broadcasting
a.expandedDimensions(axis: 1)
```

## Concatenation and Stacking

```swift
let a = MLXArray([1, 2, 3])
let b = MLXArray([4, 5, 6])

// Concatenate along existing axis
concatenate([a, b])                    // [1,2,3,4,5,6]
concatenate([a, b], axis: 0)

// Stack creates new axis
stacked([a, b])                        // [[1,2,3], [4,5,6]]
stacked([a, b], axis: 0)               // [2, 3]
stacked([a, b], axis: 1)               // [3, 2]

// Split
split(c, parts: 2)                     // Split into 2 equal parts
split(c, indices: [2, 4])              // Split at indices
```

## FFT Operations

Use the `FFT` namespace (NOT deprecated `MLXFFT` module):

```swift
import MLX

// 1D FFT
FFT.fft(a)
FFT.ifft(a)

// 2D FFT
FFT.fft2(a)
FFT.ifft2(a)

// N-dimensional FFT
FFT.fftn(a)
FFT.ifftn(a)

// Real FFT (for real input)
FFT.rfft(a)
FFT.irfft(a)
FFT.rfft2(a)
FFT.irfft2(a)
FFT.rfftn(a)
FFT.irfftn(a)
```

## Sorting and Searching

```swift
// Sort
sort(a)
sort(a, axis: 0)

// Argsort (indices that would sort)
argSort(a)
argSort(a, axis: 0)

// Partition (partial sort)
partitioned(a, kth: 3)
argPartition(a, kth: 3)

// Find indices of non-zero elements
nonZero(boolArray)
```

## Tile and Repeat

```swift
// Tile (repeat entire array)
tile(a, reps: [2, 3])

// Repeat (repeat elements) - NOTE: use 'repeated' not 'repeat'
repeated(a, count: 3, axis: 0)
```
