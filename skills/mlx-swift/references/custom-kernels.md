# Custom Metal Kernels Reference

MLXFast allows you to write custom Metal compute kernels for operations not covered by built-in functions.

## Basic Metal Kernel

```swift
import MLX

// Create a kernel
let kernel = MLXFast.metalKernel(
    name: "my_kernel",
    inputNames: ["a", "b"],
    outputNames: ["out"],
    source: """
        uint idx = thread_position_in_grid.x;
        out[idx] = a[idx] + b[idx];
    """
)

// Execute the kernel
let a = MLXArray([1.0, 2.0, 3.0, 4.0])
let b = MLXArray([5.0, 6.0, 7.0, 8.0])

let result = kernel(
    [a, b],
    grid: (4, 1, 1),
    threadGroup: (4, 1, 1),
    outputShapes: [[4]],
    outputDTypes: [.float32]
)

print(result[0])  // [6.0, 8.0, 10.0, 12.0]
```

## Kernel Parameters

### metalKernel Function

```swift
MLXFast.metalKernel(
    name: String,                        // Kernel name
    inputNames: some Sequence<String>,   // Input parameter names
    outputNames: some Sequence<String>,  // Output parameter names
    source: String,                      // Metal shader code (function body)
    header: String = "",                 // Optional header code
    ensureRowContiguous: Bool = true,    // Ensure inputs are contiguous
    atomicOutputs: Bool = false          // Use atomic outputs
)
```

### Kernel Execution

```swift
kernel(
    _ inputs: [any ScalarOrArray],                  // Input arrays
    template: [(String, any KernelTemplateArg)]?,   // Template arguments
    grid: (Int, Int, Int),                          // Grid dimensions
    threadGroup: (Int, Int, Int),                   // Threadgroup size
    outputShapes: some Sequence<[Int]>,             // Output shapes
    outputDTypes: some Sequence<DType>,             // Output types
    initValue: Float? = nil,                        // Optional init value
    verbose: Bool = false,                          // Print generated code
    stream: StreamOrDevice = .default
) -> [MLXArray]
```

## Metal Shader Syntax

### Available Variables

Inside your kernel source, these variables are available:

```metal
// Thread position
uint3 thread_position_in_grid;       // Global thread position
uint3 thread_position_in_threadgroup; // Position within threadgroup
uint3 threadgroup_position_in_grid;   // Threadgroup position
uint3 threads_per_threadgroup;        // Threadgroup dimensions
uint3 threadgroups_per_grid;          // Grid dimensions
```

### Input/Output Access

Inputs and outputs are device pointers:

```metal
// For input named "a" and output named "out":
device const float* a;    // Read-only input
device float* out;        // Writable output
```

## Template Arguments

Use templates for compile-time constants:

```swift
let kernel = MLXFast.metalKernel(
    name: "templated",
    inputNames: ["x"],
    outputNames: ["y"],
    source: """
        uint idx = thread_position_in_grid.x;
        if (USE_RELU) {
            y[idx] = max(x[idx], T(0));
        } else {
            y[idx] = x[idx];
        }
    """
)

let result = kernel(
    [input],
    template: [
        ("USE_RELU", true),
        ("T", DType.float32)
    ],
    grid: (size, 1, 1),
    threadGroup: (256, 1, 1),
    outputShapes: [input.shape],
    outputDTypes: [.float32]
)
```

### Valid Template Types

- `Bool` - Boolean constants
- `Int` - Integer constants
- `DType` - Data type templates

## Header Code

Use the `header` parameter for helper functions:

```swift
let kernel = MLXFast.metalKernel(
    name: "with_helpers",
    inputNames: ["x"],
    outputNames: ["y"],
    header: """
        inline float my_activation(float x) {
            return x > 0 ? x : 0.1 * x;  // LeakyReLU
        }
    """,
    source: """
        uint idx = thread_position_in_grid.x;
        y[idx] = my_activation(x[idx]);
    """
)
```

## Atomic Operations

For reductions or scatter operations:

```swift
let kernel = MLXFast.metalKernel(
    name: "atomic_add",
    inputNames: ["values", "indices"],
    outputNames: ["out"],
    source: """
        uint idx = thread_position_in_grid.x;
        int target = indices[idx];
        atomic_fetch_add_explicit(&out[target], values[idx], memory_order_relaxed);
    """,
    atomicOutputs: true  // Enable atomic outputs
)

let result = kernel(
    [values, indices],
    grid: (values.size, 1, 1),
    threadGroup: (256, 1, 1),
    outputShapes: [[outputSize]],
    outputDTypes: [.float32],
    initValue: 0.0  // Initialize output to zero
)
```

## 2D Grid Example

```swift
let kernel = MLXFast.metalKernel(
    name: "matrix_scale",
    inputNames: ["matrix"],
    outputNames: ["scaled"],
    source: """
        uint2 pos = uint2(thread_position_in_grid.xy);
        uint rows = 4;
        uint cols = 4;
        uint idx = pos.y * cols + pos.x;
        scaled[idx] = matrix[idx] * 2.0;
    """
)

let matrix = MLXArray(0 ..< 16, [4, 4]).asType(.float32)

let result = kernel(
    [matrix],
    grid: (4, 4, 1),
    threadGroup: (4, 4, 1),
    outputShapes: [[4, 4]],
    outputDTypes: [.float32]
)
```

## Custom Function with VJP

For differentiable custom operations:

```swift
// Define forward and backward kernels
let forwardKernel = MLXFast.metalKernel(
    name: "forward",
    inputNames: ["x"],
    outputNames: ["y"],
    source: """
        uint idx = thread_position_in_grid.x;
        y[idx] = x[idx] * x[idx];  // Square
    """
)

let backwardKernel = MLXFast.metalKernel(
    name: "backward",
    inputNames: ["x", "grad_y"],
    outputNames: ["grad_x"],
    source: """
        uint idx = thread_position_in_grid.x;
        grad_x[idx] = 2.0 * x[idx] * grad_y[idx];  // d/dx(x^2) = 2x
    """
)

// Create differentiable custom function using result builder syntax
let customSquare = CustomFunction {
    Forward { inputs in
        // Forward pass
        let x = inputs[0]
        return forwardKernel(
            [x],
            grid: (x.size, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )
    }
    VJP { primals, cotangents in
        // Backward pass
        let x = primals[0]
        let gradY = cotangents[0]
        return backwardKernel(
            [x, gradY],
            grid: (x.size, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )
    }
}

// Use in differentiable computation
let x = MLXArray([1.0, 2.0, 3.0])
let (value, gradient) = valueAndGrad { x in
    sum(customSquare([x])[0])
}(x)
```

## Performance Tips

### Grid and Threadgroup Sizing

```swift
// Typical 1D sizing
let size = array.size
let threadGroupSize = min(256, size)
let gridSize = size

kernel(
    [array],
    grid: (gridSize, 1, 1),
    threadGroup: (threadGroupSize, 1, 1),
    ...
)

// For 2D operations
let (rows, cols) = array.shape2
kernel(
    [array],
    grid: (cols, rows, 1),
    threadGroup: (16, 16, 1),  // Common 2D threadgroup size
    ...
)
```

### Memory Coalescing

```swift
// Good: Sequential memory access
source: """
    uint idx = thread_position_in_grid.x;
    out[idx] = in[idx] * 2.0;
"""

// Bad: Strided access
source: """
    uint idx = thread_position_in_grid.x;
    out[idx] = in[idx * stride];  // Cache unfriendly
"""
```

### Avoid Divergent Branches

```swift
// Prefer: Predicated execution
source: """
    uint idx = thread_position_in_grid.x;
    float val = in[idx];
    out[idx] = val > 0 ? val : 0;  // Compiles to select
"""

// Avoid: Complex branching
source: """
    uint idx = thread_position_in_grid.x;
    if (complex_condition) {
        // Long code path A
    } else {
        // Long code path B
    }
"""
```

## Debugging

Enable verbose mode to see generated code:

```swift
let result = kernel(
    [input],
    grid: (size, 1, 1),
    threadGroup: (256, 1, 1),
    outputShapes: [[size]],
    outputDTypes: [.float32],
    verbose: true  // Prints full Metal source
)
```
