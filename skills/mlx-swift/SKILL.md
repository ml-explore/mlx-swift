---
name: swift-mlx
description: MLX Swift - High-performance ML framework for Apple Silicon with lazy evaluation, automatic differentiation, and unified memory
triggers:
  - mlx
  - mlx-swift
  - mlx array
  - apple silicon ml
  - neural network swift
  - automatic differentiation swift
  - metal compute swift
---

# MLX Swift Framework

MLX Swift is Apple's high-performance machine learning framework designed specifically for Apple Silicon. It provides NumPy-like array operations with lazy evaluation, automatic differentiation, and unified CPU/GPU memory.

## When to Use This Skill

- Array operations on Apple Silicon (MLXArray)
- Building neural networks (MLXNN)
- Training models with automatic differentiation
- Custom Metal kernels via MLXFast
- Performance optimization with JIT compilation

## Architecture Overview

```
MLXOptimizers (Adam, AdamW, SGD, etc.)
       ↓
MLXNN (Layers, Modules, Losses)
       ↓
MLX (Arrays, Ops, Transforms, FFT, Linalg, Random)
       ↓
Cmlx (C/C++ bindings, Metal GPU)
```

## Key File Reference

| Purpose | File Path |
|---------|-----------|
| Core array | Source/MLX/MLXArray.swift |
| Operations | Source/MLX/Ops.swift |
| Transforms | Source/MLX/Transforms.swift |
| Factory methods | Source/MLX/Factory.swift |
| Neural layers | Source/MLXNN/*.swift |
| Optimizers | Source/MLXOptimizers/Optimizers.swift |
| Fast ops | Source/MLX/MLXFast.swift |
| Custom kernels | Source/MLX/MLXFastKernel.swift |
| Wired memory coordinator | Source/MLX/WiredMemory.swift |
| GPU working-set helper | Source/MLX/GPU+Metal.swift |

## Quick Start

### Basic Array Creation

```swift
import MLX

// Create arrays
let a = MLXArray([1, 2, 3, 4])
let b = MLXArray(0 ..< 12, [3, 4])  // Shape [3, 4]
let c = MLXArray.zeros([2, 3])
let d = MLXArray.ones([4, 4], dtype: .float32)

// Random arrays (use MLXRandom namespace or free functions)
let uniform = MLXRandom.uniform(0.0 ..< 1.0, [3, 3])
let normal = MLXRandom.normal([100])
```

### Array Properties

```swift
let array = MLXArray(0 ..< 12, [3, 4])
array.shape    // [3, 4]
array.ndim     // 2
array.size     // 12
array.dtype    // .int64
array.count    // 3 (first dimension)
```

### Basic Operations

```swift
let a = MLXArray([1.0, 2.0, 3.0])
let b = MLXArray([4.0, 5.0, 6.0])

// Arithmetic (lazy - not computed until eval)
let sum = a + b
let product = a * b
let matmul = a.matmul(b.T)

// Force evaluation
eval(sum, product)
// or
sum.eval()
```

### Building a Neural Network

```swift
import MLX
import MLXNN

class MLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self.fc1 = Linear(inputDim, hiddenDim)
        self.fc2 = Linear(hiddenDim, outputDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = relu(x)
        return fc2(x)
    }
}

let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
eval(model)  // Initialize parameters
```

### Training Loop

```swift
import MLXOptimizers

let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
let optimizer = Adam(learningRate: 0.001)

func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    let logits = model(x)
    return crossEntropy(logits: logits, targets: y, reduction: .mean)
}

// Compute loss and gradients - valueAndGrad returns a function
let lossAndGrad = valueAndGrad(model: model, loss)
let (lossValue, grads) = lossAndGrad(model, x, y)

// Update model
optimizer.update(model: model, gradients: grads)
eval(model, optimizer)
```

## Primary Workflow: Array Operations

See [arrays.md](references/arrays.md) for detailed array creation and indexing.

### Creation Functions

```swift
// Zeros and ones
MLXArray.zeros([3, 4])
MLXArray.ones([2, 2], dtype: .float16)

// Ranges
arange(0, 10, 2)           // [0, 2, 4, 6, 8]
linspace(0.0, 1.0, 5)      // [0.0, 0.25, 0.5, 0.75, 1.0]

// Identity and diagonal
MLXArray.identity(3)
diagonal(array, offset: 0)

// Full
MLXArray.full([2, 3], values: 7.0)
```

### Indexing

```swift
let a = MLXArray(0 ..< 12, [3, 4])

// Single element
a[0, 1]

// Slicing
a[0...]           // All rows
a[..<2]           // First 2 rows
a[1..., 2...]     // From row 1, column 2 onwards

// Advanced indexing
a[.ellipsis, 0]       // First column of all dimensions
a[.newAxis, .ellipsis]  // Add dimension at front
```

### Shape Manipulation

```swift
let a = MLXArray(0 ..< 12, [3, 4])

a.reshaped([4, 3])
a.reshaped(-1, 6)     // Infer first dimension
a.T                    // Transpose
a.transposed(1, 0)     // Explicit transpose
a.squeezed()           // Remove size-1 dimensions
a.expandedDimensions(axis: 0)
```

## Secondary Workflow: Neural Networks

See [neural-networks.md](references/neural-networks.md) for complete layer reference.

### Built-in Layers

```swift
// Linear layers
Linear(inputDim, outputDim, bias: true)
Bilinear(in1, in2, out)

// Convolutions
Conv1d(inputChannels, outputChannels, kernelSize: 3)
Conv2d(inputChannels, outputChannels, kernelSize: 3, stride: 1, padding: 1)

// Normalization
LayerNorm(dimensions)
RMSNorm(dimensions)
BatchNorm(featureCount)
GroupNorm(groupCount, dimensions)

// Attention
MultiHeadAttention(dimensions: 512, numHeads: 8)

// Recurrent
RNN(inputSize, hiddenSize)
LSTM(inputSize, hiddenSize)
GRU(inputSize, hiddenSize)

// Regularization
Dropout(p: 0.1)
```

### Module Property Wrappers

```swift
class MyLayer: Module {
    @ModuleInfo var layer: Linear           // Tracked module
    @ModuleInfo(key: "w") var weights: Linear  // Custom key

    let constant: MLXArray  // NOT tracked (no wrapper)
}
```

### Loss Functions

```swift
crossEntropy(logits: logits, targets: targets, reduction: .mean)
binaryCrossEntropy(logits: logits, targets: targets)
l1Loss(predictions: predictions, targets: targets, reduction: .mean)
mseLoss(predictions: predictions, targets: targets, reduction: .mean)
smoothL1Loss(predictions: predictions, targets: targets, beta: 1.0)
klDivLoss(inputs: inputs, targets: targets, reduction: .mean)
```

## Tertiary Workflow: Training

See [transforms.md](references/transforms.md) for automatic differentiation details.

### Gradient Computation

```swift
// Simple gradient
let gradFn = grad { x in
    sum(x * x)
}
let g = gradFn(MLXArray([1.0, 2.0, 3.0]))

// Value and gradient together
let (value, gradient) = valueAndGrad { x in
    sum(x * x)
}(MLXArray([1.0, 2.0, 3.0]))

// Model gradients - valueAndGrad returns a function, call it to get results
let lossAndGradFn = valueAndGrad(model: model) { model in
    model(input)
}
let (loss, grads) = lossAndGradFn(model)
```

### Optimizers

See [optimizers.md](references/optimizers.md) for all optimizers.

```swift
// Common optimizers
let sgd = SGD(learningRate: 0.01, momentum: 0.9)
let adam = Adam(learningRate: 0.001, betas: (0.9, 0.999))
let adamw = AdamW(learningRate: 0.001, weightDecay: 0.01)

// Training step
optimizer.update(model: model, gradients: grads)
eval(model, optimizer)
```

### Compilation for Performance

```swift
// Compile a pure array function for faster execution
let compiledOp = compile { (a: MLXArray, b: MLXArray) -> MLXArray in
    let x = a + b
    return sum(x * x)
}

// Use compiled version
let output = compiledOp(arrayA, arrayB)

// Note: compile() works best with pure MLXArray functions.
// For models, call model methods directly (they can use internal compilation).
```

## Quaternary Workflow: Wired Memory Coordination

See [wired-memory.md](references/wired-memory.md) for full policy, hysteresis, and admission guidance.

```swift
import MLX

let policy = WiredSumPolicy()

// Reservation: participates in admission but does not keep the wired limit high while idle.
let weightsTicket = policy.ticket(size: weightsBytes, kind: .reservation)
_ = await weightsTicket.start()

// Active work: raises limit while inference runs.
let inferenceTicket = policy.ticket(size: kvCacheBytes, kind: .active)
try await inferenceTicket.withWiredLimit {
    // run model inference
}

_ = await weightsTicket.end()
```

## Best Practices

### DO

- **Use lazy evaluation**: MLX arrays are computed lazily. Call `eval()` strategically to control memory and compute.
- **Batch eval calls**: `eval(a, b, c)` is more efficient than separate calls.
- **Use `@ModuleInfo`** for all module properties to enable quantization and updates.
- **Use actors for concurrent code**: Encapsulate MLX state within actors for thread safety.
- **Use namespaced functions**: `MLXRandom.uniform()`, `FFT.fft()`, `Linalg.inv()`.
- **Use ticket-based wired memory coordination**: Prefer `WiredMemoryTicket.withWiredLimit` and `WiredMemoryManager.shared`.

### DON'T

- **Don't share MLXArrays across tasks**: MLXArray is NOT Sendable by design.
- **Don't use deprecated module imports**: Use `import MLX` not `import MLXRandom`.
- **Don't forget to eval()**: Unevaluated arrays can accumulate large compute graphs.
- **Don't mutate arrays directly**: Use operations that return new arrays.
- **Don't call deprecated wired-limit APIs**: Avoid `GPU.withWiredLimit(...)` and `Memory.withWiredLimit(...)`.

## Deprecated Patterns

| If you see... | Use instead... |
|---------------|----------------|
| `import MLXRandom` | `import MLX` then `MLXRandom.uniform()` or free function `uniform()` |
| `import MLXFFT` | `import MLX` then `FFT.fft()` |
| `import MLXLinalg` | `import MLX` then `Linalg.inv()` |
| `GPU.activeMemory` | `Memory.activeMemory` |
| `GPU.withWiredLimit(...)` | `WiredMemoryTicket(...).withWiredLimit { ... }` via `WiredMemoryManager` |
| `Memory.withWiredLimit(...)` | `WiredMemoryTicket(...).withWiredLimit { ... }` |
| `repeat(_:count:)` | `repeated(_:count:)` |
| `addmm()` | `addMM()` |
| `LogSoftMax` | `LogSoftmax` |
| `SoftMax` | `Softmax` |

See [deprecated.md](references/deprecated.md) for the complete migration guide.

## Swift Concurrency Notes

MLX has specific concurrency behavior:

- **MLXArray is NOT Sendable**: This is intentional. Arrays contain references to compute graphs.
- **evalLock protects eval/stream creation**: The global lock serializes evaluation and stream operations.
- **Lazy operations are NOT thread-safe**: Don't share arrays across tasks without proper synchronization.
- **Use actors to encapsulate MLX state**: Create and use MLXArrays within the same actor.
- **Use wired-memory tickets for concurrent inference**: Coordinate active/reservation budgets through the shared manager.

See [concurrency.md](references/concurrency.md) for thread safety details.

## Reference Documentation

- [Arrays](references/arrays.md) - Array creation, properties, indexing
- [Operations](references/operations.md) - Math ops, broadcasting, reductions
- [Transforms](references/transforms.md) - grad, vmap, compile, eval
- [Neural Networks](references/neural-networks.md) - Layers and modules
- [Optimizers](references/optimizers.md) - Training optimizers
- [Custom Layers](references/custom-layers.md) - Building custom modules
- [Custom Kernels](references/custom-kernels.md) - Metal kernels with MLXFast
- [Wired Memory](references/wired-memory.md) - Ticket-based wired limit coordination
- [Concurrency](references/concurrency.md) - Thread safety guide
- [Deprecated](references/deprecated.md) - Migration guide
