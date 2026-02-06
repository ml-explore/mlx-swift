# Swift Concurrency Reference

MLX Swift has specific concurrency characteristics for thread safety on Apple Silicon.

## Key Concurrency Facts

### MLXArray is NOT Sendable

`MLXArray` is intentionally **not** `Sendable`. This is by design:

```swift
// WRONG: Will not compile in strict concurrency mode
Task {
    let array = MLXArray([1, 2, 3])
    await Task.detached {
        print(array)  // Error: MLXArray is not Sendable
    }.value
}
```

**Why?** MLXArray contains references to compute graphs that may be shared and mutated. Sending arrays across task boundaries could cause data races.

### Safe Patterns

```swift
// Pattern 1: Create arrays within the task
Task {
    let array = MLXArray([1, 2, 3])
    let result = array * 2
    eval(result)
    return result.item(Int.self)  // Return Sendable value
}

// Pattern 2: Copy data before sending
let array = MLXArray([1, 2, 3])
eval(array)
let data = array.asArray(Int.self)  // [Int] is Sendable

Task {
    let newArray = MLXArray(data)
    // Work with newArray
}
```

## evalLock

Evaluation and stream creation are serialized through a global `evalLock`:

```swift
// Defined in Source/MLX/Transforms+Eval.swift
let evalLock = NSRecursiveLock()

// Used in eval():
func eval(_ arrays: MLXArray...) {
    _ = evalLock.withLock {
        mlx_eval(...)
    }
}
```

This means:
- `eval()` calls and stream creation are serialized across threads
- **Important**: evalLock only protects eval/stream operations, NOT all GPU ops
- Lazy array operations are NOT automatically thread-safe
- Parallelism happens within Metal, not between MLX calls

## @unchecked Sendable Types

These types are marked `@unchecked Sendable` and are safe to share:

| Type | Location | Notes |
|------|----------|-------|
| `Stream` | Stream.swift | Immutable after creation |
| `Device` | Device.swift | Immutable after creation |
| `MLXFastKernel` | MLXFastKernel.swift | Kernel definition is immutable |
| `ErrorBox` | ErrorHandler.swift | Thread-safe container |
| `ErrorHandler` | ErrorHandler.swift | Internal, thread-safe |
| `_CustomFunctionState` | MLXCustomFunction.swift | Internal, locked |

## Device and Stream Management

Use the public APIs for device and stream management:

```swift
// Get the default device
let defaultDevice = Device.defaultDevice()

// Create a custom stream
let customStream = Stream(.gpu)

// Use specific device/stream in operations
let result = someOperation(stream: customStream)

// For scoped device changes, use withDefaultDevice
Device.withDefaultDevice(.cpu) {
    // All operations in this scope use CPU
    let result = someOperation()
}
```

### Stream Best Practices

```swift
// Create a stream for a specific device
let gpuStream = Stream(.gpu)

// Pass stream explicitly to operations
let result = matmul(a, b, stream: gpuStream)
eval(result)
```

## Async/Await Patterns

### Safe Async Computation

```swift
func computeAsync() async -> Float {
    // Create arrays in async context
    let input = MLXArray.zeros([1000, 1000])
    let result = someHeavyComputation(input)

    // Force eval and extract Sendable result
    eval(result)
    return result.item(Float.self)
}
```

### Background Processing

```swift
func processInBackground(_ data: [Float]) async -> [Float] {
    // Move Sendable data into task
    let array = MLXArray(data)

    // Heavy computation
    let result = transform(array)
    eval(result)

    // Return Sendable result
    return result.asArray(Float.self)
}
```

## Actor Integration

### Creating an MLX Actor

```swift
actor MLXProcessor {
    private var model: MyModel

    init() {
        self.model = MyModel()
        eval(model)
    }

    func predict(input: [Float]) -> [Float] {
        let x = MLXArray(input)
        let output = model(x)
        eval(output)
        return output.asArray(Float.self)
    }

    private var optimizer = Adam(learningRate: 0.001)

    func train(x: [Float], y: [Float]) -> Float {
        let xArray = MLXArray(x)
        let yArray = MLXArray(y)

        // valueAndGrad returns a function - call it to get results
        let lossAndGradFn = valueAndGrad(model: model, lossFunction)
        let (loss, grads) = lossAndGradFn(model, xArray, yArray)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)

        return loss.item(Float.self)
    }

    private func lossFunction(model: MyModel, x: MLXArray, y: MLXArray) -> MLXArray {
        mseLoss(predictions: model(x), targets: y, reduction: .mean)
    }
}
```

### Usage

```swift
let processor = MLXProcessor()

// Safe concurrent access through actor
async let prediction1 = processor.predict(input: data1)
async let prediction2 = processor.predict(input: data2)

let results = await [prediction1, prediction2]
```

## Memory Management in Concurrent Code

### Avoid Accumulating Compute Graphs

```swift
// BAD: Graphs accumulate without eval
for _ in 0..<1000 {
    result = transform(result)
}
// Huge compute graph!

// GOOD: Eval periodically
for i in 0..<1000 {
    result = transform(result)
    if i % 100 == 0 {
        eval(result)  // Materialize and free graph
    }
}
```

### Memory Limits

```swift
// Set memory limits for controlled usage
Memory.memoryLimit = 8 * 1024 * 1024 * 1024  // 8 GB

// Check memory usage
let active = Memory.activeMemory
let peak = Memory.peakMemory
let cached = Memory.cacheMemory

// Clear cache if needed
Memory.clearCache()
```

### Wired Memory Tickets for Concurrent Inference

Use tickets instead of deprecated `withWiredLimit` APIs when coordinating concurrent GPU requests.

```swift
let policy = WiredSumPolicy()

// Reservation for long-lived model state.
let weights = policy.ticket(size: weightsBytes, kind: .reservation)
_ = await weights.start()

// Active ticket for request-local state.
let request = policy.ticket(size: kvBytes, kind: .active)
try await request.withWiredLimit {
    // run inference
}

_ = await weights.end()
```

Why this is better:

- Admission control can block instead of overcommitting memory.
- Shrink hysteresis avoids rapid limit oscillation under load.
- Reservations model long-lived state without keeping wired limits elevated while idle.

## Thread Safety Checklist

### DO

- Create MLXArrays within the task/actor that uses them
- Use `eval()` before extracting values to send across boundaries
- Use actors to encapsulate MLX state
- Pass Sendable types (`[Float]`, `Int`, etc.) between tasks
- Use explicit stream parameters for device/stream control
- Use `WiredMemoryTicket.withWiredLimit` for concurrent GPU workloads

### DON'T

- Share MLXArray instances across tasks
- Assume lazy operations are thread-safe
- Create arrays in one task and use in another
- Ignore the evalLock when doing custom threading
- Use deprecated `GPU.withWiredLimit` / `Memory.withWiredLimit` in new code

## Example: Concurrent Batch Processing

```swift
actor BatchProcessor {
    private let model: MyModel

    init(model: MyModel) {
        self.model = model
    }

    func processBatch(_ data: [[Float]]) -> [[Float]] {
        var results: [[Float]] = []

        for item in data {
            let input = MLXArray(item)
            let output = model(input)
            eval(output)
            results.append(output.asArray(Float.self))
        }

        return results
    }
}

// Usage
let processor = BatchProcessor(model: model)

// Process multiple batches concurrently (actor serializes actual GPU work)
async let batch1 = processor.processBatch(data1)
async let batch2 = processor.processBatch(data2)

let allResults = await [batch1, batch2]
```
