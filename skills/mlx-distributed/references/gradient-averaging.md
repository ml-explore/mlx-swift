# Gradient Averaging API Reference

Complete API reference for `averageGradients`.

## averageGradients(gradients:group:allReduceSize:communicationType:communicationStream:)

Average a gradient tree across the processes in the distributed group.

```swift
public func averageGradients(
    gradients: ModuleParameters,
    group: DistributedGroup? = nil,
    allReduceSize: Int = 32 * 1024 * 1024,
    communicationType: DType? = nil,
    communicationStream: StreamOrDevice? = nil
) -> ModuleParameters
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradients` | `ModuleParameters` | — | The gradient tree (typically from `Module.parameters()` or `Module.trainableParameters()`) |
| `group` | `DistributedGroup?` | `nil` | The distributed group. If `nil`, uses `MLXDistributed.init()` |
| `allReduceSize` | `Int` | `32 * 1024 * 1024` (32 MiB) | Maximum byte size for batching gradient arrays into a single all-reduce call. Set to 0 or negative to disable batching |
| `communicationType` | `DType?` | `nil` | If provided, cast each gradient to this type before communication and cast back to original type after. Used for bandwidth reduction (e.g., `.float16`) |
| `communicationStream` | `StreamOrDevice?` | `nil` | Optional stream for communication. If `nil`, the default stream is used |

### Returns

The averaged gradient tree with the same structure as the input.

---

## Behavior

### N == 1 Optimization

When the group has a single member, the gradients are returned unchanged immediately. This is the fast path for single-process execution.

```swift
let group = MLXDistributed.`init`()!  // size-1 group
let averaged = averageGradients(gradients: grads, group: group)
// averaged is identical to grads (no communication)
```

### Averaging Formula

For each gradient array `g` across `N` processes:

```
averaged_g = allSum(g) / N
```

### Batching Behavior (allReduceSize)

When `allReduceSize > 0` (default: 32 MiB):

1. Flatten all gradient arrays to 1D.
2. Group gradients into batches where cumulative byte size ≥ `allReduceSize`.
3. Concatenate each batch into a single large array.
4. Perform one `allSum` per batch (fewer network round-trips).
5. Split the result back into individual gradient arrays.
6. Reshape each gradient back to its original shape.

When `allReduceSize <= 0`:

Each gradient is averaged independently with its own `allSum` call. This may result in more network round-trips but avoids concatenation overhead for very large gradients.

```swift
// Default batched mode (32 MiB chunks)
let avg1 = averageGradients(gradients: grads, group: group)

// Non-batched mode: one allSum per gradient
let avg2 = averageGradients(gradients: grads, group: group, allReduceSize: 0)

// Small batch size (forces many batches)
let avg3 = averageGradients(gradients: grads, group: group, allReduceSize: 1024)

// Very large batch size (everything in one call)
let avg4 = averageGradients(
    gradients: grads, group: group, allReduceSize: 1024 * 1024 * 1024)
```

### communicationType — Cast-on-Wire

When `communicationType` is provided, each gradient is:

1. Cast to `communicationType` before the `allSum` call.
2. The `allSum` is performed in the cast dtype (reduced bandwidth).
3. Cast back to the original dtype after receiving the result.
4. Divided by `N`.

This is useful for bandwidth reduction — e.g., casting float32 gradients to float16 halves the data transferred.

```swift
// Cast to float16 for communication, cast back to float32 after
let averaged = averageGradients(
    gradients: grads, group: group, communicationType: .float16)

// Cast to bfloat16 for better numerical stability
let averaged2 = averageGradients(
    gradients: grads, group: group, communicationType: .bfloat16)
```

The batching threshold uses `communicationType.size` (if provided) for computing byte sizes, matching Python's behavior.

### Mixed-Dtype Fallback

If the gradient tree contains arrays with different dtypes (e.g., some float32 and some float16), the batched mode falls back to non-batched mode (recursive call with `allReduceSize: 0`). This is because concatenation requires all arrays to have the same dtype.

```swift
// Mixed-dtype gradient tree: some float32, some float16
var grads = ModuleParameters()
grads["weight"] = .value(MLXRandom.uniform(0 ..< 1, [4, 8]))           // float32
grads["bias"] = .value(MLXRandom.uniform(0 ..< 1, [4]).asType(.float16))  // float16

// Automatically falls back to non-batched mode
let averaged = averageGradients(gradients: grads, group: group)
```

---

## Complete Training Loop Example

```swift
import MLX
import MLXNN
import MLXOptimizers

// Initialize distributed group
let group = MLXDistributed.`init`()!

// Set CPU device (distributed ops are CPU-only)
Device.withDefaultDevice(.cpu) {

    let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
    let optimizer = Adam(learningRate: 0.001)

    func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
        let logits = model(x)
        return crossEntropy(logits: logits, targets: y, reduction: .mean)
    }

    let lossAndGrad = valueAndGrad(model: model, loss)

    for epoch in 0 ..< numEpochs {
        for (x, y) in dataLoader {
            // Each rank computes loss and gradients on its own data shard
            let (lossValue, grads) = lossAndGrad(model, x, y)

            // Average gradients across all ranks
            // - Batched allSum with 32 MiB chunks (default)
            // - Cast to float16 for bandwidth reduction
            let avgGrads = averageGradients(
                gradients: grads,
                group: group,
                communicationType: .float16
            )

            // Update model (same on all ranks since gradients are averaged)
            optimizer.update(model: model, gradients: avgGrads)
            eval(model, optimizer)
        }
    }
}
```

---

## Parameter Combinations

| allReduceSize | communicationType | Behavior |
|---------------|-------------------|----------|
| `> 0` (default 32 MiB) | `nil` | Batched allSum, native dtype |
| `> 0` | `.float16` | Batched allSum, cast to float16 for wire |
| `0` or negative | `nil` | Per-gradient allSum, native dtype |
| `0` or negative | `.float16` | Per-gradient allSum, cast to float16 for wire |
| Any | Any | Mixed-dtype tree → falls back to non-batched |
