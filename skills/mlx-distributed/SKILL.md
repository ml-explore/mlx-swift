---
name: mlx-distributed
description: MLX Swift Distributed - Multi-device communication for tensor parallelism across Apple Silicon nodes via ring (TCP/IP) or JACCL (RDMA/Thunderbolt 5) backends
triggers:
  - mlx distributed
  - distributed mlx
  - tensor parallelism swift
  - multi-device inference
  - ring backend
  - jaccl
  - thunderbolt 5 ml
  - sharded linear
  - distributed training
  - multi-node inference
---

# MLX Swift Distributed

MLX Swift Distributed provides multi-device communication primitives for tensor parallelism across Apple Silicon nodes. It supports two backends: ring (TCP/IP sockets) and JACCL (RDMA over Thunderbolt 5). The API enables collective operations, distributed neural network layers, and gradient averaging for multi-process training and inference.

## When to Use This Skill

- Multi-device / multi-node model inference or training
- Tensor parallelism (column/row sharding)
- Gradient averaging across distributed workers
- Collective operations (allSum, allGather, allMax, allMin, send, recv)

## Architecture Overview

```
averageGradients / shardLinear / shardInPlace (utilities)
         ↓
AllToShardedLinear / ShardedToAllLinear (NN layers)
         ↓
MLXDistributed (collective ops: allSum, allGather, send, recv, etc.)
         ↓
DistributedGroup (group management, rank, size, split)
         ↓
MLX-C distributed (ring TCP + JACCL RDMA backends)
```

## Key File Reference

| Purpose | File Path |
|---------|-----------|
| Distributed group + collective ops | Source/MLX/Distributed.swift |
| NN layers + sharding utilities | Source/MLXNN/Distributed.swift |
| Example multi-process worker | Source/Examples/DistributedWorker.swift |
| Distributed primitive tests | Tests/MLXTests/DistributedTests.swift |
| Distributed NN layer tests | Tests/MLXTests/DistributedNNTests.swift |

## Quick Start

### Basic Group Initialization

```swift
import MLX

// Check if a distributed backend is available
guard MLXDistributed.isAvailable() else {
    print("No distributed backend available")
    return
}

// Initialize the distributed group (non-strict: falls back to size-1 group)
guard let group = MLXDistributed.`init`() else {
    return
}
print("Rank \(group.rank) of \(group.size)")

// Strict mode: returns nil if no multi-process backend can initialize
let strictGroup = MLXDistributed.`init`(strict: true)
```

### Simple allSum Collective Operation

```swift
import MLX

let group = MLXDistributed.`init`()!

// Each process contributes its local array
let localData = MLXArray(converting: [1.0, 2.0, 3.0])

// All processes receive the element-wise sum
let globalSum = MLXDistributed.allSum(localData, group: group)
eval(globalSum)
```

### Creating a Sharded Linear Layer

```swift
import MLX
import MLXNN

let group = MLXDistributed.`init`()!

// Start with a standard Linear layer (e.g., loaded from a model)
let linear = Linear(1024, 1024, bias: true)
eval(linear)

// Convert to a distributed sharded layer (auto-detects Linear vs QuantizedLinear)
let sharded = shardLinear(module: linear, sharding: .allToSharded, group: group)

// Use the sharded layer in a forward pass
let input = MLXRandom.uniform(0 ..< 1, [4, 1024])
let output = (sharded as! UnaryLayer)(input)
```

### Using averageGradients in a Training Loop

```swift
import MLX
import MLXNN
import MLXOptimizers

let group = MLXDistributed.`init`()!
let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
let optimizer = Adam(learningRate: 0.001)

func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    let logits = model(x)
    return crossEntropy(logits: logits, targets: y, reduction: .mean)
}

let lossAndGrad = valueAndGrad(model: model, loss)

for (x, y) in dataLoader {
    let (lossValue, grads) = lossAndGrad(model, x, y)

    // Average gradients across all distributed processes
    let avgGrads = averageGradients(gradients: grads, group: group)

    optimizer.update(model: model, gradients: avgGrads)
    eval(model, optimizer)
}
```

## Primary Workflow: Collective Operations

See [primitives.md](references/primitives.md) for complete API reference.

### allSum — Sum-reduce across all processes

```swift
public static func allSum(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6] → Both get: [5, 7, 9]
let result = MLXDistributed.allSum(localData, group: group)
eval(result)
```

### allGather — Concatenate arrays from all processes

```swift
public static func allGather(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6] → Both get: [1, 2, 3, 4, 5, 6]
let result = MLXDistributed.allGather(localData, group: group)
eval(result)
```

### allMax — Element-wise maximum across all processes

```swift
public static func allMax(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

### allMin — Element-wise minimum across all processes

```swift
public static func allMin(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

### sumScatter — Sum-reduce and scatter across processes

```swift
public static func sumScatter(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

> **Warning:** `sumScatter` is not implemented in the ring backend. It will raise an error at eval time. MPI and NCCL backends support it.

### send — Send an array to another process

```swift
public static func send(
    _ array: MLXArray, to dst: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray  // Returns a dependency token
```

```swift
// Rank 0 sends data to rank 1
let token = MLXDistributed.send(data, to: 1, group: group)
eval(token)
```

### recv — Receive an array from another process

```swift
public static func recv(
    shape: [Int], dtype: DType, from src: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray
```

```swift
// Rank 1 receives data from rank 0
let received = MLXDistributed.recv(shape: [3], dtype: .float32, from: 0, group: group)
eval(received)
```

### recvLike — Receive using a template array

```swift
public static func recvLike(
    _ array: MLXArray, from src: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray
```

```swift
// Uses template's shape and dtype automatically
let template = MLXArray(converting: [0.0, 0.0, 0.0])
let received = MLXDistributed.recvLike(template, from: 0, group: group)
eval(received)
```

> **Note:** `send`, `recv`, and `recvLike` require a multi-process setup (group size ≥ 2). They will raise errors on a singleton group.

## Secondary Workflow: Distributed NN Layers

See [nn-layers.md](references/nn-layers.md) for complete API reference.

### AllToShardedLinear — Column-parallel sharding

Each process applies part of the affine transformation such that the output is sharded across the group. Gradients are aggregated via `sumGradients`.

```swift
// Create from an existing Linear layer
let sharded = AllToShardedLinear.fromLinear(linear, segments: 1, group: group)

// Or initialize directly
let layer = AllToShardedLinear(
    inputDimensions: 1024, outputDimensions: 512, bias: true, group: group)

// Forward: input [batch, inDims] → output [batch, outDims/N]
let output = layer(input)
```

### ShardedToAllLinear — Row-parallel sharding

Each process applies part of the affine transformation and then aggregates the results via `allSum`. All nodes receive the same output.

```swift
// Create from an existing Linear layer
let sharded = ShardedToAllLinear.fromLinear(linear, segments: 1, group: group)

// Or initialize directly
let layer = ShardedToAllLinear(
    inputDimensions: 1024, outputDimensions: 512, bias: true, group: group)

// Forward: input [batch, inDims/N] → output [batch, outDims]
let output = layer(input)
```

### QuantizedAllToShardedLinear — Quantized column-parallel

Quantized equivalent of `AllToShardedLinear`. Parameters are frozen and excluded from gradient computation.

```swift
let sharded = QuantizedAllToShardedLinear.fromQuantizedLinear(
    quantizedLinear, segments: 1, group: group)
```

### QuantizedShardedToAllLinear — Quantized row-parallel

Quantized equivalent of `ShardedToAllLinear`. Parameters are frozen and excluded from gradient computation.

```swift
let sharded = QuantizedShardedToAllLinear.fromQuantizedLinear(
    quantizedLinear, segments: 1, group: group)
```

### sumGradients — Identity forward, allSum backward

```swift
public func sumGradients(group: DistributedGroup) -> (MLXArray) -> MLXArray
```

Returns a closure that passes through the input unchanged in the forward pass but performs `allSum` on cotangents during backpropagation. Used internally by `AllToShardedLinear` and `QuantizedAllToShardedLinear`.

## Tertiary Workflow: Sharding Utilities

See [sharding.md](references/sharding.md) for complete API reference.

### shardLinear — Create a distributed layer from Linear or QuantizedLinear

```swift
public func shardLinear(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
) -> Module
```

Automatically dispatches to the correct distributed layer type. `QuantizedLinear` is checked before `Linear` (since it is a subclass).

```swift
let distributed = shardLinear(module: linear, sharding: .allToSharded, group: group)
// Returns AllToShardedLinear for Linear, QuantizedAllToShardedLinear for QuantizedLinear
```

### shardInPlace — Shard parameters without changing module type

```swift
public func shardInPlace(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
)
```

### ShardingType Enum

```swift
public enum ShardingType {
    case allToSharded   // Column-parallel: replicated input → sharded output
    case shardedToAll   // Row-parallel: sharded input → replicated output
}
```

### Segments Parameter

The `segments` parameter supports fused weights (e.g., `segments: 3` for fused QKV projections). Each segment is split independently across the group, then concatenated.

```swift
// Fused QKV: weight shape [3*hidden, hidden]
let sharded = shardLinear(module: fusedQKV, sharding: .allToSharded, segments: 3, group: group)
```

## Quaternary Workflow: Gradient Averaging

See [gradient-averaging.md](references/gradient-averaging.md) for complete API reference.

```swift
public func averageGradients(
    gradients: ModuleParameters,
    group: DistributedGroup? = nil,
    allReduceSize: Int = 32 * 1024 * 1024,  // 32 MiB
    communicationType: DType? = nil,
    communicationStream: StreamOrDevice? = nil
) -> ModuleParameters
```

```swift
let grads = lossAndGrad(model, x, y).1

// Default: batched allSum with 32 MiB chunks
let avgGrads = averageGradients(gradients: grads, group: group)

// Non-batched: average each gradient independently
let avgGrads2 = averageGradients(gradients: grads, group: group, allReduceSize: 0)

// Cast to float16 before communication for bandwidth reduction
let avgGrads3 = averageGradients(
    gradients: grads, group: group, communicationType: .float16)
```

## Best Practices

### DO

- **Use CPU device for distributed operations**: Distributed ops only have CPU implementations. Set `Device.withDefaultDevice(.cpu) { ... }` in worker processes.
- **Use `_exit(0)` in multi-process workers**: The ring backend's TCP socket destructors can hang waiting for peer socket closure. Use `_exit(0)` to bypass cleanup handlers.
- **Use `shardLinear` to auto-detect layer types**: It checks `QuantizedLinear` before `Linear` (subclass ordering) and dispatches correctly.
- **Use `averageGradients` with `communicationType`** for bandwidth reduction: Cast gradients to `.float16` or `.bfloat16` before communication.
- **Check `MLXDistributed.isAvailable()` before initializing**: Verify a backend exists before attempting group creation.
- **Call `eval()` before distributed communication**: Ensure arrays are materialized before sending across processes.
- **Use sequential port allocation in tests**: Avoid ephemeral port collisions by using a monotonically increasing port counter with a random base.

### DON'T

- **Don't try to use distributed ops on GPU**: They only have CPU implementations. GPU streams will fail.
- **Don't call `group.split()`**: Ring and JACCL backends don't support it (MPI only). The call will raise an error.
- **Don't use `sumScatter` with ring backend**: Not implemented; will raise an error at eval time.
- **Don't forget to `eval()` before distributed communication**: Unevaluated arrays can cause unexpected behavior in collective ops.
- **Don't share `DistributedGroup` across actors without synchronization**: While `DistributedGroup` is `@unchecked Sendable`, the underlying C++ object is not thread-safe.

## Known Upstream Limitations

| Limitation | Impact |
|------------|--------|
| MLX-C doesn't expose backend selection parameter | Cannot choose between JACCL and ring; tries JACCL first, falls back to ring |
| `mlx_distributed_group_free()` not exposed in public C API | Groups leak small amounts of memory on deallocation (minimal practical impact) |
| `group.split()` unsupported by ring and JACCL backends | Only MPI (not available on macOS) supports sub-group creation |
| `sumScatter`/`reduceScatter` not implemented in ring backend | Use allSum + manual slicing as a workaround |
| All distributed ops are CPU-only | Must set CPU device in worker processes |

## Deprecated Patterns

There are currently no deprecated patterns in the distributed API, as it is a new addition.

## Swift Concurrency Notes

- **`DistributedGroup` is `@unchecked Sendable`**: The class wraps a C handle and can be passed across concurrency boundaries, but the underlying C++ object is not thread-safe.
- **Use actors to encapsulate distributed state**: Coordinate group access and collective operations within a single actor.
- **Workers should use `_exit(0)` for clean termination**: Avoids ring backend destructor hangs in multi-process setups.

## Reference Documentation

- [Primitives](references/primitives.md) - DistributedGroup and MLXDistributed collective operations
- [NN Layers](references/nn-layers.md) - Distributed linear layers and sumGradients
- [Sharding](references/sharding.md) - shardLinear, shardInPlace, and ShardingType
- [Gradient Averaging](references/gradient-averaging.md) - averageGradients with batching and type casting
- [Multi-Process](references/multi-process.md) - Worker setup, hostfile format, and testing patterns
