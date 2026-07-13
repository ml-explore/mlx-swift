# Distributed Primitives API Reference

Complete API reference for `DistributedGroup` and `DistributedBackend`.

## DistributedGroup

A wrapper around the MLX C distributed group handle. Represents a group of independent MLX ranks/processes that can communicate using collective operations.

```swift
public final class DistributedGroup
```

`DistributedGroup` is intentionally not `Sendable`. Treat it as an opaque runtime handle and keep it within a single isolation domain.

### Properties

#### rank

The rank of this process in the group (0-based index).

```swift
public var rank: Int { get }
```

```swift
let group = DistributedGroup()
print("I am rank \(group.rank)")  // e.g., "I am rank 0"
```

#### size

The number of ranks in the group.

```swift
public var size: Int { get }
```

```swift
let group = DistributedGroup()
print("Group has \(group.size) ranks")  // e.g., "Group has 2 ranks"
```

### Methods

#### split(color:key:)

Split this group into sub-groups based on the provided color.

```swift
public func split(color: Int, key: Int = -1) throws -> DistributedGroup
```

**Parameters:**
- `color`: Ranks with the same color are placed in the same sub-group.
- `key`: Determines rank ordering in the new group. Negative value uses the current rank. Default is `-1`.

**Returns:** A new `DistributedGroup` for the sub-group.

> **Warning:** Ring and JACCL backends do not support `split`. Only MPI (not available on macOS) supports it. This is now a call-time `throw`, so catch it with normal Swift `do` / `catch`.

```swift
do {
    let subGroup = try group.split(color: 0, key: group.rank)
    print("Created subgroup with size \(subGroup.size)")
} catch {
    print("Split not supported: \(error)")
}
```

### Lifecycle

Groups are created via `DistributedGroup()`, `DistributedGroup(backend:)`, or `DistributedGroup(strict:)`. The C API does not expose `mlx_distributed_group_free()`, so groups leak a small amount of memory on deallocation. This has minimal practical impact since groups are typically singleton-like and long-lived.

---

## DistributedBackend

Choose a backend and check whether it is available on the current runtime.

```swift
public enum DistributedBackend: String, CaseIterable, Sendable
```

Known cases: `.any`, `.ring`, `.jaccl`, `.mpi`, `.nccl`.
Use `.any` to let MLX choose the best available backend automatically.

### Properties

#### isAvailable

Check if a distributed communication backend is available.

```swift
public var isAvailable: Bool { get }
```

**Returns:** `true` when that backend is available.

```swift
// Check if any backend is available
if DistributedBackend.any.isAvailable {
    print("Distributed backend ready")
}

// Check a specific backend
if DistributedBackend.ring.isAvailable {
    print("Ring backend ready")
}
```

## DistributedGroup Constructors

#### init()

Initialize the distributed backend using `.any` and return the group containing
all discoverable ranks.

```swift
public init()
```

Returns a singleton group (rank 0, size 1) if no distributed backend can be initialized.
Equivalent to `DistributedGroup(backend: .any)`.

```swift
let group = DistributedGroup()
```

#### init(backend:)

Initialize the distributed backend and return the group containing all discoverable ranks.

```swift
public init(backend: DistributedBackend)
```

**Parameters:**
- `backend`: The backend to use.

Unlike `init(strict:)`, this preserves MLX's fallback behavior and returns a
singleton group (rank 0, size 1) if the requested backend cannot form a real
distributed group.

```swift
// Non-strict: always returns a group (size-1 fallback)
let group = DistributedGroup(backend: .ring)
```

#### init(strict:)

Initialize the distributed backend and return a real distributed group.

```swift
public init(strict backend: DistributedBackend) throws
```

```swift
do {
    let group = try DistributedGroup(strict: .ring)
    print("Ring group size: \(group.size)")
} catch {
    print("Couldn't form a ring group: \(error)")
}
```

## DistributedGroup Collective Operations

All collective operations accept a `stream` parameter (`StreamOrDevice`, default `.default`). Distributed operations only have CPU implementations.
`allSum`, `allGather`, `allMax`, and `allMin` remain lazy and non-throwing.
Use `withError { ... }` plus `checkedEval(...)` if you need Swift errors at the
evaluation boundary. On a singleton group, those collectives behave as identity
operations. `sumScatter` is now `throws` for immediate validation/setup errors
but may still report backend failures only at evaluation time.

#### allSum(_:stream:)

Sum-reduce the array across all ranks. Each rank contributes its local array and all ranks receive the element-wise sum.

```swift
public func allSum(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The local array to sum.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise sum across all ranks.

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6]
let result = group.allSum(localData)
eval(result)
// Both ranks get: [5, 7, 9]
```

#### allGather(_:stream:)

Gather arrays from all ranks. Each rank contributes its local array and all ranks receive the concatenated result.

```swift
public func allGather(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The local array to gather.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The concatenation of arrays from all ranks.

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6]
let result = group.allGather(localData)
eval(result)
// Both ranks get: [1, 2, 3, 4, 5, 6]
```

Works with multi-dimensional arrays:
```swift
// Rank 0: [[1, 2], [3, 4]], Rank 1: [[5, 6], [7, 8]]
// Result: [[1, 2], [3, 4], [5, 6], [7, 8]] shape [4, 2]
```

#### allMax(_:stream:)

Max-reduce the array across all ranks. Each rank contributes its local array and all ranks receive the element-wise maximum.

```swift
public func allMax(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The local array to max-reduce.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise maximum across all ranks.

```swift
// Rank 0: [1, 5, 3], Rank 1: [4, 2, 6]
let result = group.allMax(localData)
eval(result)
// Both ranks get: [4, 5, 6]
```

#### allMin(_:stream:)

Min-reduce the array across all ranks. Each rank contributes its local array and all ranks receive the element-wise minimum.

```swift
public func allMin(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The local array to min-reduce.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise minimum across all ranks.

```swift
// Rank 0: [1, 5, 3], Rank 1: [4, 2, 6]
let result = group.allMin(localData)
eval(result)
// Both ranks get: [1, 2, 3]
```

#### sumScatter(_:stream:)

Sum-reduce and scatter the array across all ranks. The array is sum-reduced and the result is scattered (split) across ranks so each rank receives its portion.

```swift
public func sumScatter(_ array: MLXArray, stream: StreamOrDevice = .default) throws -> MLXArray
```

**Parameters:**
- `array`: The local array to sum-scatter.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** This rank's portion of the sum-scattered result.

> **Warning:** `sumScatter` only throws immediate validation/setup errors. On the ring backend, the unsupported-operation error still appears when the returned array is evaluated.

```swift
// Both ranks: [1, 2, 3, 4], sum = [2, 4, 6, 8]
// Rank 0 gets: [2, 4], Rank 1 gets: [6, 8]
do {
    try withError {
        let result = try group.sumScatter(localData)
        try checkedEval(result)
    }
} catch {
    print("sumScatter failed: \(error)")
}
```

#### send(_:to:stream:)

Send an array to another rank in the group. Returns a dependency token that can be used to sequence operations.

```swift
public func send(_ array: MLXArray, to dst: Int, stream: StreamOrDevice = .default) throws -> MLXArray
```

**Parameters:**
- `array`: The array to send.
- `dst`: The destination rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** A dependency token (an `MLXArray`).

> **Note:** Requires group size ≥ 2. This is now a call-time `throw` on singleton groups or invalid rank setups. Transport/backend failures may still surface later when the returned token is evaluated.

```swift
do {
    let token = try group.send(data, to: 1)
    try checkedEval(token)
} catch {
    print("send failed: \(error)")
}
```

#### recv(shape:dtype:from:stream:)

Receive an array from another rank in the group.

```swift
public func recv(
    shape: [Int], dtype: DType, from src: Int, stream: StreamOrDevice = .default
) throws -> MLXArray
```

**Parameters:**
- `shape`: The shape of the expected array.
- `dtype`: The data type of the expected array.
- `src`: The source rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array.

> **Note:** Requires group size ≥ 2. This now throws for immediate validation/setup failures. Backend failures can still surface when the returned array is evaluated.

```swift
do {
    let received = try group.recv(shape: [3], dtype: .float32, from: 0)
    try checkedEval(received)
} catch {
    print("recv failed: \(error)")
}
```

#### recvLike(_:from:stream:)

Receive an array from another rank, using a template array for shape and dtype.

```swift
public func recvLike(
    _ array: MLXArray, from src: Int, stream: StreamOrDevice = .default
) throws -> MLXArray
```

**Parameters:**
- `array`: Template array whose shape and dtype define the expected result.
- `src`: The source rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array with the same shape and dtype as the template.

> **Note:** Requires group size ≥ 2. This now throws for immediate validation/setup failures. Backend failures can still surface when the returned array is evaluated.

```swift
let template = MLXArray(converting: [0.0, 0.0, 0.0])
do {
    let received = try group.recvLike(template, from: 0)
    try checkedEval(received)
} catch {
    print("recvLike failed: \(error)")
}
```

## Supported Data Types

All collective operations preserve the input dtype. Tested types include:
- `.float32` (default)
- `.float16`
- `.bfloat16`
- `.int32`

## Stream Parameter

The `stream` parameter is accepted by all collective operations but distributed ops only have CPU implementations. Passing a GPU stream will cause the operation to be scheduled on CPU internally.
