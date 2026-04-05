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
public func split(color: Int, key: Int = -1) -> DistributedGroup
```

**Parameters:**
- `color`: Ranks with the same color are placed in the same sub-group.
- `key`: Determines rank ordering in the new group. Negative value uses the current rank. Default is `-1`.

**Returns:** A new `DistributedGroup` for the sub-group.

> **Warning:** Ring and JACCL backends do not support `split`. Only MPI (not available on macOS) supports it. The call will raise a C++ error: `"[ring] Group split not supported."` Use `withErrorHandler` to catch it gracefully.

```swift
// Attempt to split (will fail on ring/JACCL backends)
withErrorHandler({ errMsg in
    print("Split not supported: \(errMsg)")
}) {
    let subGroup = group.split(color: 0, key: rank)
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

#### init?(strict:)

Initialize the distributed backend and return `nil` when no real distributed backend can be formed.

```swift
public init?(strict backend: DistributedBackend)
```

```swift
// Strict: returns nil if the requested backend can't form a real group
guard let group = DistributedGroup(strict: .ring) else {
    print("Ring backend unavailable")
    return
}
```

## DistributedGroup Collective Operations

All collective operations accept a `stream` parameter (`StreamOrDevice`, default `.default`). Distributed operations only have CPU implementations.
On a singleton group, `allSum`, `allGather`, `allMax`, `allMin`, and
`sumScatter` behave as identity operations.

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
public func sumScatter(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The local array to sum-scatter.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** This rank's portion of the sum-scattered result.

> **Warning:** Not implemented in the ring backend. Will raise a C++ error at eval time. Use `withErrorHandler` to catch the error gracefully.

```swift
// Both ranks: [1, 2, 3, 4], sum = [2, 4, 6, 8]
// Rank 0 gets: [2, 4], Rank 1 gets: [6, 8]
withErrorHandler({ errMsg in
    print("sumScatter not supported: \(errMsg)")
}) {
    let result = group.sumScatter(localData)
    eval(result)
}
```

#### send(_:to:stream:)

Send an array to another rank in the group. Returns a dependency token that can be used to sequence operations.

```swift
public func send(_ array: MLXArray, to dst: Int, stream: StreamOrDevice = .default) -> MLXArray
```

**Parameters:**
- `array`: The array to send.
- `dst`: The destination rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** A dependency token (an `MLXArray`).

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let token = group.send(data, to: 1)
eval(token)  // Must eval to initiate the send
```

#### recv(shape:dtype:from:stream:)

Receive an array from another rank in the group.

```swift
public func recv(
    shape: [Int], dtype: DType, from src: Int, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `shape`: The shape of the expected array.
- `dtype`: The data type of the expected array.
- `src`: The source rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array.

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let received = group.recv(shape: [3], dtype: .float32, from: 0)
eval(received)
```

#### recvLike(_:from:stream:)

Receive an array from another rank, using a template array for shape and dtype.

```swift
public func recvLike(
    _ array: MLXArray, from src: Int, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: Template array whose shape and dtype define the expected result.
- `src`: The source rank.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array with the same shape and dtype as the template.

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let template = MLXArray(converting: [0.0, 0.0, 0.0])
let received = group.recvLike(template, from: 0)
eval(received)
```

## Supported Data Types

All collective operations preserve the input dtype. Tested types include:
- `.float32` (default)
- `.float16`
- `.bfloat16`
- `.int32`

## Stream Parameter

The `stream` parameter is accepted by all collective operations but distributed ops only have CPU implementations. Passing a GPU stream will cause the operation to be scheduled on CPU internally.
