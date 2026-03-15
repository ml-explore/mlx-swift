# Distributed Primitives API Reference

Complete API reference for `DistributedGroup` and `MLXDistributed` enum.

## DistributedGroup

A wrapper around the MLX C distributed group handle. Represents a group of independent MLX processes that can communicate using collective operations.

```swift
public final class DistributedGroup: @unchecked Sendable
```

### Properties

#### rank

The rank of this process in the group (0-based index).

```swift
public var rank: Int { get }
```

```swift
let group = MLXDistributed.`init`()!
print("I am rank \(group.rank)")  // e.g., "I am rank 0"
```

#### size

The number of processes in the group.

```swift
public var size: Int { get }
```

```swift
let group = MLXDistributed.`init`()!
print("Group has \(group.size) members")  // e.g., "Group has 2 members"
```

### Methods

#### split(color:key:)

Split this group into sub-groups based on the provided color.

```swift
public func split(color: Int, key: Int = -1) -> DistributedGroup
```

**Parameters:**
- `color`: Processes with the same color are placed in the same sub-group.
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

Groups are created via `MLXDistributed.init(strict:)`. The C API does not expose `mlx_distributed_group_free()`, so groups leak a small amount of memory on deallocation. This has minimal practical impact since groups are typically singleton-like and long-lived.

---

## MLXDistributed

Collection of distributed communication operations.

```swift
public enum MLXDistributed
```

### Static Methods

#### isAvailable()

Check if a distributed communication backend is available.

```swift
public static func isAvailable() -> Bool
```

**Returns:** `true` when the ring backend (or another backend) is compiled and available.

```swift
if MLXDistributed.isAvailable() {
    print("Distributed backend ready")
}
```

#### init(strict:)

Initialize the distributed backend and return the group containing all discoverable processes.

```swift
public static func `init`(strict: Bool = false) -> DistributedGroup?
```

**Parameters:**
- `strict`: If `true`, returns `nil` on initialization failure instead of falling back to a singleton group. Default is `false`.

**Returns:** The `DistributedGroup` for this process, or `nil` if `strict` is `true` and initialization failed.

When `strict` is `false` (default), returns a singleton group (rank 0, size 1) if no distributed backend can be initialized. MLX-C does not expose a backend selection parameter — it tries JACCL first, then ring.

```swift
// Non-strict: always returns a group (size-1 fallback)
let group = MLXDistributed.`init`()!

// Strict: returns nil if no multi-process backend available
guard let group = MLXDistributed.`init`(strict: true) else {
    print("No distributed backend configured")
    return
}
```

### Collective Operations

All collective operations accept a `stream` parameter (`StreamOrDevice`, default `.default`). Distributed operations only have CPU implementations.

#### allSum(_:group:stream:)

Sum-reduce the array across all processes. Each process contributes its local array and all processes receive the element-wise sum.

```swift
public static func allSum(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The local array to sum.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise sum across all processes.

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6]
let result = MLXDistributed.allSum(localData, group: group)
eval(result)
// Both ranks get: [5, 7, 9]
```

#### allGather(_:group:stream:)

Gather arrays from all processes. Each process contributes its local array and all processes receive the concatenated result.

```swift
public static func allGather(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The local array to gather.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The concatenation of arrays from all processes.

```swift
// Rank 0: [1, 2, 3], Rank 1: [4, 5, 6]
let result = MLXDistributed.allGather(localData, group: group)
eval(result)
// Both ranks get: [1, 2, 3, 4, 5, 6]
```

Works with multi-dimensional arrays:
```swift
// Rank 0: [[1, 2], [3, 4]], Rank 1: [[5, 6], [7, 8]]
// Result: [[1, 2], [3, 4], [5, 6], [7, 8]] shape [4, 2]
```

#### allMax(_:group:stream:)

Max-reduce the array across all processes. Each process contributes its local array and all processes receive the element-wise maximum.

```swift
public static func allMax(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The local array to max-reduce.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise maximum across all processes.

```swift
// Rank 0: [1, 5, 3], Rank 1: [4, 2, 6]
let result = MLXDistributed.allMax(localData, group: group)
eval(result)
// Both ranks get: [4, 5, 6]
```

#### allMin(_:group:stream:)

Min-reduce the array across all processes. Each process contributes its local array and all processes receive the element-wise minimum.

```swift
public static func allMin(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The local array to min-reduce.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The element-wise minimum across all processes.

```swift
// Rank 0: [1, 5, 3], Rank 1: [4, 2, 6]
let result = MLXDistributed.allMin(localData, group: group)
eval(result)
// Both ranks get: [1, 2, 3]
```

#### sumScatter(_:group:stream:)

Sum-reduce and scatter the array across all processes. The array is sum-reduced and the result is scattered (split) across processes so each process receives its portion.

```swift
public static func sumScatter(
    _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The local array to sum-scatter.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** This process's portion of the sum-scattered result.

> **Warning:** Not implemented in the ring backend. Will raise a C++ error at eval time. Use `withErrorHandler` to catch the error gracefully.

```swift
// Both ranks: [1, 2, 3, 4], sum = [2, 4, 6, 8]
// Rank 0 gets: [2, 4], Rank 1 gets: [6, 8]
withErrorHandler({ errMsg in
    print("sumScatter not supported: \(errMsg)")
}) {
    let result = MLXDistributed.sumScatter(localData, group: group)
    eval(result)
}
```

#### send(_:to:group:stream:)

Send an array to another process in the group. Returns a dependency token that can be used to sequence operations.

```swift
public static func send(
    _ array: MLXArray, to dst: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: The array to send.
- `dst`: The destination rank.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** A dependency token (an `MLXArray`).

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let token = MLXDistributed.send(data, to: 1, group: group)
eval(token)  // Must eval to initiate the send
```

#### recv(shape:dtype:from:group:stream:)

Receive an array from another process in the group.

```swift
public static func recv(
    shape: [Int], dtype: DType, from src: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `shape`: The shape of the expected array.
- `dtype`: The data type of the expected array.
- `src`: The source rank.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array.

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let received = MLXDistributed.recv(
    shape: [3], dtype: .float32, from: 0, group: group)
eval(received)
```

#### recvLike(_:from:group:stream:)

Receive an array from another process, using a template array for shape and dtype.

```swift
public static func recvLike(
    _ array: MLXArray, from src: Int, group: DistributedGroup,
    stream: StreamOrDevice = .default
) -> MLXArray
```

**Parameters:**
- `array`: Template array whose shape and dtype define the expected result.
- `src`: The source rank.
- `group`: The communication group.
- `stream`: Stream or device to evaluate on. Default is `.default`.

**Returns:** The received array with the same shape and dtype as the template.

> **Note:** Requires group size ≥ 2. Raises an error on singleton groups.

```swift
let template = MLXArray(converting: [0.0, 0.0, 0.0])
let received = MLXDistributed.recvLike(template, from: 0, group: group)
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
