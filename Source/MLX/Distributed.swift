// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Wrapper around the MLX C distributed group handle.
///
/// A `DistributedGroup` represents a group of independent MLX processes
/// that can communicate using collective operations. Use ``MLXDistributed/init(strict:)``
/// to create the initial group, then ``split(color:key:)`` to create sub-groups.
///
/// ### See Also
/// - ``MLXDistributed``
/// - ``MLXDistributed/init(strict:)``
public final class DistributedGroup: @unchecked Sendable {

    let ctx: mlx_distributed_group

    init(_ ctx: mlx_distributed_group) {
        self.ctx = ctx
    }

    deinit {
        // UPSTREAM GAP: mlx_distributed_group is a value type wrapping a
        // heap-allocated C++ Group object (void* ctx). Other MLX-C handle
        // types (mlx_device, mlx_stream, mlx_array, etc.) expose a public
        // free function (e.g., mlx_device_free), but MLX-C v0.5.0 does NOT
        // expose mlx_distributed_group_free(). The private C++ header
        // (mlx/c/private/distributed_group.h) has mlx_distributed_group_free_()
        // but it is an inline C++ function, inaccessible from Swift/C.
        //
        // Calling C free() on ctx is NOT safe because the underlying object
        // is allocated with C++ new and may have a non-trivial destructor.
        //
        // Practical impact is minimal: groups are typically singleton-like and
        // long-lived (one per distributed init, occasionally split). The C++
        // Group internally holds a shared_ptr to the backend, so the leaked
        // memory per group is small.
        //
        // TODO: File upstream issue to add mlx_distributed_group_free() to
        // the public MLX-C API, then call it here like Device.deinit calls
        // mlx_device_free(ctx).
    }

    /// The rank of this process in the group.
    public var rank: Int {
        Int(mlx_distributed_group_rank(ctx))
    }

    /// The number of processes in the group.
    public var size: Int {
        Int(mlx_distributed_group_size(ctx))
    }

    /// Split this group into sub-groups based on the provided color.
    ///
    /// Processes that use the same color will be placed in the same sub-group.
    /// The key defines the rank of the process in the new group — the smaller
    /// the key, the smaller the rank. If the key is negative, the rank in the
    /// current group is used.
    ///
    /// - Parameters:
    ///   - color: processes with the same color go to the same sub-group
    ///   - key: determines rank ordering in the new group (negative = use current rank)
    /// - Returns: a new ``DistributedGroup`` for the sub-group
    public func split(color: Int, key: Int = -1) -> DistributedGroup {
        let result = mlx_distributed_group_split(ctx, Int32(color), Int32(key))
        return DistributedGroup(result)
    }
}

/// Collection of distributed communication operations.
///
/// Use ``MLXDistributed`` to check for distributed backend availability,
/// initialize distributed communication, and perform collective operations
/// (all-reduce, gather, scatter, send, receive).
///
/// ```swift
/// // Initialize distributed communication
/// let group = MLXDistributed.init()
/// print("Rank \(group.rank) of \(group.size)")
///
/// // Perform an all-sum reduction
/// let data = MLXArray([1.0, 2.0, 3.0])
/// let sum = MLXDistributed.allSum(data, group: group)
/// ```
///
/// ### See Also
/// - ``DistributedGroup``
public enum MLXDistributed {

    /// Check if a distributed communication backend is available.
    ///
    /// Returns `true` when the ring backend (or another backend) is compiled and
    /// available for use.
    public static func isAvailable() -> Bool {
        mlx_distributed_is_available()
    }

    /// Initialize the distributed backend and return the group containing
    /// all discoverable processes.
    ///
    /// When `strict` is `false` (the default), returns a singleton group
    /// (rank 0, size 1) if no distributed backend can be initialized.
    /// When `strict` is `true`, returns `nil` if initialization fails
    /// (e.g., no hostfile configured).
    ///
    /// - Parameter strict: if `true`, return `nil` on initialization failure
    ///   instead of falling back to a singleton group
    /// - Returns: the ``DistributedGroup`` for this process, or `nil` if
    ///   `strict` is `true` and initialization failed
    public static func `init`(strict: Bool = false) -> DistributedGroup? {
        let group = mlx_distributed_init(strict)
        if group.ctx == nil {
            return nil
        }
        return DistributedGroup(group)
    }

    // MARK: - Collective Operations

    /// Sum-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise sum.
    ///
    /// - Parameters:
    ///   - array: the local array to sum
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise sum across all processes
    public static func allSum(
        _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_sum(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Gather arrays from all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the concatenated result.
    ///
    /// - Parameters:
    ///   - array: the local array to gather
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the concatenation of arrays from all processes
    public static func allGather(
        _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_gather(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Max-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise maximum.
    ///
    /// - Parameters:
    ///   - array: the local array to max-reduce
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise maximum across all processes
    public static func allMax(
        _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_max(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Min-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise minimum.
    ///
    /// - Parameters:
    ///   - array: the local array to min-reduce
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise minimum across all processes
    public static func allMin(
        _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_min(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Sum-reduce and scatter the array across all processes in the group.
    ///
    /// The array is sum-reduced and the result is scattered (split) across
    /// processes so each process receives its portion.
    ///
    /// - Parameters:
    ///   - array: the local array to sum-scatter
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: this process's portion of the sum-scattered result
    public static func sumScatter(
        _ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_sum_scatter(&result, array.ctx, group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Send an array to another process in the group.
    ///
    /// Returns a dependency token (an ``MLXArray``) that can be used to
    /// sequence operations.
    ///
    /// - Parameters:
    ///   - array: the array to send
    ///   - to: the destination rank
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: a dependency token
    public static func send(
        _ array: MLXArray, to dst: Int, group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_send(&result, array.ctx, Int32(dst), group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array from another process in the group.
    ///
    /// - Parameters:
    ///   - shape: the shape of the expected array
    ///   - dtype: the data type of the expected array
    ///   - from: the source rank
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array
    public static func recv(
        shape: [Int], dtype: DType, from src: Int, group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let cShape = shape.map { Int32($0) }
        mlx_distributed_recv(
            &result, cShape, cShape.count, dtype.cmlxDtype, Int32(src), group.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array from another process, using a template array for
    /// shape and dtype.
    ///
    /// - Parameters:
    ///   - array: template array whose shape and dtype define the expected result
    ///   - from: the source rank
    ///   - group: the communication group
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array with the same shape and dtype as the template
    public static func recvLike(
        _ array: MLXArray, from src: Int, group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_recv_like(&result, array.ctx, Int32(src), group.ctx, stream.ctx)
        return MLXArray(result)
    }
}
