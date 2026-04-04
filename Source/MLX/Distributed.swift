// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// The distributed communication backend to use.
///
/// When ``DistributedBackend/any`` is specified, MLX chooses the best available
/// backend automatically. Use a specific case to force a particular backend.
public enum DistributedBackend: String, CaseIterable, Sendable {
    /// Let MLX choose the best available backend automatically.
    case any
    /// TCP socket-based ring backend.
    case ring
    /// Joint Accelerator Communication Library (Thunderbolt 5 RDMA).
    case jaccl
    /// Message Passing Interface backend.
    case mpi
    /// NVIDIA Collective Communications Library backend.
    case nccl

    /// Whether this backend can be initialized on the current runtime.
    public var isAvailable: Bool {
        rawValue.withCString { mlx_distributed_is_available($0) }
    }
}

/// Wrapper around the MLX C distributed group handle.
///
/// A `DistributedGroup` represents a group of independent MLX processes that
/// can communicate using collective operations. Create the initial group with
/// ``init(backend:)`` or ``init(strict:)``, then use ``split(color:key:)`` to
/// create sub-groups.
///
/// `DistributedGroup()` preserves MLX's size-1 fallback behavior: if no real
/// distributed backend can be formed, MLX returns a singleton group whose
/// collective operations become no-ops.
public final class DistributedGroup: @unchecked Sendable {

    let ctx: mlx_distributed_group

    init(_ ctx: mlx_distributed_group) {
        self.ctx = ctx
    }

    private static func initialize(strict: Bool, backend: DistributedBackend) -> mlx_distributed_group
    {
        backend.rawValue.withCString { mlx_distributed_init(strict, $0) }
    }

    /// Initialize the distributed backend and return the group containing all
    /// discoverable processes.
    ///
    /// When the backend cannot form a real distributed group, this initializer
    /// preserves MLX's fallback behavior and returns a singleton group (rank 0,
    /// size 1).
    ///
    /// - Parameter backend: the backend to use (default: `.any`, let MLX choose)
    public convenience init(backend: DistributedBackend = .any) {
        let group = Self.initialize(strict: false, backend: backend)
        precondition(
            group.ctx != nil,
            "MLX unexpectedly failed to create a distributed group for backend '\(backend.rawValue)'."
        )
        self.init(group)
    }

    /// Initialize the distributed backend and return `nil` when no real
    /// distributed group can be formed.
    ///
    /// Unlike ``init(backend:)``, this initializer does not fall back to a
    /// singleton group.
    ///
    /// - Parameter backend: the backend to use (default: `.any`, let MLX choose)
    public convenience init?(strict backend: DistributedBackend = .any) {
        let group = Self.initialize(strict: true, backend: backend)
        guard group.ctx != nil else {
            return nil
        }
        self.init(group)
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
    /// The key defines the rank of the process in the new group; the smaller
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

    /// Sum-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise sum.
    ///
    /// - Parameters:
    ///   - array: the local array to sum
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise sum across all processes
    public func allSum(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_sum(&result, array.ctx, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Gather arrays from all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the concatenated result.
    ///
    /// - Parameters:
    ///   - array: the local array to gather
    ///   - stream: stream or device to evaluate on
    /// - Returns: the concatenation of arrays from all processes
    public func allGather(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_gather(&result, array.ctx, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Max-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise maximum.
    ///
    /// - Parameters:
    ///   - array: the local array to max-reduce
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise maximum across all processes
    public func allMax(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_max(&result, array.ctx, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Min-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise minimum.
    ///
    /// - Parameters:
    ///   - array: the local array to min-reduce
    ///   - stream: stream or device to evaluate on
    /// - Returns: the element-wise minimum across all processes
    public func allMin(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_min(&result, array.ctx, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Sum-reduce and scatter the array across all processes in the group.
    ///
    /// The array is sum-reduced and the result is scattered (split) across
    /// processes so each process receives its portion.
    ///
    /// - Parameters:
    ///   - array: the local array to sum-scatter
    ///   - stream: stream or device to evaluate on
    /// - Returns: this process's portion of the sum-scattered result
    public func sumScatter(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_sum_scatter(&result, array.ctx, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Send an array to another process in the group.
    ///
    /// Returns a dependency token (an ``MLXArray``) that can be used to
    /// sequence operations.
    ///
    /// - Parameters:
    ///   - array: the array to send
    ///   - dst: the destination rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: a dependency token
    public func send(_ array: MLXArray, to dst: Int, stream: StreamOrDevice = .default) -> MLXArray
    {
        var result = mlx_array_new()
        mlx_distributed_send(&result, array.ctx, Int32(dst), ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array from another process in the group.
    ///
    /// - Parameters:
    ///   - shape: the shape of the expected array
    ///   - dtype: the data type of the expected array
    ///   - src: the source rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array
    public func recv(
        shape: [Int], dtype: DType, from src: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let cShape = shape.map { Int32($0) }
        mlx_distributed_recv(
            &result, cShape, cShape.count, dtype.cmlxDtype, Int32(src), ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array from another process, using a template array for
    /// shape and dtype.
    ///
    /// - Parameters:
    ///   - array: template array whose shape and dtype define the expected result
    ///   - src: the source rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array with the same shape and dtype as the template
    public func recvLike(
        _ array: MLXArray, from src: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_recv_like(&result, array.ctx, Int32(src), ctx, stream.ctx)
        return MLXArray(result)
    }
}
