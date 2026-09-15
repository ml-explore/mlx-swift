// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// Communication operations for running MLX across multiple processes.
///
/// This mirrors `mlx.core.distributed` in the Python API.
///
/// Like the rest of the MLX Swift API these functions do not `throw`.  Errors
/// raised by MLX -- e.g. sending in a group of size one -- are reported through
/// the standard error handling machinery:
///
/// ```swift
/// let group = try withError { MLXDistributed.initialize(strict: true) }
/// ```
///
/// ### See Also
/// - ``withError(_:)-2wfiu``
/// - ``checkedEval(_:)-(Any...)``
public enum MLXDistributed {

    /// A communication backend.
    ///
    /// Use ``isAvailable(_:)`` to check whether MLX was built with support for
    /// a given backend.  Note that availability means MLX can instantiate the
    /// backend, not that a communication group can actually be formed -- use
    /// ``initialize(backend:strict:)`` with `strict: true` for that.
    public enum Backend: String, CaseIterable, Sendable {
        /// Select the first available backend.
        case any
        case ring
        case mpi
        case nccl
        case jaccl
    }

    /// Check if a communication backend is available.
    ///
    /// - Parameter backend: the backend to check for, defaulting to ``Backend/any``
    /// - Returns: whether MLX has the capability of instantiating that backend
    public static func isAvailable(_ backend: Backend = .any) -> Bool {
        mlx_distributed_is_available(backend.rawValue)
    }

    /// A group of independent MLX processes that can communicate.
    ///
    /// Obtain a group with ``initialize(backend:strict:)``.
    public final class Group: @unchecked Sendable {

        let ctx: mlx_distributed_group

        init(_ ctx: mlx_distributed_group) {
            self.ctx = ctx
        }

        deinit {
            mlx_distributed_group_free(ctx)
        }

        /// The rank of this process in the group.
        public var rank: Int {
            Int(mlx_distributed_group_rank(ctx))
        }

        /// The number of processes in the group.
        public var size: Int {
            Int(mlx_distributed_group_size(ctx))
        }

        /// Split the group into subgroups based on the provided color.
        ///
        /// Processes that use the same color go to the same group.  The `key`
        /// argument defines the rank in the new group -- the smaller the key
        /// the smaller the rank.  If the key is negative the rank in the
        /// current group is used.
        ///
        /// - Parameters:
        ///   - color: a value to group processes into subgroups
        ///   - key: a key to optionally change the rank ordering of the processes
        /// Returns `nil` if the group cannot be split -- an empty group, for
        /// example, cannot be split further.
        public func split(color: Int, key: Int = -1) -> Group? {
            var result = mlx_distributed_group_new()
            mlx_distributed_group_split(&result, ctx, Int32(color), Int32(key))
            guard result.ctx != nil else { return nil }
            return Group(result)
        }
    }

    /// Initialize the distributed backend and return the global group.
    ///
    /// Repeated calls return the same group for a given backend.  If no
    /// backend can be initialized and `strict` is `false` this returns an
    /// empty group with `rank == 0` and `size == 1`.
    ///
    /// - Parameters:
    ///   - backend: the backend to use, defaulting to ``Backend/any``
    ///   - strict: if `true` report an error when no backend can be initialized
    /// Returns `nil` if the backend could not be initialized, which happens
    /// when `strict` is `true` and no backend can form a group.  A non-strict
    /// initialize yields an empty group of size one rather than `nil`.
    @discardableResult
    public static func initialize(backend: Backend = .any, strict: Bool = false) -> Group? {
        var result = mlx_distributed_group_new()
        mlx_distributed_init(&result, strict, backend.rawValue)
        guard result.ctx != nil else { return nil }
        return Group(result)
    }

    /// The global group.
    ///
    /// Equivalent to ``initialize(backend:strict:)`` without `strict`, which
    /// yields an empty group of size one when no backend is available.  Traps
    /// in the event MLX cannot produce a group at all.
    public static var globalGroup: Group {
        guard let group = initialize() else {
            preconditionFailure("MLX could not create a distributed group")
        }
        return group
    }

    // MARK: - Collectives

    // Note: mlx-c has no representation for an "unspecified" stream --
    // mlx_stream_get_() rejects a null handle -- so the backend's own default
    // (Group::communication_stream) is unreachable from Swift.  Python passes
    // stream=None and lets ring/mpi/jaccl pick the CPU and nccl pick the GPU.
    // The ring and JACCL backends both want the CPU, and the collectives have
    // no GPU implementation, so the CPU stream is the default here.

    /// All reduce sum.
    ///
    /// Sum `x` across all processes in the group.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func allSum(
        _ x: MLXArray, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_sum(&result, x.ctx, group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// All reduce max.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func allMax(
        _ x: MLXArray, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_max(&result, x.ctx, group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// All reduce min.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func allMin(
        _ x: MLXArray, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_min(&result, x.ctx, group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// Gather arrays from all processes, concatenating along the first axis.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func allGather(
        _ x: MLXArray, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_all_gather(&result, x.ctx, group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// Sum `x` across the group and shard the result along the first axis.
    ///
    /// `x.dim(0)` must be divisible by the group size.  The result is
    /// equivalent to `allSum(x)` sliced to this process' chunk, but is
    /// performed as a single reduce-scatter collective.
    ///
    /// Currently implemented by the NCCL and JACCL backends only.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func sumScatter(
        _ x: MLXArray, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_sum_scatter(&result, x.ctx, group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    // MARK: - Point to point

    /// Send `x` to the process with rank `dst`.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - dst: rank of the destination process
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    /// - Returns: an array identical to `x` which, when evaluated, performs the send
    public static func send(
        _ x: MLXArray, to dst: Int, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_send(&result, x.ctx, Int32(dst), group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array with the given shape and type from the process with rank `src`.
    ///
    /// - Parameters:
    ///   - shape: shape of the array to receive
    ///   - dtype: type of the array to receive
    ///   - src: rank of the source process
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func recv(
        _ shape: [Int], dtype: DType, from src: Int, group: Group? = nil,
        stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        let shape = shape.asInt32
        mlx_distributed_recv(
            &result, shape, shape.count, dtype.cmlxDtype, Int32(src), group.groupCtx, stream.ctx)
        return MLXArray(result)
    }

    /// Receive an array with the same shape and type as `x` from the process with rank `src`.
    ///
    /// - Parameters:
    ///   - x: array whose shape and type describe what will be received
    ///   - src: rank of the source process
    ///   - group: the group, or `nil` to use the global group
    ///   - stream: stream to evaluate on
    public static func recvLike(
        _ x: MLXArray, from src: Int, group: Group? = nil, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_distributed_recv_like(&result, x.ctx, Int32(src), group.groupCtx, stream.ctx)
        return MLXArray(result)
    }
}

extension Optional where Wrapped == MLXDistributed.Group {
    /// The underlying group handle, or a null handle meaning "the global group".
    fileprivate var groupCtx: mlx_distributed_group {
        self?.ctx ?? mlx_distributed_group(ctx: nil)
    }
}
