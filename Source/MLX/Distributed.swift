// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// Communication operations for running MLX across multiple processes.
///
/// This mirrors `mlx.core.distributed` in the Python API.
///
/// Forming a group can fail as soon as it is attempted, so
/// ``initialize(backend:strict:)`` and ``Group/split(color:key:)`` throw.
///
/// The collectives and the point to point operations are lazy like every
/// other MLX operation.  Invalid arguments -- e.g. sending in a group of size
/// one -- are reported when the operation is created, and communication
/// failures when its result is evaluated.  Use ``withError(_:)-2wfiu`` and
/// ``checkedEval(_:)-(Any...)`` to receive either as a Swift error:
///
/// ```swift
/// let group = try MLXDistributed.initialize(strict: true)
/// try withError {
///     let sum = MLXDistributed.allSum(x, group: group)
///     try checkedEval(sum)
/// }
/// ```
///
/// Every operation evaluates on the CPU stream by default, which is where the
/// ring, JACCL and MPI backends communicate.  NCCL, available only in a CUDA
/// build, communicates on the GPU, so pass `stream: .gpu` for an NCCL group.
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
        /// Throws if the group cannot be split.  The ring and JACCL backends do
        /// not support splitting, and neither does a group of size one.
        ///
        /// - Parameters:
        ///   - color: a value to group processes into subgroups
        ///   - key: a key to optionally change the rank ordering of the processes
        /// - Returns: the subgroup this process belongs to
        public func split(color: Int, key: Int = -1) throws -> Group {
            var result = mlx_distributed_group_new()
            do {
                try withError {
                    _ = mlx_distributed_group_split(&result, ctx, Int32(color), Int32(key))
                }
            } catch {
                mlx_distributed_group_free(result)
                throw error
            }
            return Group(result)
        }
    }

    /// Initialize the distributed backend and return the global group.
    ///
    /// If no backend is configured and `strict` is `false` this returns a
    /// group of size one, in which the collectives return their input
    /// unchanged, so the same code also runs as a single process.  Forming a
    /// real group waits for the other processes to connect.
    ///
    /// Throws if the backend cannot be initialized: when `strict` is `true` and
    /// no backend is configured, and -- even when `strict` is `false` -- when a
    /// backend is configured but cannot form its group, e.g. the ring backend
    /// with a malformed `MLX_HOSTFILE`.
    ///
    /// MLX caches the group for each backend for the life of the process,
    /// including the group of size one that a non-strict call falls back to,
    /// and a later strict call for that backend returns the cached group
    /// instead of throwing.  A non-strict ``Backend/any`` that finds no backend
    /// caches its fallback under ``Backend/jaccl``.  Initialize strictly
    /// before anything else initializes the backend.
    ///
    /// - Parameters:
    ///   - backend: the backend to use, defaulting to ``Backend/any``
    ///   - strict: if `true`, throw rather than fall back to a group of size one
    ///     when no backend can be initialized
    /// - Returns: the global group
    @discardableResult
    public static func initialize(backend: Backend = .any, strict: Bool = false) throws -> Group {
        var result = mlx_distributed_group_new()
        do {
            try withError {
                _ = mlx_distributed_init(&result, strict, backend.rawValue)
            }
        } catch {
            mlx_distributed_group_free(result)
            throw error
        }
        return Group(result)
    }

    // MARK: - Collectives

    // Note: mlx-c has no representation for an "unspecified" stream --
    // mlx_stream_get_() rejects a null handle -- so the backend's own default
    // (Group::communication_stream) is unreachable from Swift.  Python passes
    // stream=None and lets ring/mpi/jaccl pick the CPU and nccl pick the GPU.
    // The ring and JACCL backends both want the CPU, and the collectives have
    // no GPU implementation, so the CPU stream is the default here.  A CUDA
    // build with nccl has to pass .gpu, which MLXDistributed documents.

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
