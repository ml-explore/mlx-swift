// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Error type for synchronous distributed API failures.
///
/// Distributed collectives and layers are often lazy. These errors only
/// describe failures that can be detected at call time; execution-time backend
/// failures may still surface later when the returned value is evaluated.
public enum DistributedError: LocalizedError, Sendable, Equatable {
    case initializationFailed(backend: DistributedBackend)
    case initializationError(backend: DistributedBackend, message: String)
    case runtime(String)
    case invalidConfiguration(String)
    case unsupportedModuleType(String)

    public var errorDescription: String? {
        switch self {
        case .initializationFailed(let backend):
            "Failed to initialize a distributed group for backend '\(backend.rawValue)'."
        case .initializationError(let backend, let message):
            "Failed to initialize distributed backend '\(backend.rawValue)': \(message)"
        case .runtime(let message):
            "Distributed runtime error: \(message)"
        case .invalidConfiguration(let message):
            "Invalid distributed configuration: \(message)"
        case .unsupportedModuleType(let typeName):
            "Unsupported distributed module type: \(typeName)"
        }
    }
}

private func withDistributedRuntimeError<R>(_ body: () throws -> R) throws -> R {
    do {
        return try withError(body)
    } catch let MLXError.caught(message) {
        throw DistributedError.runtime(message)
    }
}

private func withDistributedInitializationError<R>(
    backend: DistributedBackend, _ body: () throws -> R
) throws -> R {
    do {
        return try withError(body)
    } catch let MLXError.caught(message) {
        if backend == .any, message.contains("Couldn't initialize any backend") {
            throw DistributedError.initializationFailed(backend: backend)
        }
        throw DistributedError.initializationError(backend: backend, message: message)
    }
}

private func requireDistributedGroup(
    _ group: mlx_distributed_group, operation: String
) throws -> DistributedGroup {
    guard group.ctx != nil else {
        throw DistributedError.runtime("\(operation) returned an empty distributed group.")
    }
    return DistributedGroup(group)
}

private func requireDistributedArray(_ array: mlx_array, operation: String) throws -> MLXArray {
    guard array.ctx != nil else {
        throw DistributedError.runtime("\(operation) returned an empty MLXArray.")
    }
    return MLXArray(array)
}

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
/// can communicate using distributed operations. Create the initial group with
/// ``init()``, ``init(backend:)``, or ``init(strict:)``, then use
/// ``split(color:key:)`` to create sub-groups.
///
/// `DistributedGroup()` preserves MLX's size-1 fallback behavior: if no real
/// distributed backend can be formed, MLX returns a singleton group (rank 0,
/// size 1). On that singleton group, collective operations such as `allSum`,
/// `allGather`, `allMax`, `allMin`, and `sumScatter` behave as no-ops.
///
/// `DistributedGroup` is an opaque runtime handle and is intentionally not
/// `Sendable`.
public final class DistributedGroup {

    let ctx: mlx_distributed_group

    init(_ ctx: mlx_distributed_group) {
        self.ctx = ctx
    }

    private static func initialize(strict: Bool, backend: DistributedBackend)
        -> mlx_distributed_group
    {
        backend.rawValue.withCString { mlx_distributed_init(strict, $0) }
    }

    /// Initialize the distributed backend and return the group containing all
    /// discoverable processes.
    ///
    /// When the backend cannot form a real distributed group, this initializer
    /// preserves MLX's fallback behavior and returns a singleton group (rank 0,
    /// size 1). This is equivalent to calling ``init(backend:)`` with
    /// ``DistributedBackend/any``.
    ///
    public convenience init() {
        self.init(backend: .any)
    }

    /// Initialize the distributed backend and return the group containing all
    /// discoverable processes.
    ///
    /// Unlike ``init(strict:)``, this initializer preserves MLX's fallback
    /// behavior and returns a singleton group (rank 0, size 1) when the chosen
    /// backend cannot form a real distributed group.
    ///
    /// - Parameter backend: the backend to use
    public convenience init(backend: DistributedBackend) {
        let group = Self.initialize(strict: false, backend: backend)
        precondition(
            group.ctx != nil,
            "MLX unexpectedly failed to create a distributed group for backend '\(backend.rawValue)'."
        )
        self.init(group)
    }

    /// Initialize the distributed backend and return a real distributed group.
    ///
    /// Unlike ``init(backend:)``, this initializer does not fall back to a
    /// singleton group. It succeeds only when the chosen backend can form a
    /// real distributed group at runtime, and throws when strict initialization
    /// reports a backend-specific configuration error.
    ///
    /// - Parameter backend: the backend to use
    public convenience init(strict backend: DistributedBackend) throws {
        let group = try withDistributedInitializationError(backend: backend) {
            Self.initialize(strict: true, backend: backend)
        }
        guard group.ctx != nil else {
            throw DistributedError.initializationFailed(backend: backend)
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
    /// This method throws only for failures that are detectable when the split
    /// is requested. It does not force later communication on the returned
    /// group to evaluate.
    ///
    /// - Parameters:
    ///   - color: processes with the same color go to the same sub-group
    ///   - key: determines rank ordering in the new group (negative = use current rank)
    /// - Returns: a new ``DistributedGroup`` for the sub-group
    public func split(color: Int, key: Int = -1) throws -> DistributedGroup {
        let result = try withDistributedRuntimeError {
            mlx_distributed_group_split(ctx, Int32(color), Int32(key))
        }
        return try requireDistributedGroup(result, operation: "split(color:key:)")
    }

    /// Sum-reduce the array across all processes in the group.
    ///
    /// Each process contributes its local array and all processes receive
    /// the element-wise sum.
    ///
    /// On a singleton group, this behaves as identity.
    /// This method is lazy and non-throwing: backend failures may still
    /// surface only when the returned array is evaluated. Use
    /// ``withError(_:)-6g4wn`` or ``checkedEval(_:)`` around the operation plus
    /// its evaluation boundary if you need a Swift error.
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
    /// On a singleton group, this behaves as identity.
    /// This method is lazy and non-throwing: backend failures may still
    /// surface only when the returned array is evaluated. Use
    /// ``withError(_:)-6g4wn`` or ``checkedEval(_:)`` around the operation plus
    /// its evaluation boundary if you need a Swift error.
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
    /// On a singleton group, this behaves as identity.
    /// This method is lazy and non-throwing: backend failures may still
    /// surface only when the returned array is evaluated. Use
    /// ``withError(_:)-6g4wn`` or ``checkedEval(_:)`` around the operation plus
    /// its evaluation boundary if you need a Swift error.
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
    /// On a singleton group, this behaves as identity.
    /// This method is lazy and non-throwing: backend failures may still
    /// surface only when the returned array is evaluated. Use
    /// ``withError(_:)-6g4wn`` or ``checkedEval(_:)`` around the operation plus
    /// its evaluation boundary if you need a Swift error.
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
    /// On a singleton group, this behaves as identity.
    /// This method throws only for immediate validation or setup failures such
    /// as an invalid input shape. Backend support and execution failures may
    /// still surface later when the returned array is evaluated. Wrap the
    /// operation plus its evaluation boundary in ``withError(_:)-6g4wn`` or
    /// use ``checkedEval(_:)`` when you need a Swift error.
    ///
    /// - Parameters:
    ///   - array: the local array to sum-scatter
    ///   - stream: stream or device to evaluate on
    /// - Returns: this process's portion of the sum-scattered result
    public func sumScatter(_ array: MLXArray, stream: StreamOrDevice = .default) throws -> MLXArray
    {
        var result = mlx_array_new()
        _ = try withDistributedRuntimeError {
            mlx_distributed_sum_scatter(&result, array.ctx, ctx, stream.ctx)
        }
        return try requireDistributedArray(result, operation: "sumScatter(_:stream:)")
    }

    /// Send an array to another process in the group.
    ///
    /// Returns a dependency token (an ``MLXArray``) that can be used to
    /// sequence operations.
    ///
    /// Requires a group size of at least 2.
    /// This method throws only for immediate validation or setup failures such
    /// as an invalid destination rank. Transport and backend failures may
    /// still surface later when the returned dependency token is evaluated.
    /// Wrap the operation plus its evaluation boundary in
    /// ``withError(_:)-6g4wn`` or use ``checkedEval(_:)`` when you need a
    /// Swift error.
    ///
    /// - Parameters:
    ///   - array: the array to send
    ///   - dst: the destination rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: a dependency token
    public func send(_ array: MLXArray, to dst: Int, stream: StreamOrDevice = .default) throws
        -> MLXArray
    {
        var result = mlx_array_new()
        _ = try withDistributedRuntimeError {
            mlx_distributed_send(&result, array.ctx, Int32(dst), ctx, stream.ctx)
        }
        return try requireDistributedArray(result, operation: "send(_:to:stream:)")
    }

    /// Receive an array from another process in the group.
    ///
    /// Requires a group size of at least 2.
    /// This method throws only for immediate validation or setup failures such
    /// as an invalid source rank. Transport and backend failures may still
    /// surface later when the returned array is evaluated. Wrap the operation
    /// plus its evaluation boundary in ``withError(_:)-6g4wn`` or use
    /// ``checkedEval(_:)`` when you need a Swift error.
    ///
    /// - Parameters:
    ///   - shape: the shape of the expected array
    ///   - dtype: the data type of the expected array
    ///   - src: the source rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array
    public func recv(
        shape: [Int], dtype: DType, from src: Int, stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        var result = mlx_array_new()
        let cShape = shape.map { Int32($0) }
        _ = try withDistributedRuntimeError {
            mlx_distributed_recv(
                &result, cShape, cShape.count, dtype.cmlxDtype, Int32(src), ctx, stream.ctx)
        }
        return try requireDistributedArray(result, operation: "recv(shape:dtype:from:stream:)")
    }

    /// Receive an array from another process, using a template array for
    /// shape and dtype.
    ///
    /// Requires a group size of at least 2.
    /// This method throws only for immediate validation or setup failures.
    /// Transport and backend failures may still surface later when the returned
    /// array is evaluated. Wrap the operation plus its evaluation boundary in
    /// ``withError(_:)-6g4wn`` or use ``checkedEval(_:)`` when you need a
    /// Swift error.
    ///
    /// - Parameters:
    ///   - array: template array whose shape and dtype define the expected result
    ///   - src: the source rank
    ///   - stream: stream or device to evaluate on
    /// - Returns: the received array with the same shape and dtype as the template
    public func recvLike(
        _ array: MLXArray, from src: Int, stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        var result = mlx_array_new()
        _ = try withDistributedRuntimeError {
            mlx_distributed_recv_like(&result, array.ctx, Int32(src), ctx, stream.ctx)
        }
        return try requireDistributedArray(result, operation: "recvLike(_:from:stream:)")
    }
}
