// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Properties to control the the memory allocation and buffer reuse.
///
/// ``activeMemory`` + ``cacheMemory`` is the total memory allocated by MLX.
/// ``activeMemory`` is in currently active ``MLXArray`` and ``cacheMemory``
/// is recently used memory that can be recycled.
///
/// ## Memory Management and Buffer Recycling
///
/// MLX uses a buffer recycling system to optimize performance. When MLXArrays
/// are no longer needed (such as intermediate computation results), their buffers
/// are not immediately deallocated. Instead, they are placed in a buffer pool where they
/// can be reused by later computations of similar size.
///
/// This recycling strategy is particularly important during model inference:
///
/// - Initial model weights might use ~500MB
/// - Inference (e.g. token generation in an LLM) creates intermediate buffers (e.g., ~1MB for the first token)
/// - These buffer might be intermediates used during computation of the graph -- if they
/// are fixed size then they will be reused on the next token -- exactly what we want!
/// - If the buffer sizes grow during inference, e.g. if not using a KVCache in LLMs
/// or other scenarios where there might be a "context" then these buffers might
/// not be reused
/// - By the end of a long inference run, you may see several GB of cached memory
///   from accumulated buffers of various sizes if cache memory is unconstrained.
///
/// This memory is observable in various system tools as _footprint_ or _RSIZE_, but
/// these tools can't discern the use of the memory.  See ``snapshot()`` and
/// the next section for more information.
///
/// The buffer pool policy is based on Metal's `recommendedMaxWorkingSetSize`, which
/// scales with available physical memory. Systems with more RAM will cache more buffers.
///
/// ## Cache Size Optimization
///
/// The optimal cache size varies significantly by workload. While unconstrained cache
/// can grow to several GB, developers often find that relatively small cache sizes
/// (e.g., 2MB) perform just as well for their specific use cases. The best approach
/// is to experiment with different cache limits and measure performance for your
/// particular workload.
///
/// Adjusting the cache limit is especially advantageous on devices with memory
/// limits (e.g. iOS devices where jetsam limits apply).
///
/// Control the size of cache memory via ``Memory/cacheLimit``
/// and the overall memory limit with ``Memory/memoryLimit``.
///
/// Examine memory use over time with ``snapshot()`` and ``Snapshot``.
///
/// **Note**: The cache limit will go into effect on the next deallocation. Because of that you
/// may observe the cache size temporarily exceeding the requested limit. To immediately
/// clear the cache, use ``Memory/clearCache()``.
///
/// ### See Also
/// - <doc:running-on-ios>
/// - ``cacheLimit``
/// - ``memoryLimit``
/// - ``snapshot()``
public enum Memory {

    static let queue = DispatchQueue(label: "GPUEnum")

    // note: these are guarded by the queue above
    #if swift(>=5.10)
        nonisolated(unsafe) static var _cacheLimit: Int?
        nonisolated(unsafe) static var _memoryLimit: Int?
    #else
        static var _cacheLimit: Int?
        static var _memoryLimit: Int?
    #endif

    /// Snapshot of memory stats.
    ///
    /// ``activeMemory`` + ``cacheMemory`` is the total memory allocated by MLX.
    /// ``activeMemory`` is in currently active ``MLXArray`` and ``cacheMemory``
    /// is recently used memory that can be recycled.
    ///
    /// See ``GPU`` for a description of how the cache sizes grow and can be tuned.
    ///
    /// Control the size of cache memory via ``Memory/cacheLimit``
    /// and the overall memory limit with ``Memory/memoryLimit``.
    ///
    /// This might be used to examine memory use over a run or sample it during a run:
    ///
    /// ```swift
    /// // load model & weights
    /// ...
    ///
    /// let startMemory = Memory.snapshot()
    ///
    /// // work
    /// ...
    ///
    /// let endMemory = Memory.snapshot()
    ///
    /// // what stats are interesting to you?
    ///
    /// print("=======")
    /// print("Memory size: \(Memory.memoryLimit / 1024)K")
    /// print("Cache size:  \(Memory.cacheLimit / 1024)K")
    ///
    /// print("")
    /// print("=======")
    /// print("Starting memory")
    /// print(startMemory.description)
    ///
    /// print("")
    /// print("=======")
    /// print("Ending memory")
    /// print(endMemory.description)
    ///
    /// print("")
    /// print("=======")
    /// print("Growth")
    /// print(startMemory.delta(endMemory).description)
    /// ```
    ///
    /// ### See Also
    /// - ``snapshot()``
    /// - <doc:running-on-ios>
    public struct Snapshot: CustomStringConvertible, Codable, Sendable {

        /// See ``Memory/activeMemory``.
        public var activeMemory: Int

        /// See ``Memory/cacheMemory``.
        public var cacheMemory: Int

        /// See ``Memory/peakMemory``.
        public var peakMemory: Int

        /// Compute the difference between two snapshots:
        ///
        /// ```swift
        /// let startMemory = Memory.snapshot()
        /// ...
        /// let endMemory = Memory.snapshot()
        /// print(startMemory.delta(endMemory))
        /// ```
        public func delta(_ other: Snapshot) -> Snapshot {
            .init(
                activeMemory: other.activeMemory - activeMemory,
                cacheMemory: other.cacheMemory - cacheMemory,
                peakMemory: other.peakMemory - peakMemory)
        }

        public var description: String {
            func scale(_ value: Int, width: Int = 12) -> String {
                let v: String
                if value > 1024 * 1024 * 10 {
                    v = "\(value / (1024 * 1024))M"
                } else {
                    v = "\(value / 1024)K"
                }
                let pad = String(repeating: " ", count: max(0, width - v.count))
                return v + pad
            }

            return """
                Peak:   \(scale(peakMemory)) (\(peakMemory))
                Active: \(scale(activeMemory)) (\(activeMemory))
                Cache:  \(scale(cacheMemory)) (\(cacheMemory))
                """
        }
    }

    /// Get the actively used memory in bytes.
    ///
    /// Note, this will not always match memory use reported by the system because
    /// it does not include cached memory buffers.
    public static var activeMemory: Int {
        var result: size_t = 0
        mlx_get_active_memory(&result)
        return result
    }

    /// Get the cache size in bytes.
    ///
    /// The cache includes memory not currently used that has not been returned
    /// to the system allocator. This represents buffers from previous
    /// computations that are kept in a buffer pool for potential reuse.
    ///
    /// During model inference, this can grow significantly as buffers of various
    /// sizes accumulate from intermediate computations.  See ``GPU`` for
    /// more information on cache size and tuning.
    ///
    /// The cache size is controlled by the cache limit (see ``cacheLimit``).
    /// When the limit is exceeded, older cached buffers are freed on the next allocation.
    public static var cacheMemory: Int {
        var result: size_t = 0
        mlx_get_cache_memory(&result)
        return result
    }

    /// Get the peak amount of active memory in bytes.
    ///
    /// The maximum memory used is recorded from the beginning of the program
    /// execution.
    public static var peakMemory: Int {
        get {
            var result: size_t = 0
            mlx_get_peak_memory(&result)
            return result
        }
        set {
            // note: ignores newValue
            mlx_reset_peak_memory()
        }
    }

    /// Return a snapshot of memory stats -- see ``Snapshot`` for more details.
    ///
    /// Get the current memory use.  This can be used to measure before/after and current memory use:
    ///
    /// ```swift
    /// let currentMemory = Memory.snapshot()
    /// print(currentMemory)
    /// ```
    public static func snapshot() -> Snapshot {
        Snapshot(activeMemory: activeMemory, cacheMemory: cacheMemory, peakMemory: peakMemory)
    }

    /// Get or set the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation.
    ///
    /// The cache limit defaults to the memory limit, which may allow very
    /// large cache sizes on systems with abundant RAM. For memory-constrained
    /// applications or to prevent excessive memory growth during long inference
    /// runs, consider setting a much lower cache limit.
    ///
    /// To disable the cache, set the limit to 0.
    ///
    /// ## Performance Optimization
    ///
    /// The optimal cache size varies by workload. Many developers find that
    /// relatively small cache sizes (e.g., 2MB) perform just as well as
    /// unconstrained cache sizes for their specific use cases. Experiment
    /// with different values and measure performance to find the best setting
    /// for your workload.
    ///
    /// See ``Memory`` for more information on cache sizing and tuning.
    ///
    /// ### See Also
    /// - ``memoryLimit``
    public static var cacheLimit: Int {
        get {
            queue.sync {
                if let cacheLimit = _cacheLimit {
                    return cacheLimit
                }

                // set it to a reasonable value in order to read it, then set it back
                // to current
                var current: size_t = 0
                var discard: size_t = 0
                mlx_set_cache_limit(&current, cacheMemory)
                mlx_set_cache_limit(&discard, current)

                _cacheLimit = current
                return current
            }
        }
        set {
            queue.sync {
                _cacheLimit = newValue
                var current: size_t = 0
                mlx_set_cache_limit(&current, newValue)
            }
        }
    }

    /// Get or set the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded. The
    /// memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded.
    ///
    /// Note: `cacheLimit` may be more convenient to manipulate.
    ///
    /// ### See Also
    /// - ``cacheLimit``
    public static var memoryLimit: Int {
        get {
            queue.sync {
                var current: size_t = 0
                mlx_get_memory_limit(&current)
                return Int(current)
            }
        }
        set {
            queue.sync {
                _memoryLimit = newValue
                var current: size_t = 0
                mlx_set_memory_limit(&current, newValue)
            }
        }
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// - Important: This synchronous overload is deprecated and is now a no-op.
    ///   Use ``WiredMemoryManager`` with tickets instead.
    ///
    /// - Parameters:
    ///   - limit: requested limit in bytes (ignored)
    ///   - body: block to perform
    @available(
        *, deprecated,
        message: "Deprecated. Use WiredMemoryManager and tickets; synchronous variant is a no-op."
    )
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () throws -> R
    ) rethrows -> R {
        _ = limit
        return try body()
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// - Important: This overload is deprecated and now uses the shared
    ///   ``WiredMemoryManager`` with a static sum policy to avoid bypassing
    ///   admission control.
    ///
    /// - Parameters:
    ///   - limit: requested limit in bytes
    ///   - body: block to perform
    @available(
        *, deprecated,
        message:
            "Deprecated. Use WiredMemoryManager and tickets; async variant uses the shared manager."
    )
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () async throws -> R
    ) async rethrows -> R {
        let ticket = WiredMemoryTicket(
            size: max(0, limit),
            policy: Memory.wiredLimitPolicy,
            manager: .shared,
            kind: .active
        )
        return try await ticket.withWiredLimit(body)
    }

    private static let wiredLimitPolicy = WiredSumPolicy(
        id: UUID(uuidString: "B8C3B7E9-1B2E-4C3A-8E3D-1C7F2B8A9D10")!)

    /// Cause all cached buffers to be deallocated.
    public static func clearCache() {
        _ = evalLock.withLock {
            mlx_clear_cache()
        }
    }
}
