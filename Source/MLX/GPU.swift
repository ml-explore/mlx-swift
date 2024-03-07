// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Properties to control the the GPU memory allocation and buffer reuse.
///
/// ``activeMemory`` + ``cacheMemory`` is the total memory allocated by MLX.
/// ``activeMemory`` is in currently active ``MLXArray`` and ``cacheMemory``
/// is recently used memory that can be recycled.
///
/// Control the size of ``cacheMemory`` via ``GPU/set(cacheLimit:)``
/// and the overall memory limit with ``GPU/set(memoryLimit:relaxed:)``.
///
/// Examine memory use over time with ``snapshot()`` and ``Snapshot``.
///
/// ### See Also
/// - <doc:running-on-ios>
/// - ``set(cacheLimit:)``
/// - ``set(memoryLimit:relaxed:)``
/// - ``snapshot()``
public enum GPU {

    static var _relaxedMemoryLimit = true
    static var _cacheLimit: Int?
    static var _memoryLimit: Int?

    /// Snapshot of memory stats.
    ///
    /// ``activeMemory`` + ``cacheMemory`` is the total memory allocated by MLX.
    /// ``activeMemory`` is in currently active ``MLXArray`` and ``cacheMemory``
    /// is recently used memory that can be recycled.
    ///
    /// Control the size of ``cacheMemory`` via ``GPU/set(cacheLimit:)``
    /// and the overall memory limit with ``GPU/set(memoryLimit:relaxed:)``.
    ///
    /// This might be used to eamine memory use over a run or sample it during a run:
    ///
    /// ```swift
    /// // load model & weights
    /// ...
    ///
    /// let startMemory = GPU.snapshot()
    ///
    /// // work
    /// ...
    ///
    /// let endMemory = GPU.snapshot()
    ///
    /// // what stats are interesting to you?
    ///
    /// print("=======")
    /// print("Memory size: \(GPU.memoryLimit / 1024)K")
    /// print("Cache size:  \(GPU.cacheLimit / 1024)K")
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
    public struct Snapshot: CustomStringConvertible, Codable {

        /// See ``GPU/activeMemory``.
        public var activeMemory: Int

        /// See ``GPU/cacheMemory``.
        public var cacheMemory: Int

        /// See ``GPU/peakMemory``.
        public var peakMemory: Int

        /// Compute the difference between two snapshots:
        ///
        /// ```swift
        /// let startMemory = GPU.snapshot()
        /// ...
        /// let endMemory = GPU.snapshot()
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
        mlx_metal_get_active_memory()
    }

    /// Get the cache size in bytes.
    ///
    /// The cache includes memory not currently used that has not been returned
    /// to the system allocator.
    public static var cacheMemory: Int {
        mlx_metal_get_cache_memory()
    }

    /// Get the peak amount of active memory in bytes.
    ///
    /// The maximum memory used is recorded from the beginning of the program
    /// execution.
    public static var peakMemory: Int {
        mlx_metal_get_peak_memory()
    }

    /// Return a snapshot of memory stats -- see ``Snapshot`` for more details.
    ///
    /// Get the current memory use.  This can be used to measure before/after and current memory use:
    ///
    /// ```swift
    /// let currentMemory = GPU.snapshot()
    /// print(currentMemory)
    /// ```
    public static func snapshot() -> Snapshot {
        Snapshot(activeMemory: activeMemory, cacheMemory: cacheMemory, peakMemory: peakMemory)
    }

    /// Get the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation.
    /// The cache limit defaults to the memory limit.
    ///
    /// ### See Also
    /// - ``set(cacheLimit:)``
    public static var cacheLimit: Int {
        if let cacheLimit = _cacheLimit {
            return cacheLimit
        }

        // set it to a reasonable value in order to read it, then set it back
        // to current
        let current = mlx_metal_set_cache_limit(cacheMemory)
        mlx_metal_set_cache_limit(current)
        _cacheLimit = current
        return current
    }

    /// Set the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation. To disable the cache,
    /// set the limit to 0.
    ///
    /// The cache limit defaults to the memory limit.
    ///
    /// Returns the previous cache limit.
    public static func set(cacheLimit: Int) {
        _cacheLimit = cacheLimit
        mlx_metal_set_cache_limit(cacheLimit)
    }

    /// Get the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded. The
    /// memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device.
    ///
    /// ### See Also
    /// - ``set(memoryLimit:relaxed:)``
    public static var memoryLimit: Int {
        if let memoryLimit = _memoryLimit {
            return memoryLimit
        }

        let current = mlx_metal_set_memory_limit(activeMemory, _relaxedMemoryLimit)
        mlx_metal_set_memory_limit(current, _relaxedMemoryLimit)
        return current
    }

    /// Set the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded.  If
    /// there are no more scheduled tasks an error will be raised if `relaxed`
    /// is false or memory will be allocated (including the potential for
    /// swap) if `relaxed` is true.
    ///
    /// The memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device ([recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/2369280-recommendedmaxworkingsetsize))
    public static func set(memoryLimit: Int, relaxed: Bool = true) {
        _relaxedMemoryLimit = relaxed
        _memoryLimit = memoryLimit
        mlx_metal_set_memory_limit(memoryLimit, relaxed)
    }
}
