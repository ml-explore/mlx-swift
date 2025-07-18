// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Metal

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
    public struct Snapshot: CustomStringConvertible, Codable, Sendable {

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
        var result: size_t = 0
        mlx_get_active_memory(&result)
        return result
    }

    /// Get the cache size in bytes.
    ///
    /// The cache includes memory not currently used that has not been returned
    /// to the system allocator.
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
        var result: size_t = 0
        mlx_get_peak_memory(&result)
        return result
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
        queue.sync {
            _cacheLimit = cacheLimit
            var current: size_t = 0
            mlx_set_cache_limit(&current, cacheLimit)
        }
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
        queue.sync {
            var current: size_t = 0
            mlx_get_memory_limit(&current)
            return Int(current)
        }
    }

    /// Set the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded.  If
    /// there are no more scheduled tasks an error will be raised if `relaxed`
    /// is false or memory will be allocated (including the potential for
    /// swap) if `relaxed` is true.
    ///
    /// The memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device ([recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize))
    public static func set(memoryLimit: Int, relaxed: Bool = true) {
        queue.sync {
            _memoryLimit = memoryLimit
            var current: size_t = 0
            mlx_set_memory_limit(&current, memoryLimit)
        }
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// Note: this manipulates a global value.  Nested calls will work as expected but
    /// concurrent calls cannot.
    ///
    /// See also ``DeviceInfo/maxRecommendedWorkingSetSize``.
    ///
    /// - Parameters:
    ///   - limit: new limit in bytes
    ///   - body: block to perform
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () throws -> R
    ) rethrows -> R {
        var current = 0
        mlx_set_wired_limit(&current, limit)
        defer {
            var tmp = 0
            mlx_set_wired_limit(&tmp, current)
        }

        return try body()
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// Note: this manipulates a global value.  Nested calls will work as expected but
    /// concurrent calls cannot.
    ///
    /// See also ``DeviceInfo/maxRecommendedWorkingSetSize``.
    ///
    /// - Parameters:
    ///   - limit: new limit in bytes
    ///   - body: block to perform
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () async throws -> R
    ) async rethrows -> R {
        var current = 0
        mlx_set_wired_limit(&current, limit)
        defer {
            var tmp = 0
            mlx_set_wired_limit(&tmp, current)
        }

        return try await body()
    }

    /// Cause all cached metal buffers to be deallocated.
    public static func clearCache() {
        mlx_clear_cache()
    }

    /// Start capturing a metal trace into the given file.
    ///
    /// > There are several requirements for this to be used.
    ///
    /// - `mlx` must be built with `MLX_METAL_DEBUG`
    ///   - in Package.swift add `.define("MLX_METAL_DEBUG"),` to `Cmlx` `cxxSettings`
    /// - when running set the `MTL_CAPTURE_ENABLED=1` environment variable
    /// - make sure the file at the given path does not already exist
    ///
    /// See [the documentation](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
    /// for more information.
    public static func startCapture(url: URL) {
        mlx_metal_start_capture(url.path().cString(using: .utf8))
    }

    /// Stop the metal capture.
    ///
    /// See ``startCapture(url:)``.
    public static func stopCapture(url: URL) {
        mlx_metal_stop_capture()
    }

    /// Reset the peak memory to zero.
    ///
    /// See ``Snapshot/peakMemory``.
    public static func resetPeakMemory() {
        mlx_reset_peak_memory()
    }

    public struct DeviceInfo: Sendable {
        public let architecture: String
        public let maxBufferSize: Int
        public let maxRecommendedWorkingSetSize: UInt64
        public let memorySize: Int
    }

    /// Get information about the GPU device and system settings
    public static func deviceInfo() -> DeviceInfo {
        var mib = [CTL_HW, HW_MEMSIZE]
        var memSize: size_t = 0
        var length: size_t = MemoryLayout.size(ofValue: memSize)
        sysctl(&mib, 2, &memSize, &length, nil, 0)

        if let device = MTLCreateSystemDefaultDevice() {
            let architecture: String
            if #available(macOS 14.0, iOS 17.0, tvOS 17.0, *) {
                architecture = device.architecture.name
            } else {
                architecture = device.name
            }

            return DeviceInfo(
                architecture: architecture, maxBufferSize: device.maxBufferLength,
                maxRecommendedWorkingSetSize: device.recommendedMaxWorkingSetSize,
                memorySize: memSize)
        } else {
            return DeviceInfo(
                architecture: "Unknown", maxBufferSize: 0, maxRecommendedWorkingSetSize: 0,
                memorySize: memSize)
        }
    }
}
