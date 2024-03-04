// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Properties to control the the GPU memory allocation and buffer reuse.
public enum GPU {

    static var _relaxedMemoryLimit = true
    static var _cacheLimit: Int?
    static var _memoryLimit: Int?

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

    /// Get the peak amount of used memory in bytes.
    ///
    /// The maximum memory used is recorded from the beginning of the program
    /// execution.
    public static var peakMemory: Int {
        mlx_metal_get_peak_memory()
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
    /// size reported by the device.
    ///
    /// ### See Also
    /// - ``relaxedMemoryLimit``
    public static func set(memoryLimit: Int, relaxed: Bool = true) {
        _relaxedMemoryLimit = relaxed
        _memoryLimit = memoryLimit
        mlx_metal_set_memory_limit(memoryLimit, relaxed)
    }
}
