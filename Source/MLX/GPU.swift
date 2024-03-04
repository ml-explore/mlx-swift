// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Properties to control the the GPU memory allocation and buffer reuse.
public enum GPU {

    /// Relaxed parameter for ``memoryLimit``.
    public static var relaxedMemoryLimit = true {
        didSet {
            mlx_metal_set_memory_limit(self.memoryLimit, relaxedMemoryLimit)
        }
    }

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

    /// Get or set the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation. To disable the cache,
    /// set the limit to 0.
    ///
    /// The cache limit defaults to the memory limit.
    ///
    /// Returns the previous cache limit.
    public static var cacheLimit: Int {
        get {
            if let cacheLimit = _cacheLimit {
                return cacheLimit
            }

            // set it to a reasonable value in order to read it, then set it back
            // to current
            let current = mlx_metal_set_cache_limit(cacheMemory)
            mlx_metal_set_cache_limit(current)
            self.cacheLimit = current
            return current
        }
        set {
            mlx_metal_set_cache_limit(newValue)
        }
    }

    /// Get or set the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded.  If
    /// there are no more scheduled tasks an error will be raised if ``relaxedMemoryLimit``
    /// is false or memory will be allocated (including the potential for
    /// swap) if `relaxedMemoryLimit` is true.
    ///
    /// The memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device.
    ///
    /// ### See Also
    /// - ``relaxedMemoryLimit``
    public static var memoryLimit: Int {
        get {
            if let memoryLimit = _memoryLimit {
                return memoryLimit
            }

            let current = mlx_metal_set_memory_limit(activeMemory, relaxedMemoryLimit)
            mlx_metal_set_memory_limit(current, relaxedMemoryLimit)
            return current
        }
        set {
            mlx_metal_set_memory_limit(newValue, relaxedMemoryLimit)
        }
    }
}
