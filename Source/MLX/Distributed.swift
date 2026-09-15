// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// Communication operations for running MLX across multiple processes.
///
/// This mirrors `mlx.core.distributed` in the Python API.
public enum MLXDistributed {

    /// A communication backend.
    ///
    /// Use ``isAvailable(_:)`` to check whether MLX was built with support for
    /// a given backend.  Note that availability means MLX can instantiate the
    /// backend, not that a communication group can actually be formed.
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
}
