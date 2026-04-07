// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// MARK: - Internal Ops

/// Broadcast a vector of arrays against one another.
func broadcast(arrays: some Collection<MLXArray>, stream: StreamOrDevice = .default) -> [MLXArray] {
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(vector_array) }

    var result = mlx_vector_array_new()
    mlx_broadcast_arrays(&result, vector_array, stream.ctx)
    defer { mlx_vector_array_free(result) }

    return mlx_vector_array_values(result)
}