// Copyright © 2026 Apple Inc.

import MLX

extension MLXArray {

    /// Returns a new array normalized by its L2 norm along the specified axis.
    ///
    /// This operation scales the vectors along `axis` to unit length. If the
    /// norm is smaller than `eps`, it is clamped to `eps` to ensure numerical
    /// stability and prevent division by zero.
    ///
    /// - Parameters:
    ///   - axis: The axis along which to compute the norm. Defaults to `-1`.
    ///   - eps: A small epsilon value to prevent division by zero. Defaults to `1e-12`.
    /// - Returns: An `MLXArray` with the same shape as the original, normalized along `axis`.
    ///
    /// - Complexity: O(n), where n is the total number of elements in the array.
    public func l2Normalized(axis: Int = -1, eps: Float = 1e-12) -> MLXArray {
        // 'self' represents the current MLXArray instance.
        // We compute the norm along the specified axis.
        let norm = MLXLinalg.norm(self, ord: 2, axis: axis, keepDims: true)

        // We use MLX.maximum to clamp the divisor.
        // This is more stable than adding eps to the norm.
        return self / MLX.maximum(norm, MLXArray(eps))
    }
}
