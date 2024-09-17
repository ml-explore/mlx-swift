// Copyright © 2024 Apple Inc.

import Cmlx
import MLX

/// Optimized implementation of `NN.RoPE`.
///
/// Used like this:
///
/// ```swift
/// let x: MLXArray
/// let dimensions: Int
/// let traditional: Bool
/// let base: Float
/// let scale: Float
/// let offset: Int
///
/// let shape = x.shape
/// var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
/// x = MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional, base: base, scale: scale, offset: offset)
/// return x.reshaped(shape)
/// ```
///
/// > Note: `MLXNN.RoPE` uses this implementation internally.
public func RoPE(
    _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float, scale: Float, offset: Int,
    freqs: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    // TODO base and freqs (scalars) should be optional and mutually exclusive -- perhaps an enum?
    MLXArray(
        mlx_fast_rope(
            array.ctx, Int32(dimensions), traditional, base, scale, Int32(offset),
            freqs?.ctx, stream.ctx))
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
///
/// Specifically this implements:
///
/// ```swift
/// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
/// if let mask {
///     scores = scores + mask
/// }
///
/// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
///
/// return matmul(scores, values).transposed(0, 2, 1, 3)
/// ```
public func scaledDotProductAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
    memoryEfficientThreshold: Int = 1_000_000, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        // TODO ideally memoryEfficientThreshold is an optional scalar -- leave the
        // default value to the backing implementation
        mlx_fast_scaled_dot_product_attention(
            queries.ctx, keys.ctx, values.ctx, scale, mask?.ctx,
            Int32(memoryEfficientThreshold), stream.ctx))
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default)
    -> MLXArray
{
    MLXArray(mlx_fast_rms_norm(x.ctx, weight.ctx, eps, stream.ctx))
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.  If not given no scaling will occur.
///   - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///     with the same size as the last axis of `x`.  It not given no offset will occur.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func layerNorm(
    _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
    stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_fast_layer_norm(x.ctx, weight?.ctx, bias?.ctx, eps, stream.ctx))
}

/// Quantize the matrix `w` using the provided `scales` and
/// `biases` and the `groupSize` and `bits` configuration.

/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.affine_quantize.html)
///
/// - Parameters:
///   - w: Matrix to be quantized
///   - scales: The scales to use per `groupSize` elements of `w`
///   - biases: The biases to use per `groupSize` elements of `w`
///   - groupSize: The size of the group in `w` that shares a scale and bias.
///   - bits: The number of bits occupied by each element in `w`.
///   - stream: stream or device to evaluate on
/// - Returns: quantized version of `w`
public func affineQuantized(
    _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int = 64, bits: Int = 4,
    stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        mlx_fast_affine_quantize(
            w.ctx, scales.ctx, biases.ctx, Int32(groupSize), Int32(bits), stream.ctx))
}
