// Copyright © 2024 Apple Inc.

import Cmlx

extension MLXFast {
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
    public static func RoPE(
        _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float,
        offset: Int,
        freqs: MLXArray? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        mlx_fast_rope(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, Int32(offset),
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Optimized implementation of `NN.RoPE` with array offset for batched inference.
    ///
    /// This overload accepts an array offset, allowing different position offsets for each
    /// sequence in a batch. The offset can be a scalar array or a vector with length
    /// matching the batch size.
    ///
    /// - Parameters:
    ///   - array: input array
    ///   - dimensions: The feature dimensions to be rotated. If the input feature is larger
    ///     than dims then the rest is left unchanged.
    ///   - traditional: If `true` choose the traditional implementation which is slightly less efficient.
    ///   - base: The base used to compute angular frequency for each dimension in the positional encodings.
    ///   - scale: The scale used to scale the positions.
    ///   - offset: The position offset as an array. Can be a scalar or a vector of offsets for each batch element.
    ///   - freqs: Optional frequencies to use with RoPE.
    ///   - stream: stream or device to evaluate on
    /// - Returns: The input with rotary positional encoding applied.
    public static func RoPE(
        _ array: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        let offset = offset
        mlx_fast_rope_dynamic(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, offset.ctx,
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
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
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: mask array
    ///   - sinks: optional array of attention sinks
    ///   - memoryEfficientThreshold: unused
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
        sinks: MLXArray? = nil,
        memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_fast_scaled_dot_product_attention(
            &result,
            queries.ctx, keys.ctx, values.ctx, scale,
            "", mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx,
            stream.ctx)
        return MLXArray(result)
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
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: a ``ScaledDotProductAttentionMaskMode``
    ///   - sinks: optional array of attention sinks
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: ScaledDotProductAttentionMaskMode,
        sinks: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_fast_scaled_dot_product_attention(
            &result,
            queries.ctx, keys.ctx, values.ctx, scale,
            mask.mode, mask.mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx,
            stream.ctx)
        return MLXArray(result)
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
    public static func rmsNorm(
        _ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_fast_rms_norm(&result, x.ctx, weight.ctx, eps, stream.ctx)
        return MLXArray(result)
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
    public static func layerNorm(
        _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_layer_norm(
            &result, x.ctx, (weight ?? .mlxNone).ctx, (bias ?? .mlxNone).ctx, eps, stream.ctx)
        return MLXArray(result)
    }
}
