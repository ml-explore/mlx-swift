// Copyright © 2024 Apple Inc.

import Cmlx

extension MLXFast {

    /// Core RoPE implementation using pure MLX operations.
    /// Matches the C++ fallback in fast.cpp (lines 417-501).
    private static func _ropeImpl(
        _ x: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray?
    ) -> MLXArray {
        let shape = x.shape
        var x = x

        // Reshape to 4D [B, N, T, D]
        if x.ndim == 3 {
            x = x.expandedDimensions(axis: 1)
        } else if x.ndim > 4 {
            x = x.flattened(start: 1, end: 1 + (x.ndim - 4))
        }

        let B = x.dim(0)
        let N = x.dim(1)
        let T = x.dim(2)
        let t = x.dtype
        let halfDims = dimensions / 2

        // Expand batch offsets [B] -> [B, 1, 1] for broadcasting
        var off = offset
        if off.size > 1 {
            off = off.expandedDimensions(axes: [-1, -2])
        }

        // positions = (arange(T) + offset) * scale
        let positions = (arange(T, dtype: .float32) + off) * MLXArray(scale)

        // Compute inverse frequencies
        let invFreqs: MLXArray
        if let freqs {
            invFreqs = reciprocal(freqs)
        } else {
            // inv_freqs = exp(arange(0, -halfDims, -1) * log(base) / halfDims)
            // = [base^0, base^(-1/halfDims), base^(-2/halfDims), ...]
            let logBasePerHalfDim = log(MLXArray(base)) / MLXArray(Float(halfDims))
            invFreqs = exp(
                arange(0.0, Double(-halfDims), step: -1.0, dtype: .float32) * logBasePerHalfDim
            )
        }

        // theta: [T, halfDims] or [B, 1, T, halfDims]
        let theta = positions.expandedDimensions(axis: -1) * invFreqs
        let coss = cos(theta).asType(t)
        let sins = sin(theta).asType(t)

        if traditional {
            // Traditional: rotate consecutive pairs (even/odd interleaved)
            let x1 = x[.ellipsis, .stride(from: 0, to: dimensions, by: 2)]
            let x2 = x[.ellipsis, .stride(from: 1, to: dimensions, by: 2)]
            let out1 = (x1 * coss - x2 * sins).expandedDimensions(axis: -1)
            let out2 = (x1 * sins + x2 * coss).expandedDimensions(axis: -1)
            // Interleave back: [.., halfDims, 2] -> reshape [.., dims]
            var out = concatenated([out1, out2], axis: -1).reshaped(B, N, T, dimensions)
            if dimensions < x.dim(-1) {
                out = concatenated([out, x[.ellipsis, dimensions...]], axis: -1)
            }
            return out.reshaped(shape)
        } else {
            // Modern: split at halfDims boundary (more efficient)
            let x1 = x[.ellipsis, ..<halfDims]
            let x2 = x[.ellipsis, halfDims ..< dimensions]
            let out1 = x1 * coss - x2 * sins
            let out2 = x1 * sins + x2 * coss
            var parts = [out1, out2]
            if dimensions < x.dim(-1) {
                parts.append(x[.ellipsis, dimensions...])
            }
            return concatenated(parts, axis: -1).reshaped(shape)
        }
    }

    public static func RoPE(
        _ x: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: Int,
        freqs: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        _ropeImpl(
            x, dimensions: dimensions, traditional: traditional,
            base: base ?? 10000.0, scale: scale,
            offset: MLXArray(Int32(offset)), freqs: freqs)
    }

    public static func RoPE(
        _ x: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        _ropeImpl(
            x, dimensions: dimensions, traditional: traditional,
            base: base ?? 10000.0, scale: scale,
            offset: offset, freqs: freqs)
    }

    // Fallback rmsNorm implementation
    public static func rmsNorm(
        _ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default
    ) -> MLXArray {
        // RMS norm: weight * x * rsqrt(mean(x^2) + eps)
        let meanSquare = mean(x * x, axis: -1, keepDims: true)
        return weight * x * rsqrt(meanSquare + eps)
    }

    // Fallback layerNorm implementation
    public static func layerNorm(
        _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        var normalized = (x - mean) * rsqrt(variance + eps)
        if let weight {
            normalized = normalized * weight
        }
        if let bias {
            normalized = normalized + bias
        }
        return normalized
    }

    // Fallback scaledDotProductAttention implementation
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXArray?,
        sinks: MLXArray? = nil,
        memoryEfficientThreshold: Int? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        Self.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.map { .array($0) } ?? .none,
            sinks: sinks, memoryEfficientThreshold: memoryEfficientThreshold, stream: stream
        )
    }

    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: ScaledDotProductAttentionMaskMode,
        sinks: MLXArray? = nil,
        memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        // Handle GQA (Grouped Query Attention) where nHeads > nKVHeads
        let nHeads = queries.dim(1)
        let nKVHeads = keys.dim(1)

        var expandedKeys = keys
        var expandedValues = values

        if nHeads != nKVHeads {
            // Repeat KV heads to match query heads
            // e.g., if nHeads=32, nKVHeads=8, each KV head is repeated 4 times
            let repeats = nHeads / nKVHeads
            let B = keys.dim(0)
            let L = keys.dim(2)
            let D = keys.dim(3)

            // Expand and repeat: [B, nKVHeads, L, D] -> [B, nHeads, L, D]
            // Use repeated() free function which is the public API for tiling along an axis
            expandedKeys = repeated(
                keys.reshaped(B, nKVHeads, 1, L, D),
                count: repeats,
                axis: 2
            ).reshaped(B, nHeads, L, D)
            expandedValues = repeated(
                values.reshaped(B, nKVHeads, 1, L, D),
                count: repeats,
                axis: 2
            ).reshaped(B, nHeads, L, D)
        }

        var scores = (queries * scale).matmul(expandedKeys.transposed(0, 1, 3, 2))

        switch mask {
        case .none:
            break
        case .causal:
            let L = queries.dim(2)
            let S = keys.dim(2)
            let indices_q = MLXArray(0 ..< L)
            let indices_k = MLXArray(0 ..< S)
            let causalMask =
                indices_q.expandedDimensions(axis: 1) .>= (indices_k - MLXArray(S - L))
            let maskValues = MLXArray(Float(-1e9))
            scores = MLX.where(causalMask, scores, maskValues)
        case .array(let maskArray):
            if maskArray.dtype == .bool {
                let maskValues = MLXArray(Float(-1e9))
                scores = MLX.where(maskArray, scores, maskValues)
            } else {
                scores = scores + maskArray
            }
        case .arrays(let maskArrays):
            if let maskArray = maskArrays.first {
                if maskArray.dtype == .bool {
                    let maskValues = MLXArray(Float(-1e9))
                    scores = MLX.where(maskArray, scores, maskValues)
                } else {
                    scores = scores + maskArray
                }
            }
        }

        scores = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return matmul(scores, expandedValues)
    }
}
