// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// A `Layer`  (``Module`` subclass) that can be evaluated with an array and offset.
public protocol OffsetLayer: Module {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray
}

/// A layer (``Module`` subclass) that can be evaluated with an input array and a per-example
/// position offset.
///
/// Use this for batched inference when each sequence in the batch may be at a different position
/// in the prompt. The `offset` is provided as an `MLXArray` so it can be:
/// - a scalar (one offset applied to the whole batch), or
/// - a vector of length `batchSize` (one offset per sequence).
///
/// Layers that support this protocol should broadcast the `offset` over the remaining dimensions
/// of `x` as appropriate.
public protocol ArrayOffsetLayer: Module {
    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray
}

/// Implements the rotary positional encoding.
///
/// The traditional implementation rotates consecutive pairs of elements in the
/// feature dimension while the default implementation rotates pairs with
/// stride half the feature dimensions for efficiency.
///
/// For more details see _RoFormer: Enhanced Transformer with Rotary Position
/// Embedding_ ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864))
///
/// ### See Also
/// - <doc:positional-encoding>
final public class RoPE: Module, UnaryLayer, OffsetLayer, ArrayOffsetLayer {

    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    /// Initialize ``RoPE``.
    ///
    /// - Parameters:
    ///   - dimensions: The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged
    ///   - traditional: If `true` choose the traditional implementation which is slightly less efficient
    ///   - base: The base used to compute angular frequency for each dimension in the positional encodings
    ///   - scale: scale used to scale the positions
    public init(dimensions: Int, traditional: Bool = false, base: Float = 10_000, scale: Float = 1)
    {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
            offset: offset)
    }

    /// Evaluate with array offset for batched inference with different positions per sequence.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - offset: Integer position offsets (recommended `.int32`) as a scalar or a vector of length `batchSize`.
    ///     For no offset, call ``callAsFunction(_:)`` or pass `MLXArray([Int32(0)])`.
    /// - Returns: the input with rotary positional encoding applied
    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base,
            scale: scale, offset: offset)
    }

    /// Evaluate with `offset` of `0`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, offset: 0)
    }
}

/// Implements sinusoidal positional encoding.
///
/// For more details see the paper "Attention Is All You Need"
/// <https://arxiv.org/abs/1706.03762>.
///
/// ### See Also
/// - <doc:positional-encoding>
open class SinusoidalPositionalEncoding: Module, UnaryLayer {

    let _sigmas: MLXArray
    public let scale: Float
    public let cosineFirst: Bool

    /// Initialize the layer.
    /// - Parameters:
    ///   - dimensions: dimensionality of the resulting positional embeddings
    ///   - minFrequency: minimum frequency expected
    ///   - maxFrequency: maximum frequency expected
    ///   - scale: multiplicative scale for the embeddings.  Default is `sqrt(2 / dimensions)`
    ///   - cosineFirst: if `true` embed using `[cos(x), sin(x)]` instead of the reverse
    ///   - fullTurns: if `true` multiply the frequencies with `2 * pi`
    public init(
        dimensions: Int, minFrequency: Float = 0.0001, maxFrequency: Float = 1, scale: Float? = nil,
        cosineFirst: Bool = false, fullTurns: Bool = false
    ) {
        let oneZero = 1 - MLXArray(0 ..< (dimensions / 2)) / (dimensions / 2 - 1)
        let minFrequency = log(minFrequency)
        let maxFrequency = log(maxFrequency)

        let sigmas = exp(oneZero * (maxFrequency - minFrequency) + minFrequency)
        if fullTurns {
            self._sigmas = sigmas * (2 * Float.pi)
        } else {
            self._sigmas = sigmas
        }

        self.scale = scale ?? pow((2.0 / Float(dimensions)), 0.5)
        self.cosineFirst = cosineFirst
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.expandedDimensions(axis: -1) * _sigmas

        let cosy = cos(y)
        let siny = sin(y)

        if cosineFirst {
            y = concatenated([cosy, siny], axis: -1)
        } else {
            y = concatenated([siny, cosy], axis: -1)
        }
        if scale != 1 {
            y = y * scale
        }

        return y
    }
}

/// ### See Also
/// - <doc:positional-encoding>
final public class ALiBi: Module {

    struct Key: Hashable {
        let qSequenceLength: Int
        let kSequenceLength: Int
        let numHeads: Int
        let offset: Int
        let dtype: DType
    }

    static let cache = Cache<Key, MLXArray>()

    public override init() {
    }

    static func alibiSlope(numHeads: Int) -> MLXArray {
        let x = pow(pow(2, 8), (1 / Float(numHeads)))
        let out = pow(x, -MLXArray(1 ..< (numHeads + 1)))
        return out.expandedDimensions(axes: [-1, -2])
    }

    static func alibiMatrix(key: Key) -> MLXArray {
        if let value = cache[key] {
            return value
        }

        let x1 = MLXArray(key.offset ..< key.qSequenceLength).expandedDimensions(axis: 1)
        let x2 = MLXArray(0 ..< key.kSequenceLength).expandedDimensions(axis: 1)
        let distanceMatrix = -abs(expandedDimensions((x1 - x2), axes: [0, 1]))

        let slope = alibiSlope(numHeads: key.numHeads)
        let mask = (distanceMatrix * slope).asType(key.dtype)

        cache[key] = mask

        return mask
    }

    public func callAsFunction(attentionScores: MLXArray, offset: Int = 0, mask: MLXArray? = nil)
        -> MLXArray
    {
        let key = Key(
            qSequenceLength: attentionScores.dim(-2) + offset,
            kSequenceLength: attentionScores.dim(-1), numHeads: attentionScores.dim(1),
            offset: offset, dtype: attentionScores.dtype)

        var alibiMask = Self.alibiMatrix(key: key)
        if let mask {
            alibiMask = alibiMask + mask
        }

        return attentionScores + alibiMask
    }
}
