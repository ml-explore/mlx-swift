// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast

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
final public class RoPE: Module, UnaryLayer {

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
        let shape = x.shape
        var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
        x = MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
            offset: offset)
        return x.reshaped(shape)
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
    ///   - scale: multiplicative scale for the embeddings.  Default is `sqrt(dimensions / 2)`
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

public class SuScaledRotaryEmbedding {
    let dimensions: Int
    let base: Float
    let scale: Float
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let shortFactor: [Float]
    let longFactor: [Float]

    var invFreqShort: [Float]
    var invFreqLong: [Float]
    var scalingFactor: Float

    /// Initialize the embedding layer.
    ///
    /// - Parameters:
    ///   - dimensions: The feature dimensions to be rotated.
    ///   - base: Base for the exponential scaling.
    ///   - scale: The scale used to scale the positions.
    ///   - maxPositionEmbeddings: The maximum sequence length that this model was trained with. This is used to determine the size of the original RoPE embeddings when using long scaling.
    ///   - originalMaxPositionEmbeddings: The maximum sequence length for original scaling. This is used to determine the size of the original RoPE embeddings when using long scaling.
    ///   - shortFactor: Scaling factors for sequences of length less than `originalMaxPositionEmbeddings`.
    ///   - longFactor: Scaling factors for sequences of length greater than `originalMaxPositionEmbeddings`.
    public init(
        dimensions: Int, base: Float = 10_000, scale: Float = 1.0,
        maxPositionEmbeddings: Int = 131072, originalMaxPositionEmbeddings: Int = 4096,
        shortFactor: [Float] = [1.0], longFactor: [Float] = [1.0]
    ) {
        self.dimensions = dimensions
        self.base = base
        self.scale = scale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.shortFactor = shortFactor
        self.longFactor = longFactor

        // Precompute inverse frequency arrays
        invFreqShort = (0 ..< dimensions / 2).map { i in
            1.0 / (shortFactor[i % shortFactor.count] * pow(base, Float(i) / Float(dimensions)))
        }
        invFreqLong = (0 ..< dimensions / 2).map { i in
            1.0
                / (scale * longFactor[i % longFactor.count]
                    * pow(base, Float(i) / Float(dimensions)))
        }

        // Compute scaling factor
        scalingFactor = sqrt(
            1.0 + log(Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings))
                / log(Float(originalMaxPositionEmbeddings)))

        print("invFreqShort: \(invFreqShort.count) elements")
        print("invFreqLong: \(invFreqLong.count) elements")
        print("Scaling Factor: \(scalingFactor)")
    }

    /// Get cosine and sine embeddings.
    private func getCosSin(offset: Int, length: Int) -> (cos: MLXArray, sin: MLXArray) {
        let positionIDs = MLXArray(offset ..< offset + length).reshaped([1, 1, -1, 1])
        print("positionIDs in getCosSin: \(positionIDs.description)")
        let invFreq = (offset + length) > originalMaxPositionEmbeddings ? invFreqLong : invFreqShort
        print("Using \(invFreq == invFreqLong ? "long" : "short") inverse frequency")
        let freqs = positionIDs * MLXArray(invFreq).reshaped([1, 1, 1, -1])
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        let cos = emb.cos() * scalingFactor
        let sin = emb.sin() * scalingFactor
        print(
            "cos.shape in getCosSin: \(cos.shape.description), sin.shape in getCosSin: \(sin.shape.description)"
        )
        return (cos, sin)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        guard let lastShape = x.shape.last else {
            fatalError(
                "Last dimension size of `x` in `rotateHalf` could not be determined. `x.description`: \(x.description)"
            )
        }
        guard lastShape % 2 == 0 else {
            fatalError(
                "Last dimension size of `x` in `rotateHalf` is \(lastShape), which is not an even number."
            )
        }
        let midpoint = lastShape / 2
        print("midpoint in rotateHalf: \(midpoint)")
        let x1 = x[.ellipsis, ..<midpoint]
        let x2 = x[.ellipsis, midpoint...]
        print("x1 shape: \(x1.shape.description), x2 shape: \(x2.shape.description)")
        return MLX.concatenated([-x2, x1], axis: -1)
    }

    /// Apply the embedding rotation.
    ///
    /// - Parameters:
    ///   - x: Input array.
    ///   - offset: Position offset.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        print("x.shape in callAsFunction: \(x.shape.description)")
        guard let lastShape = x.shape.last, lastShape == dimensions else {
            fatalError(
                "Expected the last dimension of x to be \(dimensions), but got \(String(describing: x.shape.last))"
            )
        }
        let (cos, sin) = getCosSin(offset: offset, length: x.shape[2])
        print("x.shape in callAsFunction: \(x.shape.description)")
        print("cos.shape in callAsFunction: \(cos.shape.description)")
        print("sin.shape in callAsFunction: \(sin.shape.description)")
        let result = (x * cos) + (rotateHalf(x) * sin)
        print("Result shape: \(result.shape.description)")
        return result
    }
}
