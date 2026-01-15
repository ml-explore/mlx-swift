// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Implements the scaled dot product attention with multiple heads.
///
/// ### See Also
/// - <doc:transformers>
/// - ``init(dimensions:numHeads:queryInputDimensions:keyInputDimensions:valueInputDimensions:valueDimensions:valueOutputDimensions:bias:)``
open class MultiHeadAttention: Module {

    public let numHeads: Int

    @ModuleInfo(key: "query_proj") public var queryProjection: UnaryLayer
    @ModuleInfo(key: "key_proj") public var keyProjection: UnaryLayer
    @ModuleInfo(key: "value_proj") public var valueProjection: UnaryLayer
    @ModuleInfo(key: "out_proj") public var outProjection: UnaryLayer

    /// Implements the scaled dot product attention with multiple heads.
    ///
    /// Given inputs for queries, keys and values the ``MultiHeadAttention``
    /// produces new values by aggregating information from the input values
    /// according to the similarities of the input queries and keys.
    ///
    /// All inputs as well as the output are linearly projected without biases by
    /// default.
    ///
    /// ``MultiHeadAttention`` also takes an optional additive attention mask that
    /// should be broadcastable with `(batch, numHeads, # queries, # keys)`. The
    /// mask should have `-inf` or very large negative numbers at the positions
    /// that should *not* be attended to.
    ///
    /// - Parameters:
    ///   - dimensions: model dimensions and default for the other dimensions if they are not supplied
    ///   - numHeads: number of attention heads
    ///   - queryInputDimensions: input dimensions of queries
    ///   - keyInputDimensions: input dimensions of keys
    ///   - valueInputDimensions: input dimensions of values
    ///   - valueDimensions: dimensions of values after the projection
    ///   - valueOutputDimensions: dimensions new values will be projected to
    ///   - bias: if `true` uses a bias in the `Linear` layers
    ///
    /// ### See Also
    /// - ``createAdditiveCausalMask(_:dtype:)
    public init(
        dimensions: Int,
        numHeads: Int,
        queryInputDimensions: Int? = nil,
        keyInputDimensions: Int? = nil,
        valueInputDimensions: Int? = nil,
        valueDimensions: Int? = nil,
        valueOutputDimensions: Int? = nil,
        bias: Bool = false
    ) {
        precondition(dimensions % numHeads == 0)

        let queryInputDimensions = queryInputDimensions ?? dimensions
        let keyInputDimensions = keyInputDimensions ?? dimensions
        let valueInputDimensions = valueInputDimensions ?? dimensions
        let valueDimensions = valueDimensions ?? dimensions
        let valueOutputDimensions = valueOutputDimensions ?? dimensions

        self.numHeads = numHeads

        self._queryProjection.wrappedValue = Linear(queryInputDimensions, dimensions, bias: bias)
        self._keyProjection.wrappedValue = Linear(keyInputDimensions, dimensions, bias: bias)
        self._valueProjection.wrappedValue = Linear(
            valueInputDimensions, valueDimensions, bias: bias)
        self._outProjection.wrappedValue = Linear(
            valueDimensions, valueOutputDimensions, bias: bias)
    }

    open func callAsFunction(
        _ queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var queries = queryProjection(queries)
        var keys = keyProjection(keys)
        var values = valueProjection(values)

        queries = unflatten(queries, axis: -1, shape: [numHeads, -1])
            .transposed(0, 2, 1, 3)
        keys = unflatten(keys, axis: -1, shape: [numHeads, -1])
            .transposed(0, 2, 1, 3)
        values = unflatten(values, axis: -1, shape: [numHeads, -1])
            .transposed(0, 2, 1, 3)
        let scale = sqrt(1 / Float(queries.dim(-1)))

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                .none
            }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: maskMode)

        output = output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)

        return outProjection(output)
    }

    /// Creates an attention mask for use with ``callAsFunction(_:keys:values:mask:)``
    ///
    /// - Parameters:
    ///   - n: number of dimensions
    ///   - dtype: data type of the mask
    public static func createAdditiveCausalMask(_ n: Int, dtype: DType = .float32) -> MLXArray {
        let indices = MLXArray(0 ..< n)
        var mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        mask = mask.asType(dtype) * -1e9
        return mask
    }
}

class TransformerEncoderLayer: Module {

    let attention: MultiHeadAttention
    let ln1: LayerNorm
    let ln2: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    let dropout1: Dropout
    let dropout2: Dropout
    let activation: UnaryLayer
    let normFirst: Bool

    public init(
        dimensions: Int, numHeads: Int, mlpDimensions: Int? = nil, dropout: Float = 0.0,
        activation: UnaryLayer = ReLU(), normFirst: Bool
    ) {
        let mlpDimensions = mlpDimensions ?? dimensions * 4
        self.attention = MultiHeadAttention(dimensions: dimensions, numHeads: numHeads)
        self.ln1 = LayerNorm(dimensions: dimensions)
        self.ln2 = LayerNorm(dimensions: dimensions)
        self.linear1 = Linear(dimensions, mlpDimensions)
        self.linear2 = Linear(mlpDimensions, dimensions)
        self.dropout1 = Dropout(p: dropout)
        self.dropout2 = Dropout(p: dropout)
        self.activation = activation
        self.normFirst = normFirst
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var y: MLXArray
        var x = x

        if normFirst {
            y = ln1(x)
            y = attention(y, keys: y, values: y, mask: mask)
            y = dropout1(y)
            x = x + y

            y = ln2(x)
            y = linear1(y)
            y = activation(y)
            y = dropout2(y)
            y = linear2(y)
            y = x + y

        } else {
            y = attention(x, keys: x, values: x, mask: mask)
            y = dropout1(y)
            x = ln1(x + y)

            y = linear1(x)
            y = activation(y)
            y = dropout2(y)
            y = linear2(y)
            y = ln2(x + y)
        }

        return y
    }
}

class TransformerEncoder: Module {

    let layers: [TransformerEncoderLayer]
    let ln: LayerNorm

    public init(
        layerCount: Int, dimensions: Int, numHeads: Int, mlpDimensions: Int? = nil,
        dropout: Float = 0.0, activation: UnaryLayer = ReLU(), normFirst: Bool
    ) {
        self.layers = (0 ..< layerCount)
            .map { _ in
                TransformerEncoderLayer(
                    dimensions: dimensions, numHeads: numHeads, mlpDimensions: mlpDimensions,
                    dropout: dropout, activation: activation, normFirst: normFirst)
            }
        self.ln = LayerNorm(dimensions: dimensions)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var x = x

        for l in layers {
            x = l(x, mask: mask)
        }

        return ln(x)
    }
}

class TransformerDecoderLayer: Module {

    @ModuleInfo(key: "self_attention") var selfAttention: MultiHeadAttention
    @ModuleInfo(key: "cross_attention") var crossAttention: MultiHeadAttention
    let ln1: LayerNorm
    let ln2: LayerNorm
    let ln3: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    let dropout1: Dropout
    let dropout2: Dropout
    let dropout3: Dropout
    let activation: UnaryLayer
    let normFirst: Bool

    public init(
        dimensions: Int, numHeads: Int, mlpDimensions: Int? = nil, dropout: Float = 0.0,
        activation: UnaryLayer = ReLU(), normFirst: Bool
    ) {
        let mlpDimensions = mlpDimensions ?? dimensions * 4
        self._selfAttention.wrappedValue = MultiHeadAttention(
            dimensions: dimensions, numHeads: numHeads)
        self._crossAttention.wrappedValue = MultiHeadAttention(
            dimensions: dimensions, numHeads: numHeads)
        self.ln1 = LayerNorm(dimensions: dimensions)
        self.ln2 = LayerNorm(dimensions: dimensions)
        self.ln3 = LayerNorm(dimensions: dimensions)
        self.linear1 = Linear(dimensions, mlpDimensions)
        self.linear2 = Linear(mlpDimensions, dimensions)
        self.dropout1 = Dropout(p: dropout)
        self.dropout2 = Dropout(p: dropout)
        self.dropout3 = Dropout(p: dropout)
        self.activation = activation
        self.normFirst = normFirst
    }

    public func callAsFunction(
        _ x: MLXArray, memory: MLXArray, xMask: MLXArray, memoryMask: MLXArray
    ) -> MLXArray {
        var y: MLXArray
        var x = x

        if normFirst {
            y = ln1(x)
            y = selfAttention(y, keys: y, values: y, mask: xMask)
            y = dropout1(y)
            x = x + y

            y = ln2(x)
            y = crossAttention(y, keys: memory, values: memory, mask: memoryMask)
            y = dropout2(y)
            x = x + y

            y = ln3(x)
            y = linear1(y)
            y = activation(y)
            y = dropout3(y)
            y = linear2(y)
            y = x + y

        } else {
            y = selfAttention(x, keys: x, values: x, mask: xMask)
            y = dropout1(y)
            x = ln1(x + y)

            y = crossAttention(y, keys: memory, values: memory, mask: memoryMask)
            y = dropout2(y)
            x = ln2(x + y)

            y = linear1(x)
            y = activation(y)
            y = dropout3(y)
            y = linear2(y)
            y = ln3(x + y)
        }

        return y
    }
}

class TransformerDecoder: Module {

    let layers: [TransformerDecoderLayer]
    let ln: LayerNorm

    public init(
        layerCount: Int, dimensions: Int, numHeads: Int, mlpDimensions: Int? = nil,
        dropout: Float = 0.0, activation: UnaryLayer = ReLU(), normFirst: Bool
    ) {
        self.layers = (0 ..< layerCount)
            .map { _ in
                TransformerDecoderLayer(
                    dimensions: dimensions, numHeads: numHeads, mlpDimensions: mlpDimensions,
                    dropout: dropout, activation: activation, normFirst: normFirst)
            }
        self.ln = LayerNorm(dimensions: dimensions)
    }

    public func callAsFunction(
        _ x: MLXArray, memory: MLXArray, xMask: MLXArray, memoryMask: MLXArray
    ) -> MLXArray {
        var x = x

        for l in layers {
            x = l(x, memory: memory, xMask: xMask, memoryMask: memoryMask)
        }

        return ln(x)
    }
}

/// Implements a standard Transformer model.
///
/// The implementation is based on "Attention Is All You Need"
/// <https://arxiv.org/abs/1706.03762>.
///
/// The Transformer model contains an encoder and a decoder. The encoder
/// processes the input sequence and the decoder generates the output sequence.
/// The interaction between encoder and decoder happens through the attention
/// mechanism.
///
/// ### See Also
/// - <doc:transformers>
/// - <https://arxiv.org/abs/1706.03762>
open class Transformer: Module {

    let encoder: TransformerEncoder
    let decoder: TransformerDecoder

    /// Initialize the transformer.
    ///
    /// - Parameters:
    ///   - dimensions: number of expected features in the encoder/decoder
    ///   - numHeads: number of attention heads
    ///   - encoderLayerCount: number of layers in the encoder
    ///   - decoderLayerCount: number of layers in the decoder
    ///   - mlpDimensions: hidden dimensions of the MLP block in each layer.  Defaults to `4 * dimensions`
    ///   if not specified
    ///   - dropout: dropout value for the encode and decoder.  Dropout is used after each attention layer
    ///   and the activation in the MLP layer
    ///   - activation: the activation layer for the MLP hidden layer
    ///   - normFirst: if `true` encode and decoder layers will perform layer normalization before
    ///   attention and MLP operations, otherwise after
    public init(
        dimensions: Int = 512, numHeads: Int = 8, encoderLayerCount: Int = 6,
        decoderLayerCount: Int = 6, mlpDimensions: Int? = nil, dropout: Float = 0.0,
        activation: UnaryLayer = ReLU(), normFirst: Bool = false
    ) {
        self.encoder = TransformerEncoder(
            layerCount: encoderLayerCount, dimensions: dimensions, numHeads: numHeads,
            mlpDimensions: mlpDimensions, dropout: dropout, activation: activation,
            normFirst: normFirst)
        self.decoder = TransformerDecoder(
            layerCount: decoderLayerCount, dimensions: dimensions, numHeads: numHeads,
            mlpDimensions: mlpDimensions, dropout: dropout, activation: activation,
            normFirst: normFirst)
    }

    open func callAsFunction(
        source: MLXArray, target: MLXArray, sourceMask: MLXArray, targetMask: MLXArray,
        memoryMask: MLXArray
    ) -> MLXArray {
        let memory = encoder(source, mask: sourceMask)
        return decoder(target, memory: memory, xMask: targetMask, memoryMask: memoryMask)
    }
}
