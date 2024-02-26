// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom

/// Applies an affine transformation to the input using a quantized weight matrix.
///
/// It is the quantized equivalent of ``Linear``.  For now its
/// parameters are frozen and will not be included in any gradient computation
/// but this will probably change in the future.
///
/// QuantizedLinear also provides several useful static to convert linear
/// layers to QuantizedLinear layers.
///
/// - ``from(linear:groupSize:bits:)`` -- returns a `QuantizedLinear` that applies the same
///   linear transformation up to the quantization error
/// - ``quantize(model:groupSize:bits:predicate:)`` -- swaps all the linear layers of the module
///     with `QuantizedLinear` ones
///
/// Please see the disucssion in ``Linear`` for considerations when replacing layers.
///
/// ### See Also
/// - ``init(weight:bias:groupSize:bits:)``
open class QuantizedLinear: Linear {

    public let groupSize: Int
    public let bits: Int

    public let scales: MLXArray
    public let biases: MLXArray

    /// Applies an affine transformation to the input using a quantized weight matrix.
    ///
    /// This is the quantized version of ``Linear``.  Typically this is used via ``quantize(model:groupSize:bits:predicate:)``.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    ///   - groupSize: The group size to use for the quantized weight
    ///   - bits: The bit width to use for the quantized weight
    public convenience init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true, groupSize: Int = 64,
        bits: Int = 4
    ) {
        let scale = sqrt(1 / Float(inputDimensions))
        let weight = uniform(low: -scale, high: scale, [outputDimensions, inputDimensions])

        let bias = bias ? MLXArray.zeros([outputDimensions]) : nil

        self.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits)
    }

    public init(weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits)

        self.scales = scales
        self.biases = biases

        super.init(weight: quantizedWeight, bias: bias)

        self.freeze()
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = quantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits
        )
        if let bias {
            x = x + bias
        }
        return x
    }

    /// Returns a QuantizedLinear layer that applies the same linear transformation up to the quantization error.
    ///
    /// - Parameters:
    ///   - linear: a `Linear` layer
    ///   - groupSize: The group size to use for the quantized weight
    ///   - bits: The bit width to use for the quantized weight
    /// - Returns: a new `QuantizedLayer`
    static public func from(linear: Linear, groupSize: Int = 64, bits: Int = 4) -> QuantizedLinear {
        QuantizedLinear(weight: linear.weight, bias: linear.bias, groupSize: groupSize, bits: bits)
    }

    /// Replace ``Linear`` layers with `QuantizedLinear`.
    ///
    /// Please see the disucssion in ``Linear`` for considerations when replacing layers.
    ///
    /// - Parameters:
    ///   - model: the model to update
    ///   - groupSize: The group size to use for the quantized weight
    ///   - bits: The bit width to use for the quantized weight
    ///   - predicate: optional predicate for identifying layers to change -- default finds all `Linear` layers
    static public func quantize(
        model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        predicate: (Linear) -> Bool = { _ in true }
    ) {
        let updates = model.leafModules().compactMapValues { m -> Module? in
            guard let linear = m as? Linear else { return nil }
            if predicate(linear) {
                return Self.from(linear: linear, groupSize: groupSize, bits: bits)
            } else {
                return nil
            }
        }

        model.update(modules: updates)
    }
}
