// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Protocol for layers that can be quantized
public protocol Quantizable {

    /// Return the module as a quantized representation
    @available(*, deprecated, message: "prefer the toQuantized that takes a mode")
    func toQuantized(groupSize: Int, bits: Int) -> Module

    /// Return the module as a quantized representation
    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module
}

extension Quantizable {
    public func toQuantized(groupSize: Int, bits: Int) -> Module {
        toQuantized(groupSize: groupSize, bits: bits, mode: .affine)
    }
}

/// Protocol for layers that are quantized.
public protocol Quantized: Module {
    var groupSize: Int { get }
    var bits: Int { get }
    var mode: QuantizationMode { get }
}

/// Quantize any ``Quantizable`` layer that is not already quantized.
public func quantizeSingle(
    layer: Module, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
) -> Quantized? {
    if layer is Quantized {
        // already quantized
        nil
    } else if let quantizable = layer as? Quantizable {
        quantizable.toQuantized(groupSize: groupSize, bits: bits, mode: mode) as? Quantized
    } else {
        nil
    }
}

/// Quantize the sub-modules of a module according to a filter.
///
/// By default all ``Linear`` and ``Embedding`` layers will be quantized.
///
/// - Parameters:
///   - model: model to quantize
///   - groupSize: quantization group size
///   - bits: bits per parameter
///   - mode: quantization mode
///   - filter: filter receiving path and module -- return `false` to skip a layer
///   - apply: function to attempt the quantization -- the default implementation will quantize ``Linear`` and ``Embedding``
/// ### See Also
/// - ``quantize(model:filter:apply:)-(_,_,(Module,Int,Int,QuantizationMode)->Module?)``
public func quantize(
    model: Module,
    groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine,
    filter: (String, Module) -> Bool = { _, _ in true },
    apply: (Module, Int, Int, QuantizationMode) -> Module? = quantizeSingle(
        layer:groupSize:bits:mode:)
) {
    let updates =
        model
        .leafModules()
        .flattened()
        .compactMap { (path, m) -> (String, Module)? in
            if filter(path, m) {
                if let quantized = apply(m, groupSize, bits, mode) {
                    return (path, quantized)
                }
            }

            return nil
        }

    model.update(modules: ModuleChildren.unflattened(updates))
}

@available(*, deprecated, message: "use quantize that takes a 4 argument apply")
@_disfavoredOverload
public func quantize(
    model: Module, groupSize: Int = 64, bits: Int = 4,
    filter: (String, Module) -> Bool = { _, _ in true },
    apply: (Module, Int, Int) -> Module? = {
        quantizeSingle(layer: $0, groupSize: $1, bits: $2, mode: .affine)
    }
) {
    quantize(
        model: model, groupSize: groupSize, bits: bits, mode: .affine, filter: filter,
        apply: { l, g, b, n in apply(l, g, b) }
    )
}

/// Quantize the sub-modules of a module according to a filter.
///
/// By default all ``Linear`` and ``Embedding`` layers will be quantized.
///
/// - Parameters:
///   - model: model to quantize
///   - filter: filter receiving path and module -- return a tuple of `(groupSize: Int, bits: Int, mode: QuantizationMode)` or `nil` to skip quantization
///   - apply: function to attempt the quantization -- the default implementation will quantize ``Linear`` and ``Embedding`` layers
/// ### See Also
/// - ``quantize(model:groupSize:bits:filter:apply:)``
public func quantize(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?,
    apply: (Module, Int, Int, QuantizationMode) -> Module? = quantizeSingle(
        layer:groupSize:bits:mode:)
) {
    let updates =
        model
        .leafModules()
        .flattened()
        .compactMap { (path, m) -> (String, Module)? in
            if let (groupSize, bits, mode) = filter(path, m) {
                if let quantized = apply(m, groupSize, bits, mode) {
                    return (path, quantized)
                }
            }

            return nil
        }

    model.update(modules: ModuleChildren.unflattened(updates))
}

@available(*, deprecated, message: "use quantize that takes a 4 argument apply")
@_disfavoredOverload
public func quantize(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int)?,
    apply: (Module, Int, Int) -> Module? = {
        quantizeSingle(layer: $0, groupSize: $1, bits: $2, mode: .affine)
    }
) {
    quantize(
        model: model,
        filter: {
            if let (g, b) = filter($0, $1) {
                return (g, b, .affine)
            } else {
                return nil
            }
        },
        apply: { m, g, b, mode in
            apply(m, g, b)
        }
    )
}

/// The same as ``Embedding`` but with a quantized weight matrix.
open class QuantizedEmbedding: Embedding, Quantized {

    public let groupSize: Int
    public let bits: Int

    public let mode: QuantizationMode
    public let scales: MLXArray
    public let biases: MLXArray?

    open override var shape: (Int, Int) {
        let (embeddingCount, dimensions) = super.shape
        return (embeddingCount, dimensions * 32 / self.bits)
    }

    convenience public init(
        embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        let scale = sqrt(1 / Float(dimensions))
        let weight = MLXRandom.normal([embeddingCount, dimensions]) * scale

        self.init(weight: weight, groupSize: groupSize, bits: bits, mode: mode)
    }

    public convenience init(
        _ other: Embedding, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.init(weight: other.weight, groupSize: groupSize, bits: bits, mode: mode)
    }

    public init(
        weight: MLXArray, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode)

        self.scales = scales
        self.biases = biases

        super.init(weight: quantizedWeight)

        self.freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let x = x.flattened()
        let out = dequantized(
            weight[x], scales: scales[x], biases: biases == nil ? nil : biases![x],
            groupSize: groupSize, bits: bits, mode: mode)
        return out.reshaped(s + [-1])
    }

    open override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
            bits: bits, mode: mode)
    }
}

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
/// Please see the discussion in ``Linear`` for considerations when replacing layers.
///
/// ### See Also
/// - ``QuantizedLinear/init(_:_:bias:groupSize:bits:mode:)``
open class QuantizedLinear: Linear, Quantized {

    public let groupSize: Int
    public let bits: Int

    public let mode: QuantizationMode
    public let scales: MLXArray
    public let biases: MLXArray?

    open override var shape: (Int, Int) {
        let shape = weight.shape2
        return (shape.0, shape.1 * 32 / bits)
    }

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
    ///   - mode: quantization mode
    public convenience init(
        _ inputDimensions: Int, _ outputDimensions: Int,
        bias: Bool = true, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        let scale = sqrt(1 / Float(inputDimensions))
        let weight = MLXRandom.uniform(
            low: -scale, high: scale, [outputDimensions, inputDimensions])

        let bias = bias ? MLXArray.zeros([outputDimensions]) : nil

        self.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Initialize a QuantizedLinear layer that applies the same linear transformation up to the quantization error.
    ///
    /// - Parameters:
    ///   - other: a `Linear` layer
    ///   - groupSize: The group size to use for the quantized weight
    ///   - bits: The bit width to use for the quantized weight
    ///   - mode: quantization mode
    public convenience init(
        _ other: Linear, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.init(
            weight: other.weight, bias: other.bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Initialize a ``QuantizedLinear`` with non-quantized weights and bias.
    public init(
        weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits)

        self.scales = scales
        self.biases = biases

        super.init(weight: quantizedWeight, bias: bias)

        self.freeze()
    }

    /// Initializer meant for subclasses to provide arrays directly.
    ///
    /// ### See Also
    /// - ``Linear/init(weight:bias:)``
    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.scales = scales
        self.biases = biases
        super.init(weight: weight, bias: bias)
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
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
    @available(*, deprecated, renamed: "init(_:groupSize:bits:)")
    static public func from(linear: Linear, groupSize: Int = 64, bits: Int = 4) -> QuantizedLinear {
        QuantizedLinear(linear, groupSize: groupSize, bits: bits)
    }

    /// Replace ``Linear`` layers with `QuantizedLinear`.
    ///
    /// Please see the discussion in ``Linear`` for considerations when replacing layers.
    ///
    /// - Parameters:
    ///   - model: the model to update
    ///   - groupSize: The group size to use for the quantized weight
    ///   - bits: The bit width to use for the quantized weight
    ///   - predicate: optional predicate for identifying layers to change -- default finds all `Linear` layers
    @available(*, deprecated, renamed: "quantize(model:groupSize:bits:filter:apply:)")
    static public func quantize(
        model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        predicate: (Linear) -> Bool = { _ in true }
    ) {
        let updates = model.leafModules().compactMapValues { m -> Module? in
            guard let linear = m as? Linear else { return nil }
            if predicate(linear) {
                return QuantizedLinear(linear, groupSize: groupSize, bits: bits)
            } else {
                return nil
            }
        }

        model.update(modules: updates)
    }
}
