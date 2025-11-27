// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// A placeholder identity operator that is argument-insensitive.
public class Identity: Module, UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

/// Applies an affine transformation to the input.
///
/// Concretely:
///
/// ```swift
/// matmul(x, W) + b
/// ```
///
/// where `W` has shape `[inputDimensions, outputDimensions]` and `b` has
/// shape `[outputDimensions]`.
///
/// The values are initialized from the uniform distribution:
///
/// ```swift
/// scale = sqrt(1.0 / inputDimensions)
/// W = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions])
/// b = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
/// ```
///
/// ## Using In A Module
///
/// > Use `@ModuleInfo` with all your `Linear` module uses so that ``Module/update(modules:verify:)`` can
/// replace the modules, e.g. via ``QuantizedLinear/quantize(model:groupSize:bits:predicate:)``.
///
/// For example:
///
/// ```swift
/// public class FeedForward : Module {
///
///     @ModuleInfo var w1: Linear
///     @ModuleInfo var w2: Linear
///     @ModuleInfo var w3: Linear
///
///     public init(_ args: Configuration) {
///         self.w1 = Linear(args.dimensions, args.hiddenDimensions, bias: false)
///         self.w2 = Linear(args.hiddenDimensions, args.dimensions, bias: false)
///         self.w3 = Linear(args.dimensions, args.hiddenDimensions, bias: false)
///     }
/// ```
///
/// If a `key` is needed (to change the parameters key for ``Module/parameters()``) here is the
/// way to initialize the ivar:
///
/// ```swift
/// public class Example : Module {
///
///     @ModuleInfo(key: "weights_1") var w1: Linear
///
///     public init() {
///         self._w1.wrappedValue = Linear(10, 20, bias: false)
///     }
/// ```
///
/// ### See Also
/// - <doc:custom-layers>
/// - ``QuantizedLinear``
/// - ``Bilinear``
open class Linear: Module, UnaryLayer, Quantizable {

    public let weight: MLXArray
    public let bias: MLXArray?

    open var shape: (Int, Int) {
        weight.shape2
    }

    /// Applies an affine transformation to the input.
    ///
    /// Please see discussion in ``Module``.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    public init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions])
        if bias {
            self.bias = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()
    }

    /// Convenience initializer giving parameter names.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    public convenience init(inputDimensions: Int, outputDimensions: Int, bias: Bool = true) {
        self.init(inputDimensions, outputDimensions, bias: bias)
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedLinear`` to provide quantized weights and biases
    /// rather than have ``Linear`` compute them.
    public init(weight: MLXArray, bias: MLXArray? = nil) {
        self.weight = weight
        self.bias = bias
    }

    /// Describe the `inputDimensions` and `outputDimensions`.
    open override func describeExtra(_ indent: Int) -> String {
        let (outputDimensions, inputDimensions) = self.shape
        return
            "(inputDimensions=\(inputDimensions), outputDimensions=\(outputDimensions), bias=\(self.bias != nil))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let result: MLXArray
        if let bias {
            result = addMM(bias, x, weight.T)
        } else {
            result = matmul(x, weight.T)
        }
        return result
    }

    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Applies a bilinear transformation to the inputs.
///
/// Concretely:
///
/// ```swift
/// y = x1.matmul(w.T)
/// y = x2.matmul(y)
/// y = y + b
/// ```
///
/// where `w` has shape `[outputDimensions, inputDimensions2, inputDimensions1]` and `b` has
/// shape `[outputDimensions]`.
///
/// The values are initialized from the uniform distribution:
///
/// ```swift
/// scale = sqrt(1.0 / inputDimensions)
/// W = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions2, inputDimensions1])
/// b = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
/// ```
///
/// ### See Also
/// - <doc:custom-layers>
/// - ``Linear``
open class Bilinear: Module {

    public let weight: MLXArray
    public let bias: MLXArray?

    public init(
        _ inputDimensions1: Int, _ inputDimensions2: Int, _ outputDimensions: Int, bias: Bool = true
    ) {
        let scale = sqrt(1.0 / Float(inputDimensions1))
        self.weight = MLXRandom.uniform(
            -scale ..< scale, [outputDimensions, inputDimensions2, inputDimensions1])
        if bias {
            self.bias = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()
    }

    /// Describe the `inputDimensions` and `outputDimensions`.
    public override func describeExtra(_ indent: Int) -> String {
        "(inputDimensions1=\(weight.dim(2)), inputDimensions2=\(weight.dim(1)), outputDimensions=\(weight.dim(0)), bias=\(bias == nil ? "false" : "true"))"
    }

    open func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        // normalize shapes
        let (out, in2, in1) = weight.shape3
        let xShape = x1.shape.dropLast()
        let x1 = x1.reshaped(-1, in1)
        let x2 = x2.reshaped(-1, 1, in2)

        // perform the bilinear transform
        var y: MLXArray
        let w = weight.reshaped(out * in2, in1)
        y = x1.matmul(w.T)
        y = y.reshaped(-1, out, in2).swappedAxes(-2, -1)
        y = x2.matmul(y)
        y = y.squeezed(axis: 1)

        // reset the shape
        y = y.reshaped(xShape + [out])

        if let bias {
            y = y + bias
        }
        return y
    }
}
