// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Applies the element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```swift
/// MLX.sigmoid(x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``Sigmoid``
public func sigmoid(_ x: MLXArray) -> MLXArray {
    MLX.sigmoid(x)
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(x, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``ReLU``
public func relu(_ x: MLXArray) -> MLXArray {
    compiledRelu(x)
}

/// Applies the Leaky Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(negativeSlope * x, x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``LeakyReLU``
public func leakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    let negativeSlope = negativeSlope.asMLXArray(dtype: x.dtype)
    return compiledLeakyRelu(x, negativeSlope)
}

@available(*, deprecated, renamed: "logSoftmax(_:axis:)")
@_documentation(visibility: internal)
public func logSoftMax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    logSoftmax(x, axis: axis)
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```swift
/// x - MLX.logSumExp(x, axis: axis, keepDims: true)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``LogSoftmax``
public func logSoftmax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    x - logSumExp(x, axis: axis, keepDims: true)
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.which(x .> 0, x, alpha * (exp(x) - 1))
/// ```
///
/// ### See Also
/// - <doc:activations>
public func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    let alpha = alpha.asMLXArray(dtype: x.dtype)
    return compiledElu(x, alpha)
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```swift
/// MLX.minimum(MLX.maximum(x, 0), 6)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``ReLU6``
public func relu6(_ x: MLXArray) -> MLXArray {
    compiledRelu6(x)
}

/// Applies the squared Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.relu(x).square()
/// ```
///
/// /// ### See Also
/// - <doc:activations>
/// - ``ReLUSquared``
public func reluSquared(_ x: MLXArray) -> MLXArray {
    compiledReluSquared(x)
}

@available(*, deprecated, renamed: "softplus(_:)")
@_documentation(visibility: internal)
public func softPlus(_ x: MLXArray) -> MLXArray {
    softplus(x)
}

/// Applies the Softplus function.
///
/// This is:
///
/// ```swift
/// MLX.logAddExp(x, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``Softplus``
public func softplus(_ x: MLXArray) -> MLXArray {
    logAddExp(x, 0)
}

@available(*, deprecated, renamed: "softplus(_:)")
@_documentation(visibility: internal)
public func softSign(_ x: MLXArray) -> MLXArray {
    softsign(x)
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```swift
/// x / (1 + MLX.abs(x))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``Softsign``
public func softsign(_ x: MLXArray) -> MLXArray {
    compiledSoftsign(x)
}

/// Applies the Softshrink activation function.
///
/// This is (element-wise):
///
/// ```swift
/// if x > lambda {
///     x - lambda
/// }  else if x < -lambda {
///     x + lambda
/// } else {
///     0
/// }
/// ```
///
/// - Parameters:
///   - x: input array
///   - lambda: lambda value
public func softshrink(_ x: MLXArray, lambda: Float = 0.5) -> MLXArray {
    let lambda = lambda.asMLXArray(dtype: x.dtype)
    return compiledSoftshrink(x, lambda)
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(x, 0.0) + alpha * (MLX.exp(MLX.minimum(x, 0.0) / alpha) - 1)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``CELU``
public func celu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    let alpha = alpha.asMLXArray(dtype: x.dtype)
    return compiledCelu(x, alpha)
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```swift
/// x * MLX.sigmoid(x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``SiLU``
public func silu(_ x: MLXArray) -> MLXArray {
    compiledSilu(x)
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```swift
/// -softPlus(-x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``LogSigmoid``
public func logSigmoid(_ x: MLXArray) -> MLXArray {
    compiledLogSigmoid(x)
}

/// Applies the Gaussian Error Linear Units function.
///
/// This is:
///
/// ```swift
/// x * (1 + MLX.erf(x / sqrt(2))) / 2
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``GELU``
/// - ``geluApproximate(_:)``
/// - ``geluFastApproximate(_:)``
public func gelu(_ x: MLXArray) -> MLXArray {
    compiledGelu(x)
}

/// An approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```swift
/// 0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x**3)))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``GELU``
/// - ``gelu(_:)``
/// - ``geluFastApproximate(_:)``
public func geluApproximate(_ x: MLXArray) -> MLXArray {
    compiledGeluApproximate(x)
}

/// A fast approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```swift
/// x * sigmoid(1.702 * x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``GELU``
/// - ``gelu(_:)``
/// - ``geluApproximate(_:)``
public func geluFastApproximate(_ x: MLXArray) -> MLXArray {
    compiledGeluFastApproximate(x)
}

/// Applies the gated linear unit function.
///
/// This function splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
///
/// ### See Also
/// - <doc:activations>
/// - ``GLU``
public func glu(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let (a, b) = x.split(axis: axis)
    return a * sigmoid(b)
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```swift
/// MLX.where(x .> threshold, 1, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``Step``
public func step(_ x: MLXArray, threshold: Float = 0.0) -> MLXArray {
    MLX.where(x .> threshold, 1, 0)
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.elu(x, alpha: 1.67326) * 1.0507
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``SELU``
/// - ``elu(_:alpha:)``
public func selu(_ x: MLXArray) -> MLXArray {
    compiledSelu(x)
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```swift
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``PReLU``
public func prelu(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    compiledPrelu(x, alpha)
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```swift
/// x * MLX.tanh(softPlus(x))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``Mish``
public func mish(_ x: MLXArray) -> MLXArray {
    compiledMish(x)
}

/// Applies the hardswish function, element-wise
///
/// This is:
///
/// ```swift
/// x * MLX.minimum(MLX.maximum(x + 3, 0), 6) / 6
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``HardSwish``
public func hardSwish(_ x: MLXArray) -> MLXArray {
    compiledHardSwish(x)
}

/// Applies the HardTanh function.
///
/// This is (element-wise):
///
/// ```swift
/// maximum(minimum(x, max), min)
/// ```
///
/// - Parameters:
///   - x: input array
///   - min: minimum value
///   - max: maximum value
public func hardTanH(_ x: MLXArray, min: Float = -1, max: Float = 1) -> MLXArray {
    let min = min.asMLXArray(dtype: x.dtype)
    let max = max.asMLXArray(dtype: x.dtype)
    return compiledHardTanh(x, min, max)
}

/// Applies the HardShrink activation function.
///
/// This is (element-wise):
///
/// ```swift
/// if x > lambda {
///     x
/// }  else if x < -lambda {
///     x
/// } else {
///     0
/// }
/// ```
///
/// - Parameters:
///   - x: input array
///   - lambda: lambda value
public func hardShrink(_ x: MLXArray, lambda: Float = 0.5) -> MLXArray {
    let lambda = lambda.asMLXArray(dtype: x.dtype)
    return compiledHardShrink(x, lambda)
}

/// Applies the Softmin function.
///
/// This operation is a numerically stable version of:
///
/// ```swift
///exp(-a) / sum(exp(-a), axis, keepdims: true)
/// ```
///
/// - Parameters:
///   - x: input array
///   - axis: axis to evaluate on
public func softmin(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    softmax(-x, axis: axis)
}

/// Applies the gated linear unit function.
///
/// This function splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
///
/// ### See Also
/// - <doc:activations>
/// - ``glu(_:axis:)``
open class GLU: Module, UnaryLayer {
    public var axis: Int

    public init(axis: Int = -1) {
        self.axis = axis
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        glu(x, axis: axis)
    }
}

/// Applies the element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```swift
/// MLX.sigmoid(x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``sigmoid(_:)``
open class Sigmoid: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        sigmoid(x)
    }
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```swift
/// x * MLX.tanh(softPlus(x))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``mish(_:)``
open class Mish: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        mish(x)
    }
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(x, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``relu(_:)``
open class ReLU: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(x)
    }
}

/// Applies the Leaky Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(negativeSlope * x, x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``leakyRelu(_:negativeSlope:)``
open class LeakyReLU: Module, UnaryLayer {

    public var negativeSlope: Float

    public init(negativeSlope: Float = 0.01) {
        self.negativeSlope = negativeSlope
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        leakyRelu(x, negativeSlope: negativeSlope)
    }
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```swift
/// MLX.minimum(MLX.maximum(x, 0), 6)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``relu6(_:)``
open class ReLU6: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu6(x)
    }
}

/// Applies the squared Rectified Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(x, 0).square()
///
///
/// ### See Also
/// - <doc:activations>
/// - ``reluSquared(_:)``
open class ReLUSquared: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        reluSquared(x)
    }
}

@available(*, deprecated, renamed: "Softmax")
@_documentation(visibility: internal)
open class SoftMax: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softmax(x)
    }
}

/// Applies the Softmax function.
///
/// This is:
///
/// ```swift
/// MLX.softmax(x)
/// ```
///
/// ### See Also
/// - <doc:activations>
open class Softmax: Module, UnaryLayer {
    public var axis: Int

    public init(axis: Int = -1) {
        self.axis = axis
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softmax(x, axis: axis)
    }
}

/// Applies the Softmin function.
///
/// This operation is a numerically stable version of:
///
/// ```swift
/// exp(-a) / sum(exp(-a), axis, keepdims: true)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``softmin(_:axis:)``
open class Softmin: Module, UnaryLayer {
    public var axis: Int

    public init(axis: Int = -1) {
        self.axis = axis
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softmin(x, axis: axis)
    }
}

@available(*, deprecated, renamed: "Softplus")
@_documentation(visibility: internal)
open class SoftPlus: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softPlus(x)
    }
}

/// Applies the Softplus function.
///
/// This is:
///
/// ```swift
/// MLX.logAddExp(x, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``softplus(_:)``
open class Softplus: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softplus(x)
    }
}

@available(*, deprecated, renamed: "Softsign")
@_documentation(visibility: internal)
open class SoftSign: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softsign(x)
    }
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```swift
/// x / (1 + MLX.abs(x))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``softsign(_:)``
open class Softsign: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softsign(x)
    }
}

/// Applies the Softshrink activation function.
///
/// This is (element-wise):
///
/// ```swift
/// if x > lambda {
///     x - lambda
/// }  else if x < -lambda {
///     x + lambda
/// } else {
///     0
/// }
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``softshrink(_:lambda:)``
open class Softshrink: Module, UnaryLayer {
    public var lambda: Float

    public init(lambda: Float = 0.5) {
        self.lambda = lambda
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        softshrink(x, lambda: lambda)
    }
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.which(x .> 0, x, alpha * (exp(x) - 1))
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``elu(_:alpha:)``
open class ELU: Module, UnaryLayer {
    public var alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        elu(x, alpha: alpha)
    }
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.maximum(x, 0.0) + alpha * (MLX.exp(MLX.minimum(x, 0.0) / alpha) - 1)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``celu(_:alpha:)``
open class CELU: Module, UnaryLayer {
    public var alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        celu(x, alpha: alpha)
    }
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```swift
/// x * MLX.sigmoid(x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``silu(_:)``
open class SiLU: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(x)
    }
}

@available(*, deprecated, renamed: "LogSoftmax")
@_documentation(visibility: internal)
open class LogSoftMax: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSoftmax(x)
    }
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```swift
/// x - MLX.logSumExp(x, axis: axis, keepDims: true)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``logSoftmax(_:axis:)``
open class LogSoftmax: Module, UnaryLayer {
    public var axis: Int

    public init(axis: Int = -1) {
        self.axis = axis
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSoftmax(x, axis: axis)
    }
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```swift
/// -softPlus(-x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``logSigmoid(_:)``
open class LogSigmoid: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSigmoid(x)
    }
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```swift
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``prelu(_:alpha:)``
open class PReLU: Module, UnaryLayer {

    public let weight: MLXArray

    public init(count: Int = 1, value: Float = 0.25) {
        self.weight = MLXArray.full([count], values: MLXArray(value))
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        prelu(x, alpha: weight)
    }
}

/// Applies the Gaussian Error Linear Units function.
///
/// There are three variations:
///
/// - ``Approximation/none``
/// - ``Approximation/precise``
/// - ``Approximation/fast``
///
/// ### See Also
/// - <doc:activations>
/// - ``gelu(_:)``
/// - ``geluApproximate(_:)``
/// - ``geluFastApproximate(_:)``
open class GELU: Module, UnaryLayer {

    public enum Approximation: Sendable {
        /// See ``gelu(_:)``
        case none
        /// See ``geluApproximate(_:)``
        case precise
        /// Alias for ``precise`` -- see ``geluApproximate(_:)``
        case tanh
        /// See ``geluFastApproximate(_:)``
        case fast
    }

    public let approximation: Approximation

    public init(approximation: Approximation = .none) {
        self.approximation = approximation
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch approximation {
        case .none:
            gelu(x)
        case .precise, .tanh:
            geluApproximate(x)
        case .fast:
            geluFastApproximate(x)
        }
    }
}

/// Applies the hyperbolic tangent function
open class Tanh: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        tanh(x)
    }
}

/// Applies the hardswish function, element-wise
///
/// This is:
///
/// ```swift
/// x * MLX.minimum(MLX.maximum(x + 3, 0), 6) / 6
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``hardSwish(_:)``
open class HardSwish: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        hardSwish(x)
    }
}

/// Applies the HardTanh function.
///
/// This is (element-wise):
///
/// ```swift
/// maximum(minimum(x, max), min)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``hardTanH(_:min:max:)``
open class HardTanh: Module, UnaryLayer {
    public var min: Float
    public var max: Float

    public init(min: Float = -1, max: Float = 1) {
        self.min = min
        self.max = max
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        hardTanH(x, min: min, max: max)
    }
}

/// Applies the HardShrink activation function.
///
/// This is (element-wise):
///
/// ```swift
/// if x > lambda {
///     x
/// }  else if x < -lambda {
///     x
/// } else {
///     0
/// }
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``hardShrink(_:lambda:)``
open class HardShrink: Module, UnaryLayer {
    public var lambda: Float

    public init(lambda: Float = 0.5) {
        self.lambda = lambda
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        hardShrink(x, lambda: lambda)
    }
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```swift
/// MLX.where(x .> threshold, 1, 0)
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``step(_:threshold:)``
open class Step: Module, UnaryLayer {

    public var threshold: Float

    public init(threshold: Float = 0.0) {
        self.threshold = threshold
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        step(x, threshold: threshold)
    }
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```swift
/// MLX.elu(x, alpha: 1.67326) * 1.0507
/// ```
///
/// ### See Also
/// - <doc:activations>
/// - ``selu(_:)``
open class SELU: Module, UnaryLayer {
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        selu(x)
    }
}

// MARK: - Compiled Activation Functions

private let compiledLeakyRelu: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, negativeSlope in
        maximum(negativeSlope * x, x)
    }
}()

private let compiledElu: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, alpha in
        which(x .> 0, x, alpha * (MLX.exp(x) - 1))
    }
}()

private let compiledRelu6: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        minimum(maximum(x, 0), 6)
    }
}()

private let compiledSoftsign: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        x / (1 + abs(x))
    }
}()

private let compiledSoftshrink: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, lambda in
        which(abs(x) .> lambda, x - sign(x) * lambda, 0)
    }
}()

private let compiledCelu: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, alpha in
        maximum(x, 0.0) + alpha * (exp(minimum(x, 0.0) / alpha) - 1)
    }
}()

private let compiledSilu: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        x * sigmoid(x)
    }
}()

private let compiledLogSigmoid: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        -softplus(-x)
    }
}()

private let compiledGelu: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        x * (1 + erf(x / sqrt(2))) / 2
    }
}()

private let compiledGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x ** 3)))
    }
}()

private let compiledGeluFastApproximate: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        x * sigmoid(1.702 * x)
    }
}()

private let compiledSelu: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        elu(x, alpha: 1.67326) * 1.0507
    }
}()

private let compiledPrelu: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, alpha in
        maximum(0, x) + alpha * minimum(0, x)
    }
}()

private let compiledMish: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        x * tanh(softplus(x))
    }
}()

private let compiledHardSwish: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        let maxXPlus3 = maximum(x + 3, 0)
        return x * minimum(maxXPlus3, 6) / 6
    }
}()

private let compiledHardTanh: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, min, max in
        minimum(maximum(x, min), max)
    }
}()

private let compiledHardShrink: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, lambda in
        which(abs(x) .> lambda, x, 0)
    }
}()

private let compiledRelu: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        maximum(x, 0)
    }
}()

private let compiledReluSquared: @Sendable (MLXArray) -> MLXArray = {
    compile(shapeless: true) { x in
        return relu(x).square()
    }
}()
