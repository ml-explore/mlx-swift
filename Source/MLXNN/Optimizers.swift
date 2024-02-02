// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Public interface for all Optimizer types.
///
/// ### See Also
/// - <doc:optimizers>
/// - ``OptimizerBase``
public protocol Optimizer {

    /// Apply the gradients to the parameters of the model and update the model with the new parameters.
    func update(model: Module, gradients: ModuleParameters)

    /// Return any parameters that should be passed to `eval()`
    func parameters() -> [MLXArray]
}

/// The base class for all optimizers. It allows us to implement an optimizer on a per-parameter basis
/// and apply it to a parameter tree.
///
/// Subclasses need to implement:
/// - `func newState(gradient: MLXArray) -> State`
/// - `func parameters() -> ModuleParameters`
/// - `func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (MLXArray, State)`
///
/// ### See Also
/// - <doc:optimizers>
public class OptimizerBase<State>: Optimizer {

    /// Stores a `State` value in a structure that matches the model parameters
    var state = NestedDictionary<String, State>()

    /// Subclasses must implment this to create a new `State` when needed.  This is called
    /// with the gradient in question.
    ///
    /// For example:
    ///
    /// ```swift
    /// override func newState(gradient: MLXArray) -> MLXArray {
    ///     MLXArray.zeros(like: gradient)
    /// }
    /// ```
    func newState(gradient: MLXArray) -> State {
        fatalError("newState() not implemented \(type(of: self))")
    }

    /// Return any parameters that should be passed to `eval()`.
    ///
    /// For example if `State` is `MLXArray`:
    ///
    /// ```swift
    /// public override func parameters() -> [MLXArray] {
    ///     state.flattened().map { $0.1 }
    /// }
    /// ```
    public func parameters() -> [MLXArray] {
        fatalError("parameters() not implemented \(type(of: self))")
    }

    final public func update(model: Module, gradients: ModuleParameters) {
        model.update(parameters: apply(gradients: gradients, model: model))
    }

    /// Evaluate `applySingle(gradient: MLXArray, parameter: MLXArray, state: State)` for
    /// each parameter and update `self.state` and produce new `ModuleParameters`.
    final func apply(gradients: ModuleParameters, model: Module) -> ModuleParameters {
        let (p, s) = gradients.mapValues(model.parameters(), state) { gradient, parameter, state in
            // handle optionality of the visitor params
            applySingle(
                gradient: gradient, parameter: parameter!,
                state: state ?? newState(gradient: gradient))
        }
        self.state = s
        return p
    }

    func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (MLXArray, State) {
        fatalError("applySingle() not implemented \(type(of: self))")
    }
}

/// Convenience subclass of OptimizerBase that provides `MLXArray` State.
///
/// ### See Also
/// - <doc:optimizers>
public class OptimizerBaseArrayState: OptimizerBase<MLXArray> {

    override func newState(gradient: MLXArray) -> MLXArray {
        MLXArray.zeros(like: gradient)
    }

    public override func parameters() -> [MLXArray] {
        state.flattened().map { $0.1 }
    }
}

/// Stochastic gradient descent optimizer.
///
/// ### See Also
/// - <doc:optimizers>
public class SGD: OptimizerBaseArrayState {

    var learningRate: Float
    var momentum: Float = 0
    var weightDecay: Float = 0
    var dampening: Float = 0
    var nesterov = false

    /// Initialize the SGD optimizer.
    ///
    /// `learningRate` is the only required parameter and `0.1` might be a reasonable number to try.
    ///
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - momentum: momentum strength
    ///   - weightDecay: weight decay (L2 penalty)
    ///   - dampening: dampening for momentum
    ///   - nesterov: enables Nesterov momentum
    public init(
        learningRate: Float, momentum: Float = 0, weightDecay: Float = 0, dampening: Float = 0,
        nesterov: Bool = false
    ) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.dampening = dampening
        self.nesterov = nesterov
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        if momentum <= 0 {
            return (parameter - learningRate * gradient, state)
        }

        var gradient = gradient
        if weightDecay != 0 {
            gradient = gradient + weightDecay * parameter
        }

        var v = momentum * state
        if dampening > 0 {
            v = v + (1 - dampening) * gradient
        } else {
            v = v + gradient
        }

        let update: MLXArray
        if nesterov {
            update = gradient + momentum * v
        } else {
            update = v
        }

        return (parameter - learningRate * update, v)
    }
}

/// The RMSprop optimizer [1].
///
/// [1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning
///
/// ### See Also
/// - <doc:optimizers>
public class RMSprop: OptimizerBaseArrayState {

    var learningRate: Float
    var alpha: Float = 0.99
    var eps: Float = 1e-8

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - alpha: the smoothing constant
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    public init(learningRate: Float, alpha: Float = 0.99, eps: Float = 1e-8) {
        precondition(alpha >= 0)
        precondition(eps >= 0)

        self.learningRate = learningRate
        self.alpha = alpha
        self.eps = eps
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let v = alpha * state + (1 - alpha) * square(gradient)
        return (parameter - learningRate * gradient / (sqrt(v) + eps), v)
    }
}

/// The Adagrad optimizer [1].
///
/// Our Adagrad implementation follows the original paper. In detail,
///
/// [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online learning and stochastic optimization. JMLR 2011.
///
/// ### See Also
/// - <doc:optimizers>
public class AdaGrad: OptimizerBaseArrayState {

    var learningRate: Float
    var eps: Float = 1e-8

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    public init(learningRate: Float, eps: Float = 1e-8) {
        precondition(eps >= 0)

        self.learningRate = learningRate
        self.eps = eps
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let v = state + square(gradient)
        return (parameter - learningRate * gradient / (sqrt(v) + eps), v)
    }
}

/// The AdaDelta optimizer with a learning rate [1].
///
/// Our AdaDelta implementation follows the original paper. In detail,
///
/// [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
///
/// ### See Also
/// - <doc:optimizers>
public class AdaDelta: OptimizerBase<(MLXArray, MLXArray)> {

    typealias State = (MLXArray, MLXArray)

    var learningRate: Float
    var rho: Float = 0.99
    var eps: Float = 1e-6

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - rho: the coefficient used for computing a running average of squared gradients
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    public init(learningRate: Float, rho: Float = 0.9, eps: Float = 1e-6) {
        precondition(rho >= 0)
        precondition(eps >= 0)

        self.learningRate = learningRate
        self.rho = rho
        self.eps = eps
    }

    override func newState(gradient: MLXArray) -> State {
        (MLXArray.zeros(like: gradient), MLXArray.zeros(like: gradient))
    }

    public override func parameters() -> [MLXArray] {
        state.flattened().flatMap { [$0.1.0, $0.1.1] }
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        var v = state.0
        var u = state.1

        v = rho * v + (1 - rho) * square(gradient)
        let d = sqrt(u + eps) / sqrt(v + eps) * gradient
        u = rho * u + (1 - rho) * square(d)

        return (parameter - learningRate * d, (v, u))
    }
}

/// The Adam optimizer [1].
///
/// Our Adam implementation follows the original paper and omits the bias
/// correction in the first and second moment estimates. In detail,
///
/// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
///
/// ### See Also
/// - <doc:optimizers>
public class Adam: OptimizerBase<(MLXArray, MLXArray)> {

    typealias State = (MLXArray, MLXArray)

    var learningRate: Float
    var betas: (Float, Float) = (0.9, 0.999)
    var eps: Float = 1e-8

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - betas: coefficients used for computing running averages of the gradient and its square
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
    }

    override func newState(gradient: MLXArray) -> State {
        (gradient, square(gradient))
    }

    public override func parameters() -> [MLXArray] {
        state.flattened().flatMap { [$0.1.0, $0.1.1] }
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        let (b1, b2) = betas

        var m = state.0
        var v = state.1

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * square(gradient)

        return (parameter - learningRate * m / (sqrt(v) + eps), (m, v))
    }
}

/// The AdamW optimizer [1].
///
/// Following the above convention, in contrast with [1], we do not use bias
/// correction in the first and second moments for AdamW. We update the weights
/// with a `weightDecay` lambda value:
///
/// [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay regularization. ICLR 2019.
///
/// ### See Also
/// - <doc:optimizers>
public class AdamW: Adam {

    var weightDecay: Float = 0.01

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - betas: coefficients used for computing running averages of the gradient and its square
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    ///   - weightDecay:the weight decay
    public init(
        learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8,
        weightDecay: Float = 0.01
    ) {
        self.weightDecay = weightDecay
        super.init(learningRate: learningRate, betas: betas, eps: eps)
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: Adam.State) -> (
        MLXArray, Adam.State
    ) {
        super.applySingle(
            gradient: gradient, parameter: parameter * (1 - learningRate * weightDecay),
            state: state)
    }
}

/// The Adamax optimizer, a variant of Adam based on the infinity norm [1].
///
/// Our Adam implementation follows the original paper and omits the bias
/// correction in the first and second moment estimates. In detail,
///
/// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
///
/// ### See Also
/// - <doc:optimizers>
public class Adamax: OptimizerBase<(MLXArray, MLXArray)> {

    typealias State = (MLXArray, MLXArray)

    var learningRate: Float
    var betas: (Float, Float) = (0.9, 0.999)
    var eps: Float = 1e-8

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - betas: coefficients used for computing running averages of the gradient and its square
    ///   - eps: the epsilon added to the denominator to improve numerical stability
    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
    }

    override func newState(gradient: MLXArray) -> State {
        (MLXArray.zeros(like: gradient), MLXArray.zeros(like: gradient))
    }

    public override func parameters() -> [MLXArray] {
        state.flattened().flatMap { [$0.1.0, $0.1.1] }
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        let (b1, b2) = betas

        var m = state.0
        var v = state.1

        m = b1 * m + (1 - b1) * gradient
        v = maximum(b2 * v, abs(gradient))

        return (parameter - learningRate * m / (v + eps), (m, v))
    }
}

/// The Lion optimizer [1].
///
/// Since updates are computed through the sign operation, they tend to
/// have larger norm than for other optimizers such as SGD and Adam.
/// We recommend a learning rate that is 3-10x smaller than AdamW and a
/// weight decay 3-10x larger than AdamW to maintain the strength
/// `(lr * wd)`. Our Lion implementation follows the original paper. In
/// detail,
///
/// [1]: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv preprint arXiv:2302.06675.
///
/// ### See Also
/// - <doc:optimizers>
public class Lion: OptimizerBaseArrayState {

    var learningRate: Float
    var betas: (Float, Float) = (0.9, 0.999)
    var weightDecay: Float = 0.0

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - betas: coefficients used for computing running averages of the gradient and its square
    ///   - weightDecay:the weight decay
    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), weightDecay: Float = 0.0)
    {
        self.learningRate = learningRate
        self.betas = betas
        self.weightDecay = weightDecay
    }

    override func newState(gradient: MLXArray) -> MLXArray {
        gradient
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let (b1, b2) = betas
        var m = state
        var parameter = parameter

        let c = b1 * m + (1 - b1) * gradient
        m = b2 * m + (1 - b2) * gradient
        if weightDecay > 0 {
            parameter = (1 - learningRate * weightDecay) * parameter
        }

        return (parameter - learningRate * sign(c), m)
    }
}

/// The Adafactor optimizer.
///
/// Our Adafactor implementation follows the original paper: `Adafactor:
/// Adaptive Learning Rates with Sublinear Memory Cost
/// <https://arxiv.org/abs/1804.04235>
///
/// ### See Also
/// - <doc:optimizers>
public class Adafactor: OptimizerBase<Adafactor.State> {

    var learningRate: Float? = nil
    var eps: (Float, Float) = (1e-30, 1e-3)
    var clipThreshold: Float = 1
    var decayRate: Float = -0.8
    var beta1: Float? = nil
    var weightDecay: Float = 0
    var scaleParameter = true
    var relativeStep = true
    var warmupInit = false

    public struct State {
        var step = 0
        var expAvgSqRow: MLXArray? = nil
        var expAvgSqCol: MLXArray? = nil
        var expAvgSq: MLXArray? = nil
        var expAvg: MLXArray? = nil
    }

    /// Initialize the optimizer.
    /// - Parameters:
    ///   - learningRate: the learning rate
    ///   - eps: the first term is added to the square of the gradients to improve numerical
    ///   stability and the second term is used for parameter scaling if `parameterScale` is `true`
    ///   - clipThreshold: clips the unscaled update
    ///   - decayRate: ceofficient for the running average of the squared gradient
    ///   - beta1: if set then the first moment will be used
    ///   - weightDecay: the weight decay
    ///   - scaleParameter: if `true` the `learningRate` will be scaled by `max(eps.0, RMS(parameter))`
    ///   - relativeStep: if `true` the `learningRate` will be ignored and the relative step size will be computed
    ///   - warmupInit: if `true` the relative step size will be calculated by the current step
    public init(
        learningRate: Float? = nil, eps: (Float, Float) = (1e-30, 1e-3), clipThreshold: Float = 1,
        decayRate: Float = -0.8, beta1: Float? = nil, weightDecay: Float = 0,
        scaleParameter: Bool = true, relativeStep: Bool = true, warmupInit: Bool = false
    ) {
        precondition(learningRate != nil || relativeStep)

        self.learningRate = learningRate
        self.eps = eps
        self.clipThreshold = clipThreshold
        self.decayRate = decayRate
        self.beta1 = beta1
        self.weightDecay = weightDecay
        self.scaleParameter = scaleParameter
        self.relativeStep = relativeStep
        self.warmupInit = warmupInit
    }

    override func newState(gradient: MLXArray) -> State {
        State()
    }

    public override func parameters() -> [MLXArray] {
        state.flattened().flatMap {
            [$0.1.expAvgSqRow, $0.1.expAvgSqCol, $0.1.expAvgSq, $0.1.expAvg]
                .compactMap { $0 }
        }
    }

    func rms(_ inputs: MLXArray) -> MLXArray {
        sqrt(mean(square(inputs)))
    }

    func computeLearningRate(step: Int, parameterRMS: MLXArray) -> MLXArray {
        let relativeStepSize: Float
        if relativeStep {
            let minStep = warmupInit ? 1e-6 * Float(step) : 1e-2
            relativeStepSize = min(minStep, 1 / sqrt(Float(step)))
        } else {
            // the precondition verified this
            relativeStepSize = learningRate!
        }

        var parameterScale = MLXArray(1.0)
        if scaleParameter {
            parameterScale = maximum(eps.1, parameterRMS)
        }

        return parameterScale * relativeStepSize
    }

    func approvateExpMovingAverage(expAvgSqRow: MLXArray, expAvgSqCol: MLXArray) -> MLXArray {
        let rFactor = rsqrt(expAvgSqRow / mean(expAvgSqRow, axis: -1, keepDims: true))
        let cFactor = rsqrt(expAvgSqCol)
        return matmul(rFactor.expandedDimensions(axis: -1), cFactor.expandedDimensions(axis: 0))
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        var state = state
        state.step += 1

        let gradientShape = gradient.shape
        let factored = gradientShape.count >= 2
        let step = state.step

        let parameterRMS = rms(parameter)
        let learningRate = computeLearningRate(step: step, parameterRMS: parameterRMS)
        let beta2 = 1.0 - pow(Float(step), decayRate)

        var update = square(gradient) + eps.0

        if factored {
            var expAvgSqRow =
                state.expAvgSqRow
                ?? MLXArray.zeros(gradient.shape.dropLast(), dtype: gradient.dtype)
            var expAvgSqCol =
                state.expAvgSqCol
                ?? MLXArray.zeros(Array(gradient.shape.dropFirst()), dtype: gradient.dtype)

            expAvgSqRow = (beta2 * expAvgSqRow) + (1 - beta2) * mean(update, axis: -1)
            expAvgSqCol = (beta2 * expAvgSqCol) + (1 - beta2) * mean(update, axis: -2)

            state.expAvgSqRow = expAvgSqRow
            state.expAvgSqCol = expAvgSqCol

            update = approvateExpMovingAverage(expAvgSqRow: expAvgSqRow, expAvgSqCol: expAvgSqCol)
            update = update * gradient
        } else {
            var expAvgSq = state.expAvgSq ?? MLXArray.zeros(like: gradient)
            expAvgSq = (beta2 * expAvgSq) + (1 - beta2) * update
            state.expAvgSq = expAvgSq
            update = rsqrt(expAvgSq) * gradient
        }

        update = update / maximum(1.0, rms(update) / clipThreshold)
        update = learningRate * update

        if let beta1 {
            var expAvg = state.expAvg ?? MLXArray.zeros(like: gradient)
            expAvg = (beta1 * expAvg) + (1 - beta1) * update
            state.expAvg = expAvg
            update = expAvg
        }

        var parameter = parameter
        if weightDecay != 0 {
            parameter = parameter + parameter * (-weightDecay * learningRate)
        }

        return (parameter - update, state)
    }
}
