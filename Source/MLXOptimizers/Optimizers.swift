// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Public interface for all Optimizer types.
///
/// ### See Also
/// - <doc:MLXOptimizers>
/// - ``OptimizerBase``
public protocol Optimizer: Updatable, Evaluatable {

    /// Apply the gradients to the parameters of the model and update the model with the new parameters.
    func update(model: Module, gradients: ModuleParameters)
}

/// The base class for all optimizers. It allows us to implement an optimizer on a per-parameter basis
/// and apply it to a parameter tree.
///
/// Subclasses need to implement:
/// - `func newState(parameter: MLXArray) -> State`
/// - `func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (MLXArray, State)`
///
/// ### See Also
/// - <doc:MLXOptimizers>
open class OptimizerBase<State: Updatable>: Optimizer {

    /// Stores a `State` value in a structure that matches the model parameters
    var stateStorage = NestedDictionary<String, State>()

    /// Subclasses must implment this to create a new `State` when needed.  This is called
    /// for each parameter in the model.
    ///
    /// For example:
    ///
    /// ```swift
    /// override func newState(parameter: MLXArray) -> MLXArray {
    ///     MLXArray.zeros(like: gradient)
    /// }
    /// ```
    open func newState(parameter: MLXArray) -> State {
        fatalError("newState() not implemented \(type(of: self))")
    }

    open func innerState() -> [MLXArray] {
        stateStorage
            .flattenedValues()
            .flatMap { $0.innerState() }
    }

    final public func update(model: Module, gradients: ModuleParameters) {
        model.update(parameters: apply(gradients: gradients, modelParameters: model.parameters()))
    }

    /// Evaluate `applySingle(gradient: MLXArray, parameter: MLXArray, state: State)` for
    /// each parameter and update `self.state` and produce new `ModuleParameters`.
    final func apply(gradients: ModuleParameters, modelParameters: ModuleParameters)
        -> ModuleParameters
    {
        let (p, s) = gradients.mapValues(modelParameters, stateStorage) {
            gradient, parameter, state in
            // handle optionality of the visitor params
            applySingle(
                gradient: gradient, parameter: parameter!,
                state: state ?? newState(parameter: parameter!))
        }
        self.stateStorage = s
        return p
    }

    open func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        fatalError("applySingle() not implemented \(type(of: self))")
    }
}

/// Convenience subclass of OptimizerBase that provides `MLXArray` State.
///
/// ### See Also
/// - <doc:MLXOptimizers>
open class OptimizerBaseArrayState: OptimizerBase<MLXArray> {
    override open func newState(parameter: MLXArray) -> MLXArray {
        MLXArray.zeros(like: parameter)
    }
}

/// State container for ``OptimizerBase`` holding a tuple of `MLXArray`.
public struct TupleState: Updatable {
    let values: (MLXArray, MLXArray)

    init(_ values: (MLXArray, MLXArray)) {
        self.values = values
    }

    init(_ a: MLXArray, _ b: MLXArray) {
        self.values = (a, b)
    }

    init(zeros array: MLXArray) {
        self.values = (MLXArray.zeros(like: array), MLXArray.zeros(like: array))
    }

    public func innerState() -> [MLXArray] {
        [values.0, values.1]
    }
}

/// Stochastic gradient descent optimizer.
///
/// ### See Also
/// - <doc:MLXOptimizers>
open class SGD: OptimizerBaseArrayState {

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

    override open func newState(parameter: MLXArray) -> MLXArray {
        MLXArray.zeros(like: parameter)
    }

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        var gradient = gradient
        if weightDecay != 0 {
            gradient = gradient + weightDecay * parameter
        }

        if momentum <= 0 {
            return (parameter - learningRate * gradient, state)
        }

        var v = state
        if dampening > 0 {
            v = v * momentum
            v = v + (1 - dampening) * gradient
        } else {
            v = v * momentum
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
/// - <doc:MLXOptimizers>
open class RMSprop: OptimizerBaseArrayState {

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

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
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
/// - <doc:MLXOptimizers>
open class AdaGrad: OptimizerBaseArrayState {

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

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
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
/// - <doc:MLXOptimizers>
open class AdaDelta: OptimizerBase<TupleState> {

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

    override open func newState(parameter: MLXArray) -> TupleState {
        TupleState(zeros: parameter)
    }

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: TupleState) -> (
        MLXArray, TupleState
    ) {
        var (v, u) = state.values

        v = rho * v + (1 - rho) * square(gradient)
        let d = sqrt(u + eps) / sqrt(v + eps) * gradient
        u = rho * u + (1 - rho) * square(d)

        return (parameter - learningRate * d, TupleState(v, u))
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
/// - <doc:MLXOptimizers>
open class Adam: OptimizerBase<TupleState> {

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

    override open func newState(parameter: MLXArray) -> TupleState {
        TupleState(zeros: parameter)
    }

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: TupleState) -> (
        MLXArray, TupleState
    ) {
        let (b1, b2) = betas

        var (m, v) = state.values

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * square(gradient)

        return (parameter - learningRate * m / (sqrt(v) + eps), TupleState(m, v))
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
/// - <doc:MLXOptimizers>
open class AdamW: Adam {

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

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: TupleState) -> (
        MLXArray, TupleState
    ) {
        return super.applySingle(
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
/// - <doc:MLXOptimizers>
open class Adamax: OptimizerBase<TupleState> {

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

    override open func newState(parameter: MLXArray) -> TupleState {
        TupleState(zeros: parameter)
    }

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: TupleState) -> (
        MLXArray, TupleState
    ) {
        let (b1, b2) = betas

        var (m, v) = state.values

        m = b1 * m + (1 - b1) * gradient
        v = maximum(b2 * v, abs(gradient))

        return (parameter - learningRate * m / (v + eps), TupleState(m, v))
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
/// - <doc:MLXOptimizers>
open class Lion: OptimizerBaseArrayState {

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

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
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
/// - <doc:MLXOptimizers>
open class Adafactor: OptimizerBase<Adafactor.State> {

    var learningRate: Float? = nil
    var eps: (Float, Float) = (1e-30, 1e-3)
    var clipThreshold: Float = 1
    var decayRate: Float = -0.8
    var beta1: Float? = nil
    var weightDecay: Float = 0
    var scaleParameter = true
    var relativeStep = true
    var warmupInit = false

    public struct State: Updatable {
        var step = MLXArray(0)
        var expAvgSqRow: MLXArray? = nil
        var expAvgSqCol: MLXArray? = nil
        var expAvgSq: MLXArray? = nil
        var expAvg: MLXArray? = nil

        public func innerState() -> [MLXArray] {
            [expAvgSqRow, expAvgSqCol, expAvgSq, expAvg].compactMap { $0 }
        }
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

    override open func newState(parameter: MLXArray) -> State {
        var s = State()
        if parameter.ndim >= 2 {
            let shape = parameter.shape
            let dtype = parameter.dtype

            s.expAvgSqRow = MLXArray.zeros(shape.dropLast(), dtype: dtype)
            s.expAvgSqCol = MLXArray.zeros(shape.dropLast(2) + [shape.last!], dtype: dtype)
        } else {
            s.expAvgSq = MLXArray.zeros(like: parameter)
        }

        if beta1 != nil {
            s.expAvg = MLXArray.zeros(like: parameter)
        }

        return s
    }

    func rms(_ inputs: MLXArray) -> MLXArray {
        sqrt(mean(square(inputs)))
    }

    func computeLearningRate(step: MLXArray, parameterRMS: MLXArray) -> MLXArray {
        let relativeStepSize: MLXArray
        if relativeStep {
            let minStep = warmupInit ? 1e-6 * step : MLXArray(1e-2)
            relativeStepSize = minimum(minStep, 1 / sqrt(step))
        } else {
            // the precondition verified this
            relativeStepSize = MLXArray(learningRate!)
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

    override open func applySingle(gradient: MLXArray, parameter: MLXArray, state: State) -> (
        MLXArray, State
    ) {
        var state = state
        state.step = state.step + 1

        let gradientShape = gradient.shape
        let factored = gradientShape.count >= 2
        let step = state.step

        let parameterRMS = rms(parameter)
        let learningRate = computeLearningRate(step: step, parameterRMS: parameterRMS)
        let beta2 = 1.0 - pow(step, decayRate)

        var update = square(gradient) + eps.0

        if factored {
            var expAvgSqRow = state.expAvgSqRow!
            var expAvgSqCol = state.expAvgSqCol!

            expAvgSqRow = (beta2 * expAvgSqRow) + (1 - beta2) * mean(update, axis: -1)
            expAvgSqCol = (beta2 * expAvgSqCol) + (1 - beta2) * mean(update, axis: -2)

            state.expAvgSqRow = expAvgSqRow
            state.expAvgSqCol = expAvgSqCol

            update = approvateExpMovingAverage(expAvgSqRow: expAvgSqRow, expAvgSqCol: expAvgSqCol)
            update = update * gradient
        } else {
            var expAvgSq = state.expAvgSq!
            expAvgSq = (beta2 * expAvgSq) + (1 - beta2) * update
            state.expAvgSq = expAvgSq
            update = rsqrt(expAvgSq) * gradient
        }

        update = update / maximum(1.0, rms(update) / clipThreshold)
        update = learningRate * update

        if let beta1 {
            var expAvg = state.expAvg!
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
