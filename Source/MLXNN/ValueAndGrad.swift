// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Transform the passed function `f(Model, [MLXArray])` to a function that computes the
/// gradients of ``f`` wrt the model's trainable parameters and also its value.
///
/// - Parameters:
///   - model: model to apply parameters to
///   - f: function to compute the gradients for
/// - Returns: function that returns the value of `f()` and the gradient of
///     the parameters of the model
///
/// ### See Also
/// - ``valueAndGrad(model:_:)-12a2c``
public func valueAndGrad<Model: Module>(
    model: Model, _ f: @escaping (Model, [MLXArray]) -> [MLXArray]
) -> (Model, [MLXArray]) -> ([MLXArray], ModuleParameters) {

    func inner(parameters: ModuleParameters, arrays: [MLXArray]) -> [MLXArray] {
        model.update(parameters: parameters)
        return f(model, arrays)
    }

    let vg = valueAndGrad(inner)

    func wrapped(model: Model, arrays: [MLXArray]) -> ([MLXArray], ModuleParameters) {
        vg(model.trainableParameters(), arrays)
    }

    return wrapped
}

/// Transform the passed function `f(Model, MLXArray)` to a function that computes the
/// gradients of ``f`` wrt the model's trainable parameters and also its value.
///
/// - Parameters:
///   - model: model to apply parameters to
///   - f: function to compute the gradients for
/// - Returns: function that returns the value of `f()` and the gradient of
///     the parameters of the model
///
/// ### See Also
/// - ``valueAndGrad(model:_:)-12a2c``
public func valueAndGrad<Model: Module>(model: Model, _ f: @escaping (Model, MLXArray) -> MLXArray)
    -> (Model, MLXArray) -> (MLXArray, ModuleParameters)
{

    func inner(parameters: ModuleParameters, arrays: [MLXArray]) -> [MLXArray] {
        model.update(parameters: parameters)
        return [f(model, arrays[0])]
    }

    let vg = valueAndGrad(inner)

    func wrapped(model: Model, array: MLXArray) -> (MLXArray, ModuleParameters) {
        let (v, g) = vg(model.trainableParameters(), [array])
        return (v[0], g)
    }

    return wrapped
}

/// Transform the passed function `f(Model, MLXArray, MLXArray)` to a
/// function that computes the gradients of ``f`` wrt the model's trainable
/// parameters and also its value.
///
/// For example:
///
/// ```swift
/// class M : Module, UnaryLayer { ... }
///
/// // create the model and realize the parameters
/// let m = M()
/// MLX.eval(m.parameters())
///
/// // input and targets
/// let x = MLXArray(0 ..< 5, [1, 5])
/// let y = MLXArray(0 ..< 5)
///
/// func loss(model: M, x: MLXArray, y: MLXArray) -> MLXArray {
///     crossEntropy(logits: model(x), targets: y, reduction: .mean)
/// }
///
/// let lg = valueAndGrad(model: m, loss)
///
/// // loss is a scalar MLXArray with the loss value and
/// // grads are something that could be fed to an Optimizer
/// // to produce an update for the model parameters.
/// let (loss, grads) = lg(m, x, y)
/// ```
///
/// - Parameters:
///   - model: model to apply parameters to
///   - f: function to compute the gradients for
/// - Returns: function that returns the value of `f()` and the gradient of
///     the parameters of the model
public func valueAndGrad<Model: Module>(
    model: Model, _ f: @escaping (Model, MLXArray, MLXArray) -> MLXArray
) -> (Model, MLXArray, MLXArray) -> (MLXArray, ModuleParameters) {

    func inner(parameters: ModuleParameters, arrays: [MLXArray]) -> [MLXArray] {
        model.update(parameters: parameters)
        return [f(model, arrays[0], arrays[1])]
    }

    let vg = valueAndGrad(inner)

    // outer function
    func wrapped(model: Model, a1: MLXArray, a2: MLXArray) -> (MLXArray, ModuleParameters) {
        let (v, g) = vg(model.trainableParameters(), [a1, a2])
        return (v[0], g)
    }

    return wrapped
}
