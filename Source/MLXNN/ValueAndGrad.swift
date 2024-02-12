// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Transform the passed function `f(Model, MLXArray, MLXArray)` to a
/// function that computes the gradients of `f` with regard to the model's trainable
/// parameters and also its value.
///
/// For example:
///
/// ```swift
/// class M : Module, UnaryLayer { ... }
///
/// // create the model and realize the parameters
/// let m = M()
/// MLX.eval(m)
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
/// ### Other Arguments
///
/// If other arguments to the loss function are needed there are two variants of `valueAndGrad()` that
/// can be used to build these:
///
/// - ``valueAndGrad(model:_:)-548r7`` -- passing only `[MLXArray]` and returning `[MLXArray]`
/// - ``valueAndGrad(model:_:)-45dg5``-- passing any arguments and returning `[MLXArray]`
///
/// Prefer the former as it can cache the value of the underlying `valueAndGrad()` call while
/// the latter must rebuild it for each call.
///
/// - Parameters:
///   - model: model to apply parameters to
///   - f: function to compute the gradients for
/// - Returns: function that returns the value of `f()` and the gradient of
///     the parameters of the model
///
/// ### See Also
/// - <doc:training>
public func valueAndGrad<Model: Module>(
    model: Model, _ f: @escaping (Model, MLXArray, MLXArray) -> MLXArray
) -> (Model, MLXArray, MLXArray) -> (MLXArray, ModuleParameters) {

    // because we know the structure of the parameters and they are all
    // arrays we can capture the result of the valueAndGrad and use it
    // over and over
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

/// Variant of `valueAndGrad(model:_:)` that can be used to pass an arbitrary number of `MLXArray` to
/// a loss function.
///
/// For example, given a loss function:
///
/// ```swift
/// func loss(model: Model, a: MLXArray, b: MLXArray, c: MLXArray) -> MLXArray {
///     ...
/// }
/// ```
///
/// it can be wrapped as:
///
/// ```swift
/// let vg = valueAndGrad(model: m) { model, args in
///     // wrap the result in an array
///     [loss(model: model, a: args[0], b: args[1], c: args[2])]
/// }
/// ```
///
/// ### See Also
/// - <doc:training>
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

/// Variant of `valueAndGrad(model:_:)` that can be used to pass an arbitrary parameters to a loss
/// function.
///
/// > Prefer the other `valueAndGrad(model:_:)` functions if the arguments are all `MLXArray`.
/// This variant requires re-constructing the gradient function per call.
///
/// For example, given a loss function:
///
/// ```swift
/// func loss(model: Model, a: MLXArray, x: Int) -> MLXArray {
///     ...
/// }
/// ```
///
/// it can be wrapped as:
///
/// ```swift
/// let vg = valueAndGrad(model: m) { (model: Module, args: (MLXArray, Int)) in
///     // wrap the result in an array
///     [loss(model: model, a: args.0, x: args.1)]
/// }
/// ```
///
/// ### See Also
/// - <doc:training>
/// - ``valueAndGrad(model:_:)-12a2c``
public func valueAndGrad<Model: Module, Arguments>(
    model: Model, _ f: @escaping (Model, Arguments) -> [MLXArray]
) -> (Model, Arguments) -> ([MLXArray], ModuleParameters) {

    func wrapped(model: Model, arguments: Arguments) -> ([MLXArray], ModuleParameters) {

        func inner(parameters: ModuleParameters, _ extra: ()) -> [MLXArray] {
            model.update(parameters: parameters)
            return f(model, arguments)
        }

        let vg = valueAndGrad(inner)

        let (v, g) = vg(model.trainableParameters(), ())
        return (v, g)
    }

    return wrapped
}
