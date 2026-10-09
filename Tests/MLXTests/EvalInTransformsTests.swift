// Copyright © 2026 Apple Inc.

import Foundation
import MLXNN
import Testing

@testable import MLX

// MARK: - Support

/// Results of evaluating inside a function transformation.
///
/// MLX core (`eval_impl` in `mlx/transforms.cpp`) distinguishes:
///
/// - `vjp` / `jvp` (and so `grad`, `valueAndGrad`, custom VJPs): tracers are
///   real arrays with primitives, so a *synchronous* eval of an intermediate is
///   allowed and the graph is retained for the backward pass.  An
///   *asynchronous* eval of a tracer is rejected.  See `test_eval_in_grad` and
///   `test_async_eval_in_trace` in the python tests.
/// - `compile` / `vmap`: tracing runs on placeholder inputs with no primitive,
///   so evaluating anything derived from them is rejected by either kind of
///   eval.
///
/// Several of the failure modes here abort the process (the default error
/// handler calls `fatalError`) or crash outright (reading the data pointer of
/// an array whose eval failed), so those cases run as exit tests: the body
/// runs in a child process and the parent checks how it exited.

/// d/dx sum(x^2) at x = 3, used by most of the `grad` cases.
private let x0: Float = 3
private let expectedGradient: Float = 6

private final class Model: Module, UnaryLayer {
    let linear: Linear

    override init() {
        linear = Linear(weight: MLXArray([Float(2)], [1, 1]), bias: MLXArray([Float(1)]))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
}

// MARK: Bodies (also run inside exit tests, so they must not capture state)

/// `eval` of an intermediate inside `grad`, default error handler.
private func evalIntermediateInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        eval(y)
        return y.sum()
    }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// `MLXArray.eval()` of an intermediate inside `grad`.
private func arrayEvalInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        y.eval()
        return y.sum()
    }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// `asArray` (debug printing an intermediate) inside `grad`.
private func asArrayInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        #expect(y.asArray(Float.self) == [x0 * x0])
        return y.sum()
    }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// `asArray` inside `grad`, under `withError` -- the error (if any) is
/// captured, so the eval is skipped and `asArray` reads a null pointer.
private func asArrayInsideGradWithError() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        let values = try? withError { y.asArray(Float.self) }
        #expect(values == [x0 * x0])
        return y.sum()
    }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// Control flow on an evaluated value inside `grad` -- the `y[0] >= 0` shape
/// of python code, which needs a real read of the data.
private func controlFlowInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        if (y .> 0).item(Bool.self) {
            return y.sum()
        } else {
            return (-y).sum()
        }
    }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// `eval` inside `valueAndGrad(model:)`'s loss function, e.g. to log the loss.
private func evalInsideModelValueAndGrad() {
    let model = Model()
    eval(model)

    func loss(model: Model, x: MLXArray, y: MLXArray) -> MLXArray {
        let l = mseLoss(predictions: model(x), targets: y, reduction: .mean)
        eval(l)
        _ = l.asArray(Float.self)
        return l
    }

    let lg = valueAndGrad(model: model, loss)

    // prediction = 2 * 1 + 1 = 3, target 0: loss = 9,
    // dL/dw = 2 * 3 * x = 6, dL/db = 2 * 3 = 6
    let (l, grads) = lg(model, MLXArray([Float(1)], [1, 1]), MLXArray([Float(0)], [1, 1]))
    eval(l, grads)

    let flat = Dictionary(uniqueKeysWithValues: grads.flattened())
    #expect(l.item(Float.self) == 9)
    #expect(flat["linear.weight"]?.asArray(Float.self) == [6])
    #expect(flat["linear.bias"]?.asArray(Float.self) == [6])
}

/// `eval` inside the VJP of a custom function: this runs during the backward
/// pass, where every array is a tracer because core holds `RetainGraph`.
private func evalInsideCustomVJP() {
    let f = CustomFunction {
        Forward { inputs in [inputs[0] * inputs[0]] }
        VJP { primals, cotangents in
            let v = 2 * primals[0] * cotangents[0]
            eval(v)
            return [v]
        }
    }

    let g = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
    let d = g(MLXArray(x0))
    #expect(d.item(Float.self) == expectedGradient)
}

/// Port of `test_eval_in_grad` from the python tests.
private func evalInsideVJP() {
    let arr = MLXArray([Float(1)])
    let cotangent = MLXArray([Float(1), 1])
    let y = MLXArray([Float(2), 2])

    // reading a value derived from the primal
    let (_, vjps1) = vjp(
        { x in
            let x = x[0] + y
            let condition = x .< 1
            _ = condition.asArray(Bool.self)
            return [x ** 2]
        }, primals: [arr], cotangents: [cotangent])
    #expect(vjps1[0].item(Float.self) == 12)

    // evaluating a value derived from the primal
    let (_, vjps2) = vjp(
        { x in
            let x = x[0] + MLXArray([Float(1), 1])
            eval(x)
            return [x ** 2]
        }, primals: [arr], cotangents: [cotangent])
    #expect(vjps2[0].item(Float.self) == 8)
}

/// `eval` of an intermediate inside `jvp`.
private func evalInsideJVP() {
    let (_, jvps) = jvp(
        { x in
            let y = x[0] * x[0]
            eval(y)
            return [y]
        }, primals: [MLXArray(x0)], tangents: [MLXArray(Float(1))])
    #expect(jvps[0].item(Float.self) == expectedGradient)
}

/// `asyncEval` of an intermediate inside `grad` -- not allowed.
private func asyncEvalInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let y = x * x
        asyncEval(y)
        return y.sum()
    }
    _ = g(MLXArray(x0))
}

/// `eval` of a traced value inside `vmap` -- not allowed.
private func evalInsideVmap() {
    let mapped = vmap { (x: MLXArray) -> MLXArray in
        let y = x * 2
        eval(y)
        return y
    }
    _ = mapped(MLXArray(0 ..< 4).asType(.float32))
}

/// `eval` of a traced value inside `compile` -- not allowed.
private func evalInsideCompile() {
    let compiled = compile { (x: MLXArray) -> MLXArray in
        let y = x * 2
        eval(y)
        return y
    }
    _ = compiled(MLXArray(Float(3)))
}

// MARK: - Tests

@Suite("eval inside function transformations")
struct EvalInTransformsTests {

    // MARK: Allowed: synchronous eval of vjp/jvp tracers

    // These run as exit tests so that a regression (fatalError from the default
    // error handler, or a crash reading an unevaluated array) fails the test
    // rather than taking down the test runner.

    @Test func evalInsideGrad() async {
        await #expect(processExitsWith: .success) {
            evalIntermediateInsideGrad()
        }
    }

    @Test func arrayEvalMethodInsideGrad() async {
        await #expect(processExitsWith: .success) {
            arrayEvalInsideGrad()
        }
    }

    @Test func asArrayInsideGrad() async {
        await #expect(processExitsWith: .success) {
            MLXTests.asArrayInsideGrad()
        }
    }

    @Test func asArrayInsideGradWithError() async {
        await #expect(processExitsWith: .success) {
            MLXTests.asArrayInsideGradWithError()
        }
    }

    @Test func controlFlowOnAValueInsideGrad() async {
        await #expect(processExitsWith: .success) {
            controlFlowInsideGrad()
        }
    }

    @Test func evalInsideModelValueAndGrad() async {
        await #expect(processExitsWith: .success) {
            MLXTests.evalInsideModelValueAndGrad()
        }
    }

    @Test func evalInsideCustomVJP() async {
        await #expect(processExitsWith: .success) {
            MLXTests.evalInsideCustomVJP()
        }
    }

    @Test func evalInsideVJP() async {
        await #expect(processExitsWith: .success) {
            MLXTests.evalInsideVJP()
        }
    }

    @Test func evalInsideJVP() async {
        await #expect(processExitsWith: .success) {
            MLXTests.evalInsideJVP()
        }
    }

    /// In-process and checked, so a regression reports MLX's error message.
    @Test func checkedEvalInsideGrad() throws {
        var thrown: Error?
        let g = grad { (x: MLXArray) -> MLXArray in
            let y = x * x
            do {
                try checkedEval(y)
            } catch {
                thrown = error
            }
            return y.sum()
        }
        let d = g(MLXArray(x0))

        #expect(thrown == nil, "\(String(describing: thrown))")
        #expect(d.item(Float.self) == expectedGradient)
    }

    /// In-process and checked, inside `valueAndGrad(model:)`.
    @Test func checkedEvalInsideModelValueAndGrad() throws {
        let model = Model()
        eval(model)

        var thrown: Error?
        func loss(model: Model, x: MLXArray, y: MLXArray) -> MLXArray {
            let l = mseLoss(predictions: model(x), targets: y, reduction: .mean)
            do {
                try checkedEval(l)
            } catch {
                thrown = error
            }
            return l
        }

        let lg = valueAndGrad(model: model, loss)
        let (l, _) = lg(model, MLXArray([Float(1)], [1, 1]), MLXArray([Float(0)], [1, 1]))

        #expect(thrown == nil, "\(String(describing: thrown))")
        #expect(l.item(Float.self) == 9)
    }

    // MARK: Not allowed

    @Test func asyncEvalInsideGradAborts() async {
        await #expect(processExitsWith: .failure) {
            asyncEvalInsideGrad()
        }
    }

    @Test func checkedAsyncEvalInsideGradThrows() {
        #expect {
            try withError { asyncEvalInsideGrad() }
        } throws: { error in
            String(describing: error).contains(
                "[async_eval] Not allowed inside a graph transformation")
        }
    }

    @Test func evalInsideVmapAborts() async {
        await #expect(processExitsWith: .failure) {
            evalInsideVmap()
        }
    }

    @Test func checkedEvalInsideVmapThrows() {
        #expect {
            try withError { evalInsideVmap() }
        } throws: { error in
            String(describing: error).contains(
                "[eval] Attempting to eval an array during function transformations like compile or vmap"
            )
        }
    }

    @Test func evalInsideCompileAborts() async {
        await #expect(processExitsWith: .failure) {
            evalInsideCompile()
        }
    }

    @Test func checkedEvalInsideCompileThrows() {
        #expect {
            try withError { evalInsideCompile() }
        } throws: { error in
            String(describing: error).contains(
                "[eval] Attempting to eval an array during function transformations like compile or vmap"
            )
        }
    }
}
