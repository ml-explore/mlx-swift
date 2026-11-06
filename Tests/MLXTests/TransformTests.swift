// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXOptimizers

@Suite
struct TransformTests {

    @Test func evalStructures() {
        // eval various structures
        let a = MLXArray(0)
        let b = (a, a)
        let c = [
            "foo": a,
            "bar": a,
        ]
        let d = [
            "foo": [a, a],
            "bar": [a, a],
        ]
        let e = [
            ("foo", (a, a)),
            ("bar", (a, a)),
        ]

        eval(a, b, c, d, e)
    }

    @Test func gradOfSquare() {
        func fn(_ x: MLXArray) -> MLXArray {
            x.square()
        }

        let x = MLXArray(1.5)

        let gradFn = grad(fn)

        // derivative of x^2 is 2*x
        let dfdx = gradFn(x)
        #expect(dfdx.item() == Float(2 * 1.5))

        // derivative of 2*x is 2
        let df2dx2 = grad(grad(fn))(x)
        #expect(df2dx2.item() == Float(2))
    }

    @Test func valueAndGradOfSquare() {
        // note: valueAndGrad only deals with arrays of MLXArray
        func fn(_ x: [MLXArray]) -> [MLXArray] {
            [x[0].square()]
        }

        let x = MLXArray(1.5)

        let vg = valueAndGrad(fn)

        let (value, grad) = vg([x])

        #expect(value[0].item() == Float(1.5 * 1.5))
        #expect(grad[0].item() == Float(2 * 1.5))
    }

    @Test(.randomSeed(0)) func valueAndGradNested() throws {
        // valueAndGrad on a nested structure, e.g. parameters.
        // this isn't a real model but can exercise the
        // machinery

        class M: Module, UnaryLayer {
            let linear = Linear(5, 5)

            func callAsFunction(_ x: MLXArray) -> MLXArray {
                relu(linear(x) * 5 + 1)
            }
        }

        let m = M()
        MLX.eval(m)

        let x = MLXArray(0 ..< 5, [1, 5])
        let y = MLXArray(0 ..< 5)

        func loss(model: M, x: MLXArray, y: MLXArray) -> MLXArray {
            crossEntropy(logits: model(x), targets: y, reduction: .mean)
        }

        let lg = valueAndGrad(model: m, loss)

        let (loss, grads) = lg(m, x, y)

        // we have known loss because of the random seed
        expectEqual(loss.item(), Float(5.98534), accuracy: 0.001)

        let linear = try #require(grads["linear"], "missing linear key")
        switch linear {
        case .dictionary(let d):
            #expect(d["bias"] != nil)
            #expect(d["weight"] != nil)
        default:
            Issue.record("linear is not a dictionary")
        }
    }

    @Test func compileMatchesUncompiled() {
        @Sendable
        func f(inputs: [MLXArray]) -> [MLXArray] {
            [square(inputs[0] * inputs[1])]
        }

        let i1 = MLXRandom.normal([20, 20])
        let i2 = MLXRandom.normal([20, 20])

        // evaluate directly
        let r1 = f(inputs: [i1, i2])[0]

        // evaluate compiled
        let compiled = compile(f)
        let r2 = compiled([i1, i2])[0]

        expectEqual(r1, r2)

        let r3 = compiled([i1, i2])[0]
        expectEqual(r1, r3)
    }

    class CompileTestState: CustomStringConvertible, Updatable {
        var y: MLXArray
        var o: MLXArray?

        init(_ y: Float = 2.0) {
            self.y = MLXArray(y)
        }

        var description: String {
            "State: y=\(y)"
        }

        func innerState() -> [MLXArray] {
            [y, o].compactMap { $0 }
        }
    }

    @Test func compileStateNoCompile() {
        // tests of the function & state without using compile
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            let x = x[0] + state.y
            return [x]
        }

        // state is not mutated
        _ = testState([MLXArray(1)])
        #expect(state.y.item(Float.self) == 2.0)

        // make sure it consumed the captured state
        state.y = MLXArray(3)
        let r1 = testState([MLXArray(2)])
        #expect(r1[0].item(Float.self) == 5.0)
    }

    @Test func compiledStateReadOnly() {
        // tests of the function & state without using compile
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            let x = x[0] + state.y
            return [x]
        }

        let compiled = compile(inputs: [state], testState(_:))

        // compiled function uses the state
        let r1 = compiled([MLXArray(5)])
        #expect(r1[0].item(Float.self) == 7.0)

        // state is not mutated
        #expect(state.y.item(Float.self) == 2.0)

        // and uses the captured state and passed argument
        state.y = MLXArray(8)
        let r2 = compiled([MLXArray(3)])
        #expect(r2[0].item(Float.self) == 11.0)

        // state is not mutated
        #expect(state.y.item(Float.self) == 8.0)
    }

    @Test func compiledStateMutation() {
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            state.o = x[0] + 3
            return [abs(x[0])]
        }

        let compiled = compile(inputs: [state], outputs: [state], testState(_:))

        // input state is only on array but output state is 2
        let r1 = compiled([MLXArray(3)])
        #expect(r1[0].item(Float.self) == 3)
        #expect(state.o!.item(Float.self) == 6)

        // mutates the state
        let r2 = compiled([MLXArray(-11)])
        #expect(r2[0].item(Float.self) == 11)
        #expect(state.o!.item(Float.self) == -8)
    }

    // https://github.com/ml-explore/mlx-swift-lm/issues/586 --
    // compile can capture Module parameters
    class ScaleModule: Module, UnaryLayer {
        var scale: MLXArray

        init(_ value: Float) {
            self.scale = MLXArray(value)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x * scale
        }
    }

    @Test func compileCapturesModuleParameterWithoutState() {
        // verify that compile captures all values used if not
        // passed as inputs.

        let model = ScaleModule(2.0)

        // lie about the inputs so that we can use a non-Sendable
        // closure.  this is meant to demonstrate the failure
        let compiled = compile(inputs: []) { (x: MLXArray) -> MLXArray in
            model(x)
        }

        let r1 = compiled(MLXArray(Float(3)))
        #expect(r1.item(Float.self) == 6)

        // simulate a LoRA-style in place update of the module's weights
        model.update(parameters: .unflattened([("scale", MLXArray(Float(10)))]))
        eval(model)

        // Verify capture of Module parameters -- it doesn't see the update
        let r2 = compiled(MLXArray(Float(3)))
        #expect(
            r2.item(Float.self) == 6,
            "compiled function saw the update and it should not")
    }

    @Test func compileWithStateObservesModuleParameterUpdates() {
        // verify inputs correctly handles updated
        // Module parameters

        let model = ScaleModule(2.0)

        let compiled = compile(inputs: [model]) { (x: MLXArray) -> MLXArray in
            model(x)
        }

        let r1 = compiled(MLXArray(Float(3)))
        #expect(r1.item(Float.self) == 6)

        // simulate a LoRA-style in place update of the module's weights
        model.update(parameters: .unflattened([("scale", MLXArray(Float(10)))]))
        eval(model)

        // with state:, the compiled function observes the updated parameter
        let r2 = compiled(MLXArray(Float(3)))
        #expect(r2.item(Float.self) == 30)
    }

    @Test func compiledRandom() {
        @Sendable
        func f(_ bias: MLXArray) -> MLXArray {
            MLXRandom.uniform(0 ..< 1, [4]) + bias
        }

        let bias = MLXArray(0)

        // use a task-local random state rather than MLXRandom.globalState: tests run
        // in parallel and while c2 is traced/called the captured state holds tracers,
        // which other tests drawing from the global state would pick up
        let state = MLXRandom.RandomState()
        withRandomState(state) {
            // without capturing state this won't mutate the random state
            let c1 = compile(f)

            let c1a = c1(bias)
            let c1b = c1(bias)
            #expect(allClose(c1a, c1b).item())

            // now cature the random state and the random numbers should change per call
            let c2 = compile(inputs: [state], outputs: [state], f)

            let c2a = c2(bias)
            let c2b = c2(bias)
            #expect(!allClose(c2a, c2b).item())
        }
    }

    @Test func compilePerformance() {
        // this is the code from compilation.md, sized
        // sensibly for a test

        func measure(_ f: (MLXArray) -> MLXArray, _ x: MLXArray) {
            // warm up
            for _ in 0 ..< 3 {
                eval(f(x))
            }

            let start = Date.timeIntervalSinceReferenceDate
            let iterations = 10
            for _ in 0 ..< iterations {
                eval(f(x))
            }
            let end = Date.timeIntervalSinceReferenceDate

            let timePerIteration = 1000.0 * (end - start) / Double(iterations)

            print("Time per iteration \(timePerIteration.formatted()) ms")
        }

        let x = MLXRandom.uniform(0 ..< 1, [16, 250, 1024])

        // gelu() is already compiled, so use a local copy that is not
        func localGelu(_ x: MLXArray) -> MLXArray {
            x * (1 + erf(x / sqrt(2))) / 2
        }

        measure(localGelu, x)
        measure(compile(localGelu), x)
    }

    @Test func vmapSimple() {
        func f(_ x: MLXArray) -> MLXArray { x * 2 }

        let x = MLXArray(0 ..< 6, [3, 2])
        let vf = vmap(f)

        let expected = stacked((0 ..< 3).map { f(x[$0]) }, axis: 0)

        expectEqual(vf(x), expected)
    }

    @Test func vmapSendableSimple() async {
        @Sendable
        func f(_ x: MLXArray) -> MLXArray { x * 2 }

        let x = MLXArray(0 ..< 6, [3, 2])

        // this result type is Sendable
        let vf = vmapSendable(f)

        let t = Task {
            let expected = stacked((0 ..< 3).map { f(x[$0]) }, axis: 0)

            expectEqual(vf(x), expected)
        }

        _ = await t.value
    }

    @Test func vmapTwoInputs() {
        @Sendable
        func f(_ x: MLXArray, _ y: MLXArray) -> MLXArray { x + y }

        let x = MLXArray(0 ..< 6, [3, 2])
        let y = MLXArray(10 ..< 16, [3, 2])
        let vf = vmap(f)

        let expected = stacked((0 ..< 3).map { f(x[$0], y[$0]) }, axis: 0)

        expectEqual(vf(x, y), expected)
    }

    @Test func vmapAxisPermutations() {
        @Sendable
        func f(_ x: MLXArray) -> MLXArray { x * 2 }

        let x = MLXArray(0 ..< 6, [2, 3])

        let map1 = vmap(f, inAxes: 1, outAxes: 0)
        let expected1 = stacked((0 ..< 3).map { f(x[.ellipsis, $0]) }, axis: 0)
        expectEqual(map1(x), expected1)

        let map2 = vmap(f, inAxes: 0, outAxes: 1)
        let expected2 = stacked((0 ..< 2).map { f(x[$0]) }, axis: 1)
        expectEqual(map2(x), expected2)

        let map3 = vmap(f, outAxes: 1)
        let expected3 = stacked((0 ..< 2).map { f(x[$0]) }, axis: 1)
        expectEqual(map3(x), expected3)
    }

    @Test func vmapTwoInputsAxisPermutations() {
        @Sendable
        func f(_ x: MLXArray, _ y: MLXArray) -> MLXArray { x + y }

        let x = MLXArray(0 ..< 6, [2, 3])
        let y = MLXArray(10 ..< 16, [3, 2])

        let map1 = vmap(f, inAxes: (0, 1), outAxes: 0)
        let expected1 = stacked((0 ..< 2).map { i in f(x[i], y[.ellipsis, i]) }, axis: 0)
        expectEqual(map1(x, y), expected1)

        let map2 = vmap(f, inAxes: (0, 1), outAxes: 1)
        let expected2 = stacked((0 ..< 2).map { i in f(x[i], y[.ellipsis, i]) }, axis: 1)
        expectEqual(map2(x, y), expected2)
    }

    @Test func vmapWithCompile() {
        @Sendable
        func f(_ x: MLXArray) -> MLXArray { x.square() }

        let x = MLXRandom.normal([4, 2])

        let compiled = compile(f)
        let mapCompile = vmap(compiled)
        expectEqual(mapCompile(x), vmap(f)(x))

        // use a non-Sendable version of compile because vmap doesn't produce
        // a Sendable function
        let compiledMap = compile(inputs: [], vmap(f))
        expectEqual(compiledMap(x), vmap(f)(x))
    }

    @Test func vmapWithGrad() {
        func f(_ x: MLXArray) -> MLXArray { x.square().sum() }

        let x = MLXArray(0 ..< 6, [3, 2])

        let gradF = grad(f)
        let vg = vmap(gradF)
        expectEqual(vg(x), x * 2)

        let gv = grad { vmap(f)($0).sum() }
        expectEqual(gv(x), x * 2)
    }

    @Test func compileThreadSafety() async throws {

        @Sendable
        func swiglu(_ xLinear: MLXArray, _ xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0)
            -> MLXArray
        {
            var xLinear = xLinear
            var xGlu = xGlu
            xGlu = clip(xGlu, max: MLXArray(limit))
            xLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

            let gluScaled = alpha * xGlu
            let sig = sigmoid(gluScaled)

            let outGlu = xGlu * sig
            return outGlu * (xLinear + 1)
        }

        func compileSwiglu() -> @Sendable (MLXArray, MLXArray) -> MLXArray {
            compile(shapeless: true) { xLinear, xGlu in
                swiglu(xLinear, xGlu)
            }
        }

        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 10 {
                group.addTask {
                    withRandomState(.init()) {
                        let x = MLXRandom.normal([1024, 1024])
                        let y = MLXRandom.normal(x.shape)
                        let _ = compileSwiglu()(x, y)
                    }
                }
            }

            for await _ in group {

            }
        }
    }

    @Test func vmapThreadSafety() async throws {
        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 10 {
                group.addTask {
                    func f(_ x: MLXArray) -> MLXArray { x.square().sum() }

                    let x = MLXArray(0 ..< 6, [3, 2])

                    let gradF = grad(f)
                    let vg = vmap(gradF)
                    expectEqual(vg(x), x * 2)

                    let _ = grad { vmap(f)($0).sum() }
                }
            }

            for await _ in group {

            }
        }
    }

    // Note: OptimizerTests contains additional integration tests of compile()

    @Test func compileSingleArrayErrorPropagatesViaWithError() throws {
        // A compiled function that produces a broadcast shape mismatch at evaluation time.
        // The constant has shape [3]; passing an input with shape [2] triggers an MLX error
        // inside mlx_closure_apply.
        //
        // Before fix: mlx_closure_apply's non-zero return value was silently ignored,
        // causing innerCall to return [], and the single-array overload to crash with
        // "Fatal error: Index out of range" — a Swift trap that bypasses withError.
        //
        // After fix: innerCall returns [] early on error, the overload returns a placeholder,
        // and withError properly surfaces the MLXError.
        let compiled = compile { (x: MLXArray) -> MLXArray in
            let constant = MLXArray([Float](repeating: 1, count: 3))
            return x + constant
        }
        let x = MLXArray([Float](repeating: 0, count: 2))

        #expect(throws: MLXError.self) {
            try withError {
                eval(compiled(x))
            }
        }
    }

    @Test func compileMultiArrayErrorPropagatesViaWithError() throws {
        // Same scenario with the [MLXArray] -> [MLXArray] overload.
        let compiled = compile { (inputs: [MLXArray]) -> [MLXArray] in
            let constant = MLXArray([Float](repeating: 1, count: 3))
            return [inputs[0] + constant]
        }
        let x = MLXArray([Float](repeating: 0, count: 2))

        #expect(throws: MLXError.self) {
            try withError {
                eval(compiled([x]))
            }
        }
    }

    @Test func compileTwoArrayErrorPropagatesViaWithError() throws {
        // Same scenario with the (MLXArray, MLXArray) -> MLXArray overload.
        let compiled = compile { (a: MLXArray, _: MLXArray) -> MLXArray in
            let constant = MLXArray([Float](repeating: 1, count: 3))
            return a + constant
        }
        let x = MLXArray([Float](repeating: 0, count: 2))
        let y = MLXArray([Float](repeating: 0, count: 2))

        #expect(throws: MLXError.self) {
            try withError {
                eval(compiled(x, y))
            }
        }
    }

    @Test func customFunctionVJP2() {
        // Custom function with 2-parameter VJP: (primals, cotangents)
        let customSquare = CustomFunction {
            Forward { inputs in
                [inputs[0].square()]
            }
            VJP { primals, cotangents in
                // d/dx(x^2) = 2 * x, so cotangent * 2 * x
                [cotangents[0] * 2 * primals[0]]
            }
        }

        let x = MLXArray(3.0)
        let forwardOut = customSquare([x])
        expectEqual(forwardOut[0], MLXArray(9.0))

        let gradFn = grad { inputs in
            customSquare(inputs)
        }
        let grads = gradFn([x])
        expectEqual(grads[0], MLXArray(6.0))
    }

    @Test func customFunctionVJP3WithOutputs() {
        // Custom function with 3-parameter VJP: (primals, cotangents, outputs)
        // For sigmoid: s(x) = output, and ds/dx = s * (1 - s) = output * (1 - output)
        // This directly exercises the 3rd parameter `outputs`
        let customSigmoid = CustomFunction {
            Forward { inputs in
                [sigmoid(inputs[0])]
            }
            VJP { primals, cotangents, outputs in
                let out = outputs[0]
                return [cotangents[0] * out * (1 - out)]
            }
        }

        let x = MLXArray(0.0)
        let forwardOut = customSigmoid([x])
        expectEqual(forwardOut[0], MLXArray(0.5))

        let gradFn = grad { inputs in
            customSigmoid(inputs)
        }
        let grads = gradFn([x])
        // At x = 0, sigmoid(0) = 0.5, grad = 0.5 * (1 - 0.5) = 0.25
        expectEqual(grads[0], MLXArray(0.25))
    }

    @Test func customFunctionSendable() async {
        // Verify Sendable closures and Sendable return from CustomFunction
        let fn: @Sendable ([MLXArray]) -> [MLXArray] = CustomFunction {
            Forward { inputs in
                [inputs[0] + 1]
            }
            VJP { primals, cotangents, outputs in
                [cotangents[0]]
            }
        }

        // CustomFunction result is @Sendable and can be transferred across Task boundaries
        let val = await Task { @Sendable in
            let input = MLXArray(5.0)
            let result = fn([input])
            return result[0].item(Float.self)
        }.value

        #expect(val == 6.0)
    }

}
