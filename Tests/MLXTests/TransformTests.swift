// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXOptimizers

class TransformTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testEval() {
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

    func testGrad() {
        func fn(_ x: MLXArray) -> MLXArray {
            x.square()
        }

        let x = MLXArray(1.5)

        let gradFn = grad(fn)

        // derivative of x^2 is 2*x
        let dfdx = gradFn(x)
        XCTAssertEqual(dfdx.item(), Float(2 * 1.5))

        // derivative of 2*x is 2
        let df2dx2 = grad(grad(fn))(x)
        XCTAssertEqual(df2dx2.item(), Float(2))
    }

    func testValueAndGrad() {
        // note: valueAndGrad only deals with arrays of MLXArray
        func fn(_ x: [MLXArray]) -> [MLXArray] {
            [x[0].square()]
        }

        let x = MLXArray(1.5)

        let vg = valueAndGrad(fn)

        let (value, grad) = vg([x])

        XCTAssertEqual(value[0].item(), Float(1.5 * 1.5))
        XCTAssertEqual(grad[0].item(), Float(2 * 1.5))
    }

    func testValueAndGradNested() {
        // valueAndGrad on a nested structure, e.g. parameters.
        // this isn't a real model but can exercise the
        // machinery

        MLXRandom.seed(0)

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
        XCTAssertEqual(loss.item(), Float(5.98534), accuracy: 0.001)

        if let linear = grads["linear"] {
            switch linear {
            case .dictionary(let d):
                XCTAssertNotNil(d["bias"])
                XCTAssertNotNil(d["weight"])
            default:
                XCTFail("linear is not a dictionary")
            }
        } else {
            XCTFail("missing linear key")
        }
    }

    func testCompile() {
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

        assertEqual(r1, r2)

        let r3 = compiled([i1, i2])[0]
        assertEqual(r1, r3)
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

    func testCompileStateNoCompile() {
        // tests of the function & state without using compile
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            let x = x[0] + state.y
            return [x]
        }

        // state is not mutated
        _ = testState([MLXArray(1)])
        XCTAssertEqual(state.y.item(Float.self), 2.0)

        // make sure it consumed the captured state
        state.y = MLXArray(3)
        let r1 = testState([MLXArray(2)])
        XCTAssertEqual(r1[0].item(Float.self), 5.0)
    }

    func testCompiledStateReadOnly() {
        // tests of the function & state without using compile
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            let x = x[0] + state.y
            return [x]
        }

        let compiled = compile(inputs: [state], testState(_:))

        // compiled function uses the state
        let r1 = compiled([MLXArray(5)])
        XCTAssertEqual(r1[0].item(Float.self), 7.0)

        // state is not mutated
        XCTAssertEqual(state.y.item(Float.self), 2.0)

        // and uses the captured state and passed argument
        state.y = MLXArray(8)
        let r2 = compiled([MLXArray(3)])
        XCTAssertEqual(r2[0].item(Float.self), 11.0)

        // state is not mutated
        XCTAssertEqual(state.y.item(Float.self), 8.0)
    }

    func testCompiledStateMutation() {
        let state = CompileTestState()

        func testState(_ x: [MLXArray]) -> [MLXArray] {
            state.o = x[0] + 3
            return [abs(x[0])]
        }

        let compiled = compile(inputs: [state], outputs: [state], testState(_:))

        // input state is only on array but output state is 2
        let r1 = compiled([MLXArray(3)])
        XCTAssertEqual(r1[0].item(Float.self), 3)
        XCTAssertEqual(state.o!.item(Float.self), 6)

        // mutates the state
        let r2 = compiled([MLXArray(-11)])
        XCTAssertEqual(r2[0].item(Float.self), 11)
        XCTAssertEqual(state.o!.item(Float.self), -8)
    }

    func testCompiledRandom() {
        func f(_ bias: MLXArray) -> MLXArray {
            MLXRandom.uniform(0 ..< 1, [4]) + bias
        }

        let bias = MLXArray(0)

        // without capturing state this won't mutate the random state
        let c1 = compile(f)

        let c1a = c1(bias)
        let c1b = c1(bias)
        XCTAssertTrue(allClose(c1a, c1b).item())

        // now cature the random state and the random numbers should change per call
        let c2 = compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState], f)

        let c2a = c2(bias)
        let c2b = c2(bias)
        XCTAssertFalse(allClose(c2a, c2b).item())
    }

    func testCompilePerformance() {
        // this is the code from compilation.md

        func measure(_ f: (MLXArray) -> MLXArray, _ x: MLXArray) {
            // warm up
            for _ in 0 ..< 10 {
                eval(f(x))
            }

            let start = Date.timeIntervalSinceReferenceDate
            let iterations = 100
            for _ in 0 ..< iterations {
                eval(f(x))
            }
            let end = Date.timeIntervalSinceReferenceDate

            let timePerIteration = 1000.0 * (end - start) / Double(iterations)

            print("Time per iteration \(timePerIteration.formatted()) ms")
        }

        let x = MLXRandom.uniform(0 ..< 1, [32, 1000, 4096])

        measure(gelu, x)
        measure(compile(gelu), x)
    }

    // Note: OptimizerTests contains additional integration tests of compile()

}
