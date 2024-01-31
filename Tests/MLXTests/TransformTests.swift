// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

class TransformTests: XCTestCase {

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
        MLX.eval(m.parameters())

        let x = MLXArray(0 ..< 5, [1, 5])
        let y = MLXArray(0 ..< 5)

        func loss(model: M, x: MLXArray, y: MLXArray) -> MLXArray {
            crossEntropy(logits: model(x), targets: y, reduction: .mean)
        }

        let lg = valueAndGrad(model: m, loss)

        let (loss, grads) = lg(m, x, y)

        // we have known loss because of the random see
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
}
