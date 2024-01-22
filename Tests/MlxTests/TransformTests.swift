import Foundation
import XCTest
@testable import Mlx

class TransformTests : XCTestCase {

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
}
