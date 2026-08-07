// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class CustomFunctionTests: XCTestCase {

    func testCustomFunctionForwardOnly() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * 2] }
        }

        let result = f([MLXArray(Float(3))])
        XCTAssertEqual(result[0].item(), Float(6))
    }

    func testCustomFunctionForwardAndVJP() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * inputs[0]] }
            VJP { inputs, cotangents in [cotangents[0] * 2 * inputs[0]] }
        }

        let gradFn = grad { x in f([x])[0] }
        let dfdx = gradFn(MLXArray(Float(3)))

        // d/dx x^2 = 2x = 6 at x=3
        XCTAssertEqual(dfdx.item(), Float(6))
    }
}
