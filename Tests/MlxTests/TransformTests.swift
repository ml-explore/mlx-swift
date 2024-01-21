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

}
