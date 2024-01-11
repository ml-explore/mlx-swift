import Foundation
import XCTest
@testable import Mlx

class MLXArrayOpsTests : XCTestCase {
    
    // MARK: - Operators

    func testArithmeticSimple() {
        let a = MLXArray([1, 2, 3])
        let b = MLXArray(converting: [-5.0, 37.5, 4])
        
        // example of an expression -- the - 1 is using the 1 as ExpressibleByIntegerLiteral
        let r = a + b - 1
        
        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [-5, 38.5, 6]))
    }
    
    func testArithmeticMatrix() {
        let a = MLXArray([1, 2, 3, 4], [2, 2])
        let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])

        // (a * 5)^2 matmul b
        let r = (a * 5) ** 2 *** b
        
        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [2, 3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [575, 1037.5, 100, 1675, 8837.5, 900], [2, 3]))
    }

    func testArithmeticDivide() {
        let a = MLXArray([1, 2, 3])
        
        // in python: (a // 2) / 0.5
        let r = (a /% 2) / 0.5
        
        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [0, 2, 2]))
    }

    func testOpsLogical() {
        let a = MLXArray([1, 2, 3])
        
        let r = (a < (a + 1))
        
        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .bool)
        XCTAssertEqual(r.all().item(Bool.self), true)
        XCTAssertEqual(r.allTrue(), true)
    }

}
