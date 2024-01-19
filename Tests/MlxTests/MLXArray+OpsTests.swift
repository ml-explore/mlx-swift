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

        // (a * 5)^2 @ b
        let r = ((a * 5) ** 2).matmul(b)
        
        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [2, 3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [575, 1037.5, 100, 1675, 8837.5, 900], [2, 3]))
    }

    func testArithmeticDivide() {
        let a = MLXArray([1, 2, 3])
        
        // in python: (a // 2) / 0.5
        let r = a.floorDivide(2) / 0.5
        
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

    func testFunctions() {
        let a = MLXArray(0 ..< 12, [4, 3])
        
        let r = a.square().sqrt().transpose().T
        assertEqual(a, r)
    }
    
    func testSplitEqual() {
        let a = MLXArray(0 ..< 12, [4, 3])
        
        let s = a.split(parts: 2)
        XCTAssertEqual(s[0].shape, [2, 3])
        XCTAssertEqual(s[1].shape, [2, 3])
    }
    
    func testSplitIndices() {
        let a = MLXArray(0 ..< 12, [4, 3])
        
        let s = a.split(indices: [0, 3])
        XCTAssertEqual(s[0].shape, [0, 3])
        XCTAssertEqual(s[1].shape, [3, 3])
        XCTAssertEqual(s[2].shape, [1, 3])
    }
    
    func testFlatten() {
        let a = MLXArray(0 ..< (8 * 4 * 3), [8, 4, 3])
        
        let f1 = a.flatten()
        XCTAssertEqual(f1.shape, [8 * 4 * 3])
        
        let f2 = a.flatten(start: 1)
        XCTAssertEqual(f2.shape, [8, 4 * 3])
    }

    public func testMoveAxis() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
        
        let r = array.moveAxis(source: 0, destination: 3)
        
        let expected = MLXArray([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], [2, 2, 2, 2])
        assertEqual(r, expected)
    }
    
    public func testSwapAxes() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
        
        let r = array.swapAxes(0, 3)
        
        let expected = MLXArray([0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15], [2, 2, 2, 2])
        assertEqual(r, expected)
    }
    
}
