// Copyright © 2024 Apple Inc.

import Foundation
import Numerics
import XCTest

@testable import MLX

class MLXArrayOpsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Operators

    func testArithmeticSimple() {
        var a = MLXArray([1, 2, 3])
        var b = MLXArray(converting: [-5.0, 37.5, 4])

        // example of an expression -- the - 1 is using the 1 as ExpressibleByIntegerLiteral
        let r = a + b - 1

        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [-5, 38.5, 6]))

        a += b
        XCTAssertEqual(a.shape, [3])
        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, MLXArray(converting: [-4, 39.5, 7]))

        a += 1
        XCTAssertEqual(a.shape, [3])
        XCTAssertEqual(a.dtype, .float32)
        assertEqual(a, MLXArray(converting: [-3, 40.5, 8]))
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
        let r = a.floorDivide(2) / Float(0.5)

        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .float32)
        assertEqual(r, MLXArray(converting: [0, 2, 2]))
    }

    func testOpsLogical() {
        let a = MLXArray([1, 2, 3])

        let r = (a .< (a + 1))

        // make sure everything got hooked up
        XCTAssertEqual(r.shape, [3])
        XCTAssertEqual(r.dtype, .bool)
        XCTAssertEqual(r.all().item(Bool.self), true)
        XCTAssertEqual(r.all().item(), true)
    }

    func testOpsLogicalBoolContext() {
        let a = MLXArray([1, 2, 3])

        if (a .< (a + 1)).all().item() {
            // expected
        } else {
            XCTFail("should be true")
        }
    }

    func testFunctions() {
        let a = MLXArray(0 ..< 12, [4, 3])

        let r = a.square().sqrt().transposed().T
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

        let f1 = a.flattened()
        XCTAssertEqual(f1.shape, [8 * 4 * 3])

        let f2 = a.flattened(start: 1)
        XCTAssertEqual(f2.shape, [8, 4 * 3])
    }

    public func testMoveAxis() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])

        let r = array.movedAxis(source: 0, destination: 3)

        let expected = MLXArray(
            [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], [2, 2, 2, 2])
        assertEqual(r, expected)
    }

    public func testSwapAxes() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])

        let r = array.swappedAxes(0, 3)

        let expected = MLXArray(
            [0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15], [2, 2, 2, 2])
        assertEqual(r, expected)
    }

    public func testPromotionTests() {
        // expect that adding a scalar will not promote to a larger type
        let x = MLXArray.ones([10], type: Float16.self)

        XCTAssertEqual((x * 15.5).dtype, .float16)
        XCTAssertEqual((15.5 * x).dtype, .float16)
    }

    public func testBFloat16() {
        let x = MLXArray([1, 5, 9]).asType(.bfloat16)
        let y = x * x
        XCTAssertEqual(y.dtype, .bfloat16)

        // this will convert the 3 to bfloat16 via init<T: HasDType>(_ value: T, dtype: DType)
        let z = (y + 3).sum()
        XCTAssertEqual(z.dtype, .bfloat16)

        // read out the value as a usable type
        let r1 = z.item(Float32.self)
        XCTAssertEqual(r1, 116)

        let r2 = z.asArray(Float.self)
        XCTAssertEqual(r2, [116])
    }

    public func testComplexPromote() {
        let x = MLXArray(Complex(3, 4))
        let y = x * x
        XCTAssertEqual(y.dtype, .complex64)

        // this will convert the 3 to (3, 0i) via init<T: HasDType>(_ value: T, dtype: DType)
        let z = y + 3
        XCTAssertEqual(z.dtype, .complex64)

        // read out as complex
        let r1 = z.item(Complex.self)
        XCTAssertEqual(r1, Complex(-4, 24))

        // read out the value as a usable type
        let r2 = z.asArray(Complex.self)
        XCTAssertEqual(r2, [Complex(-4, 24)])
    }

}
