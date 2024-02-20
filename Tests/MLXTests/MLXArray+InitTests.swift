// Copyright © 2024 Apple Inc.

import Foundation
import Numerics
import XCTest

@testable import MLX

class MLXArrayInitTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Creation

    func testInt() {
        // array creation with Int -- we want it to produce .int32
        let a1 = MLXArray(500)
        XCTAssertEqual(a1.dtype, .int32)

        // eplicit int64
        let a2 = MLXArray(int64: 500)
        XCTAssertEqual(a2.dtype, .int64)

        let a3 = MLXArray([1, 2, 3])
        XCTAssertEqual(a3.dtype, .int32)

        let a4 = MLXArray(int64: [1, 2, 3])
        XCTAssertEqual(a4.dtype, .int64)

        let a5 = MLXArray(0 ..< 12)
        XCTAssertEqual(a5.dtype, .int32)

        let a6 = MLXArray(int64: 0 ..< 12)
        XCTAssertEqual(a6.dtype, .int64)
    }

    func testArrayCreationLiteralArray() {
        let a: MLXArray = [20, 30, 40]
        assertEqual(a, MLXArray([20, 30, 40].asInt32))
    }

    func testArrayCreationDoubleArray() {
        // this transforms the array to [Float] and constructs (as a convenience)
        let a = MLXArray(converting: [0.1, 0.5])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a[0].item(Float.self), 0.1, accuracy: 0.01)
    }

    func testArrayCreationArray1D() {
        let a = MLXArray([1, 2, 3])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.count, 3)
        XCTAssertEqual(a.ndim, 1)
        XCTAssertEqual(a.dim(0), 3)
    }

    func testArrayCreationRange() {
        let a = MLXArray(0 ..< 12, [3, 4])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.size, 12)
        XCTAssertEqual(a.ndim, 2)
    }

    func testArrayCreationClosedRange() {
        let a = MLXArray(Int16(3) ... Int16(6))
        XCTAssertEqual(a.dtype, .int16)
        XCTAssertEqual(a.count, 4)
        XCTAssertEqual(a.ndim, 1)
    }

    func testArrayCreationStride() {
        let a = MLXArray(stride(from: Float(0.5), to: Float(1.5), by: Float(0.1)))
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.count, 10)
        XCTAssertEqual(a.ndim, 1)
    }

    func testArrayCreationZeros() {
        let a = MLXArray.zeros([2, 4], type: Int.self)
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.size, 8)
        XCTAssertEqual(a.ndim, 2)
    }

    func testData() {
        let data = Data([1, 2, 3, 4])
        let a = MLXArray(data, [2, 2], type: UInt8.self)
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        assertEqual(a, expected)
    }

    func testUnsafeRawPointer() {
        let data = Data([1, 2, 3, 4])
        let a = data.withUnsafeBytes { ptr in
            MLXArray(ptr, [2, 2], type: UInt8.self)
        }
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        assertEqual(a, expected)
    }

    func testUnsafeBufferPointer() {
        let values: [UInt16] = [1, 2, 3, 4]
        let a = values.withUnsafeBufferPointer { ptr in
            MLXArray(ptr, [2, 2])
        }
        let expected = MLXArray(UInt16(1) ... 4, [2, 2])
        assertEqual(a, expected)
    }

    func testComplexScalar() {
        let c1 = MLXArray(real: 3, imaginary: 4)
        XCTAssertEqual(c1.realPart().item(), 3)
        XCTAssertEqual(c1.imaginaryPart().item(), 4)

        let c2 = MLXArray(Complex(3, 4))
        assertEqual(c1, c2)
    }

    func testComplexArray() {
        let r1 = MLXArray(converting: [2, 3, 4])
        let i1 = MLXArray(converting: [7, 8, 9])
        let c1 = r1 + i1.asImaginary()

        assertEqual(c1.realPart(), r1)
        assertEqual(c1.imaginaryPart(), i1)

        let a1: [Complex<Float>] = [Complex(2, 7), Complex(3, 8), Complex(4, 9)]
        let c2 = MLXArray(a1)

        assertEqual(c1, c2)
    }
}
