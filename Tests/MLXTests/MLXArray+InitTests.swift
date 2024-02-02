// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class MLXArrayInitTests: XCTestCase {

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

}
