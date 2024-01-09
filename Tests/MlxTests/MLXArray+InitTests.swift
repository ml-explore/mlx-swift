import Foundation
import XCTest
@testable import Mlx

class MLXArrayInitTests : XCTestCase {
    
    // MARK: - Creation

    private func assertSingleton(_ array: MLXArray, _ dtype: DType) {
        XCTAssertEqual(array.dtype, dtype)
        XCTAssertEqual(array.count, 1)
        XCTAssertEqual(array.ndim, 0)
    }
    
    func testArrayCreationLiteralInt32() {
        let a: MLXArray = 3
        assertSingleton(a, .int32)
    }

    func testArrayCreationLiteralBool() {
        let a: MLXArray = true
        assertSingleton(a, .bool)
    }

    func testArrayCreationLiteralFloat() {
        let a: MLXArray = 3.5
        assertSingleton(a, .float32)
    }

    func testArrayCreationArray1D() {
        let a = MLXArray([1, 2, 3])
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.count, 3)
        XCTAssertEqual(a.ndim, 1)
        XCTAssertEqual(a.dim(0), 3)
    }
    
    func testArrayCreationRange() {
        let a = MLXArray(0 ..< 12, [3, 4])
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.count, 12)
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
        let a = MLXArray.zeros(Int32.self, [2, 4])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.count, 8)
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
