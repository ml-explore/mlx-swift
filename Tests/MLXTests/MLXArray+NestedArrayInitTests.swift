// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class MLXArrayNestedArrayInitTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - 2-D init

    func testInit2DFloat() {
        let a = MLXArray([[Float(1), 2], [3, 4]])
        XCTAssertEqual(a.shape, [2, 2])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 2, 3, 4])
    }

    func testInit2DInt() {
        let a = MLXArray([[1, 2, 3], [4, 5, 6]])
        XCTAssertEqual(a.shape, [2, 3])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.asArray(Int32.self), [1, 2, 3, 4, 5, 6])
    }

    func testInit2DInt64() {
        let a = MLXArray(int64: [[1, 2], [3, 4]])
        XCTAssertEqual(a.shape, [2, 2])
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.asArray(Int.self), [1, 2, 3, 4])
    }

    func testInit2DDoubleConverting() {
        let a = MLXArray(converting: [[1.5, 2.5], [3.5, 4.5]])
        XCTAssertEqual(a.shape, [2, 2])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1.5, 2.5, 3.5, 4.5])
    }

    func testInit2DSingleRow() {
        let a = MLXArray([[Float(1), 2, 3]])
        XCTAssertEqual(a.shape, [1, 3])
        XCTAssertEqual(a.asArray(Float.self), [1, 2, 3])
    }

    // MARK: - 3-D init

    func testInit3DFloat() {
        let a = MLXArray([
            [[Float(1), 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        XCTAssertEqual(a.shape, [2, 2, 2])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    func testInit3DInt() {
        let a = MLXArray([
            [[1], [2]],
            [[3], [4]],
        ])
        XCTAssertEqual(a.shape, [2, 2, 1])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.asArray(Int32.self), [1, 2, 3, 4])
    }

    // MARK: - 4-D init

    func testInit4DFloat() {
        // batch=1, channel=2, height=2, width=2
        let a = MLXArray([
            [
                [[Float(1), 2], [3, 4]],
                [[5, 6], [7, 8]],
            ]
        ])
        XCTAssertEqual(a.shape, [1, 2, 2, 2])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.asArray(Float.self), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    // MARK: - Round-trip with asArray2 / 3 / 4

    func testAsArray2RoundTrip() {
        let original: [[Float]] = [[1, 2, 3], [4, 5, 6]]
        let a = MLXArray(original)
        let recovered = a.asArray2(Float.self)
        XCTAssertEqual(recovered, original)
    }

    func testAsArray3RoundTrip() {
        let original: [[[Float]]] = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
        let a = MLXArray(original)
        let recovered = a.asArray3(Float.self)
        XCTAssertEqual(recovered, original)
    }

    func testAsArray4RoundTrip() {
        let original: [[[[Float]]]] = [
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ]
        ]
        let a = MLXArray(original)
        let recovered = a.asArray4(Float.self)
        XCTAssertEqual(recovered, original)
    }

    // MARK: - Cross-type extraction (asArray converts via asType)

    func testAsArray2WithTypeConversion() {
        let a = MLXArray([[1, 2], [3, 4]])  // int32
        let recovered = a.asArray2(Float.self)  // requested as Float
        XCTAssertEqual(recovered, [[1.0, 2.0], [3.0, 4.0]])
    }

    // MARK: - asArray2/3/4 honor the actual MLX layout

    func testAsArray2ReshapesFromComputedTensor() {
        // construct a 2-D array via reshape then verify extraction respects strides
        let a = MLXArray(0 ..< 12).reshaped([3, 4])
        let recovered = a.asArray2(Int32.self)
        XCTAssertEqual(recovered, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    }
}
