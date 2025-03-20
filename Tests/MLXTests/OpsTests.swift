// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class OpsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testAsStridedReshape() {
        // just changing the shape and using the default strides is the same as reshape
        let a = MLXArray(0 ..< 12, [4, 3])

        // this uses [4, 1] as the strides
        let b = asStrided(a, [3, 4])
        assertEqual(b, a.reshaped([3, 4]))

        let c = asStrided(a, [3, 4], strides: [4, 1])
        assertEqual(b, c)
    }

    func testAsStridedTranspose() {
        // strides in the reverse order is a transpose
        let a = MLXArray(0 ..< 12, [4, 3])

        let b = asStrided(a, [3, 4], strides: [1, 3])
        assertEqual(b, a.transposed())
    }

    func testAsStridedOffset() {
        let a = MLXArray(0 ..< 16, [4, 4])

        let b = asStrided(a, [3, 4], offset: 1)
        assertEqual(b, MLXArray(1 ..< 13, [3, 4]))
    }

    func testTensordot() {
        let a = MLXArray(0 ..< 60, [3, 4, 5]).asType(.float32)
        let b = MLXArray(0 ..< 24, [4, 3, 2]).asType(.float32)
        let c = tensordot(a, b, axes: ([1, 0], [0, 1]))

        let expected = MLXArray(
            converting: [
                4400.0, 4730.0,
                4532.0, 4874.0,
                4664.0, 5018.0,
                4796.0, 5162.0,
                4928.0, 5306.0,
            ], [5, 2])
        assertEqual(c, expected)
    }

    func testConvertScalarInt() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 25
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .int32)
    }

    func testConvertScalarFloat16() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float16(2.5)
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float16)
    }

    func testConvertScalarFloat() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float(2.5)
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float32)
    }

    func testConvertScalarDouble() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 2.5
        XCTAssertEqual(b.dtype, .bool)
        XCTAssertEqual(c.dtype, .float32)
    }

    func testFlatten() {
        let a = zeros([4, 5, 6, 7])
        let b = flatten(a, startAxis: 1, endAxis: 2)
        let c = unflatten(b, axis: 1, shape: [5, 6])
        assertEqual(a, c)
    }

}
