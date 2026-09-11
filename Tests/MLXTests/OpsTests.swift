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

    func testTensordotDefaultAxes() {
        // axes defaults to 2 (as in numpy and python mlx): sum over the last two
        // dimensions of a and the first two of b
        let a = MLXArray(0 ..< 24, [2, 3, 4]).asType(.float32)
        let b = MLXArray(0 ..< 60, [3, 4, 5]).asType(.float32)

        assertEqual(tensordot(a, b), tensordot(a, b, axes: 2))
        XCTAssertEqual(tensordot(a, b).shape, [2, 5])
    }

    func testNanToNumDefaults() {
        // by default infinities become the largest finite value for the dtype
        // (matching python mlx) and NaN becomes 0
        let a = MLXArray(
            [1.5, Float.nan, Float.infinity, -Float.infinity] as [Float])

        let result = nanToNum(a)
        XCTAssertEqual(result.dtype, .float32)
        assertEqual(
            result,
            MLXArray(
                [
                    1.5, 0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude,
                ] as [Float]))

        // and they can be replaced explicitly
        assertEqual(
            nanToNum(a, nan: -1, posInf: 100, negInf: -100),
            MLXArray([1.5, -1, 100, -100] as [Float]))
    }

    func testNanToNumDefaultsFloat16() {
        // the replacement follows the dtype
        let a = MLXArray([Float.infinity, -Float.infinity] as [Float]).asType(.float16)
        let limit = Float(DType.float16.finfo!.max)

        let result = nanToNum(a)
        XCTAssertEqual(result.dtype, .float16)
        XCTAssertEqual(result[0].item(Float.self), limit)
        XCTAssertEqual(result[1].item(Float.self), -limit)
    }

    func testConvolveModeShapes() {
        // matches numpy/python mlx: full is M + K - 1, same is M, valid is M - K + 1
        let a = MLXArray(0 ..< 20).asType(.float32)

        for kernelSize in [3, 4, 5, 6] {
            let v = MLXArray(1 ..< (kernelSize + 1)).asType(.float32)

            XCTAssertEqual(
                convolve(a, v, mode: .full).shape, [a.size + kernelSize - 1],
                "full, kernel \(kernelSize)")
            XCTAssertEqual(
                convolve(a, v, mode: .same).shape, [a.size], "same, kernel \(kernelSize)")
            XCTAssertEqual(
                convolve(a, v, mode: .valid).shape, [a.size - kernelSize + 1],
                "valid, kernel \(kernelSize)")
        }
    }

    func testConvolveSameAndValidAreWindowsOfFull() {
        // `same` is the centered `a.size` window of the full convolution and
        // `valid` is the window with no zero padding at all -- true for both odd
        // and even sized weights (even sizes need asymmetric padding)
        let a = MLXArray(0 ..< 20).asType(.float32)

        for kernelSize in [3, 4, 5, 6] {
            let v = MLXArray(1 ..< (kernelSize + 1)).asType(.float32)
            let full = convolve(a, v, mode: .full)

            let sameStart = (kernelSize - 1) / 2
            assertEqual(
                convolve(a, v, mode: .same), full[sameStart ..< (sameStart + a.size)])

            assertEqual(
                convolve(a, v, mode: .valid), full[(kernelSize - 1) ..< a.size])
        }
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

    func testQuantized() {
        let a = MLXRandom.uniform(low: 0, high: 1, [8, 64])

        let (wq1, s1, b1) = quantized(a, mode: .affine)
        XCTAssertEqual(wq1.dtype, .uint32)
        XCTAssertEqual(wq1.shape, [8, 8])
        XCTAssertEqual(s1.shape, [8, 1])
        if let b1 {
            XCTAssertEqual(b1.shape, [8, 1])
        } else {
            XCTFail("b1 should not be nil")
        }

        let (wq2, s2, b2) = quantized(a, groupSize: 32, mode: .mxfp4)
        XCTAssertEqual(wq2.dtype, .uint32)
        XCTAssertEqual(wq2.shape, [8, 8])
        XCTAssertEqual(s2.shape, [8, 2])
        XCTAssertNil(b2)
    }

}
