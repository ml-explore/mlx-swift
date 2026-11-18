// Copyright © 2024 Apple Inc.

import Foundation
import Testing

@testable import MLX

@Suite
struct OpsTests {

    @Test func asStridedReshape() {
        // just changing the shape and using the default strides is the same as reshape
        let a = MLXArray(0 ..< 12, [4, 3])

        // this uses [4, 1] as the strides
        let b = asStrided(a, [3, 4])
        expectEqual(b, a.reshaped([3, 4]))

        let c = asStrided(a, [3, 4], strides: [4, 1])
        expectEqual(b, c)
    }

    @Test func asStridedTranspose() {
        // strides in the reverse order is a transpose
        let a = MLXArray(0 ..< 12, [4, 3])

        let b = asStrided(a, [3, 4], strides: [1, 3])
        expectEqual(b, a.transposed())
    }

    @Test func asStridedOffset() {
        let a = MLXArray(0 ..< 16, [4, 4])

        let b = asStrided(a, [3, 4], offset: 1)
        expectEqual(b, MLXArray(1 ..< 13, [3, 4]))
    }

    @Test func tensordotExplicitAxes() {
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
        expectEqual(c, expected)
    }

    @Test func tensordotDefaultAxes() {
        // axes defaults to 2 (as in numpy and python mlx): sum over the last two
        // dimensions of a and the first two of b
        let a = MLXArray(0 ..< 24, [2, 3, 4]).asType(.float32)
        let b = MLXArray(0 ..< 60, [3, 4, 5]).asType(.float32)

        expectEqual(tensordot(a, b), tensordot(a, b, axes: 2))
        #expect(tensordot(a, b).shape == [2, 5])
    }

    @Test func nanToNumDefaults() {
        // by default infinities become the largest finite value for the dtype
        // (matching python mlx) and NaN becomes 0
        let a = MLXArray(
            [1.5, Float.nan, Float.infinity, -Float.infinity] as [Float])

        let result = nanToNum(a)
        #expect(result.dtype == .float32)
        expectEqual(
            result,
            MLXArray(
                [
                    1.5, 0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude,
                ] as [Float]))

        // and they can be replaced explicitly
        expectEqual(
            nanToNum(a, nan: -1, posInf: 100, negInf: -100),
            MLXArray([1.5, -1, 100, -100] as [Float]))
    }

    @Test func nanToNumDefaultsFloat16() {
        // the replacement follows the dtype
        let a = MLXArray([Float.infinity, -Float.infinity] as [Float]).asType(.float16)
        let limit = Float(DType.float16.finfo!.max)

        let result = nanToNum(a)
        #expect(result.dtype == .float16)
        #expect(result[0].item(Float.self) == limit)
        #expect(result[1].item(Float.self) == -limit)
    }

    @Test func convolveModeShapes() {
        // matches numpy/python mlx: full is M + K - 1, same is M, valid is M - K + 1
        let a = MLXArray(0 ..< 20).asType(.float32)

        for kernelSize in [3, 4, 5, 6] {
            let v = MLXArray(1 ..< (kernelSize + 1)).asType(.float32)

            #expect(
                convolve(a, v, mode: .full).shape == [a.size + kernelSize - 1],
                "full, kernel \(kernelSize)")
            #expect(
                convolve(a, v, mode: .same).shape == [a.size], "same, kernel \(kernelSize)")
            #expect(
                convolve(a, v, mode: .valid).shape == [a.size - kernelSize + 1],
                "valid, kernel \(kernelSize)")
        }
    }

    @Test func convolveSameAndValidAreWindowsOfFull() {
        // `same` is the centered `a.size` window of the full convolution and
        // `valid` is the window with no zero padding at all -- true for both odd
        // and even sized weights (even sizes need asymmetric padding)
        let a = MLXArray(0 ..< 20).asType(.float32)

        for kernelSize in [3, 4, 5, 6] {
            let v = MLXArray(1 ..< (kernelSize + 1)).asType(.float32)
            let full = convolve(a, v, mode: .full)

            let sameStart = (kernelSize - 1) / 2
            expectEqual(
                convolve(a, v, mode: .same), full[sameStart ..< (sameStart + a.size)])

            expectEqual(
                convolve(a, v, mode: .valid), full[(kernelSize - 1) ..< a.size])
        }
    }

    @Test func convertScalarInt() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 25
        #expect(b.dtype == .bool)
        #expect(c.dtype == .int32)
    }

    @Test func convertScalarFloat16() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float16(2.5)
        #expect(b.dtype == .bool)
        #expect(c.dtype == .float16)
    }

    @Test func convertScalarFloat() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * Float(2.5)
        #expect(b.dtype == .bool)
        #expect(c.dtype == .float32)
    }

    @Test func convertScalarDouble() {
        let a = MLXArray(0 ..< 10)
        let b = a .< (a + 1)
        let c = b * 2.5
        #expect(b.dtype == .bool)
        #expect(c.dtype == .float32)
    }

    @Test func countNonzeroAxes() {
        let a = MLXArray([0, 1, 2, 0, 3, 0], [2, 3])
        let resAxes0 = countNonzero(a, axes: [0])
        let resAxes1 = countNonzero(a, axes: [1])
        let resAxis0 = countNonzero(a, axis: 0)
        let resAxis1 = countNonzero(a, axis: 1)
        let resAll = countNonzero(a)

        expectEqual(resAxes0, [0, 2, 1])
        expectEqual(resAxes0, resAxis0)
        expectEqual(resAxes1, [2, 1])
        expectEqual(resAxes1, resAxis1)
        expectEqual(resAll, 3.asMLXArray(dtype: .int32))
    }

    @Test func flattenUnflatten() {
        let a = zeros([4, 5, 6, 7])
        let b = flatten(a, startAxis: 1, endAxis: 2)
        let c = unflatten(b, axis: 1, shape: [5, 6])
        expectEqual(a, c)
    }

    @Test func quantizedShapes() throws {
        let a = MLXRandom.uniform(low: 0, high: 1, [8, 64])

        let (wq1, s1, b1) = quantized(a, mode: .affine)
        #expect(wq1.dtype == .uint32)
        #expect(wq1.shape == [8, 8])
        #expect(s1.shape == [8, 1])
        let biases1 = try #require(b1, "b1 should not be nil")
        #expect(biases1.shape == [8, 1])

        let (wq2, s2, b2) = quantized(a, groupSize: 32, mode: .mxfp4)
        #expect(wq2.dtype == .uint32)
        #expect(wq2.shape == [8, 8])
        #expect(s2.shape == [8, 2])
        #expect(b2 == nil)
    }

    @Test func convolveValues() {
        let a = MLXArray([1, 2, 3, 4, 5] as [Float])
        let vEven = MLXArray([1, 2, 3, 4] as [Float])
        let vOdd = MLXArray([1, 2, 3] as [Float])

        // Even kernel size
        let fullEven = convolve(a, vEven, mode: .full)
        expectEqual(fullEven, MLXArray([1, 4, 10, 20, 30, 34, 31, 20] as [Float]))

        let validEven = convolve(a, vEven, mode: .valid)
        expectEqual(validEven, MLXArray([20, 30] as [Float]))

        let sameEven = convolve(a, vEven, mode: .same)
        #expect(sameEven.shape == [5])
        expectEqual(sameEven, MLXArray([4, 10, 20, 30, 34] as [Float]))

        // Odd kernel size
        let fullOdd = convolve(a, vOdd, mode: .full)
        expectEqual(fullOdd, MLXArray([1, 4, 10, 16, 22, 22, 15] as [Float]))

        let validOdd = convolve(a, vOdd, mode: .valid)
        expectEqual(validOdd, MLXArray([10, 16, 22] as [Float]))

        let sameOdd = convolve(a, vOdd, mode: .same)
        #expect(sameOdd.shape == [5])
        expectEqual(sameOdd, MLXArray([4, 10, 16, 22, 22] as [Float]))
    }

}
