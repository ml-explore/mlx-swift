// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class MLXNNPoolingTests: XCTestCase {
    func testMaxPooling1dStride1() {
        let input = MLXArray(0 ..< 4, [1, 4, 1])
        let pool = MaxPool1d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([1, 2, 3], [1, 3, 1]))
    }

    func testMaxPooling1dStride2() {
        let input = MLXArray(0 ..< 8, [2, 4, 1])
        let pool = MaxPool1d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([1, 3, 5, 7], [2, 2, 1]))
    }

    func testMaxPoolingStride1() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MaxPool2d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([5, 6, 7, 9, 10, 11, 13, 14, 15], [1, 3, 3, 1]))
    }

    func testMaxPoolingStride2() {
        let input = MLXArray(0 ..< 32, [2, 4, 4, 1])
        let pool = MaxPool2d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([5, 7, 13, 15, 21, 23, 29, 31], [2, 2, 2, 1]))
    }

    func testAvgPooling1dStride1() {
        let input = MLXArray(0 ..< 4, [1, 4, 1])
        let pool = AvgPool1d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(
            output,
            MLXArray(converting: [0.5, 1.5, 2.5], [1, 3, 1]))
    }

    func testAvgPooling1dStride2() {
        let input = MLXArray(0 ..< 8, [2, 4, 1])
        let pool = AvgPool1d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(
            output,
            MLXArray(converting: [0.5, 2.5, 4.5, 6.5], [2, 2, 1]))
    }

    func testAvgPoolingStride1() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = AvgPool2d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(
            output,
            MLXArray(converting: [2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5], [1, 3, 3, 1]))
    }

    func testAvgPoolingStride2() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = AvgPool2d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray(converting: [2.5, 4.5, 10.5, 12.5], [1, 2, 2, 1]))
    }

    // MARK: - padding
    //
    // padding shifts every spatial dimension and must leave the batch and channel
    // dimensions alone: the output is
    // floor((size + 2 * padding - kernel) / stride) + 1

    func testMaxPooling1dPadding() {
        let input = MLXArray(0 ..< 4, [1, 4, 1])
        let pool = MaxPool1d(kernelSize: 2, stride: 2, padding: 1)
        let output = pool(input)

        // padded with -inf: [-inf, 0, 1, 2, 3, -inf]
        XCTAssertEqual(output.shape, [1, 3, 1])
        assertEqual(output, MLXArray([0, 2, 3], [1, 3, 1]))
    }

    func testAvgPooling1dPadding() {
        let input = MLXArray(0 ..< 4, [1, 4, 1]).asType(.float32)
        let pool = AvgPool1d(kernelSize: 2, stride: 2, padding: 1)
        let output = pool(input)

        // padded with 0, which is included in the average: [0, 0, 1, 2, 3, 0]
        XCTAssertEqual(output.shape, [1, 3, 1])
        assertEqual(output, MLXArray(converting: [0, 1.5, 1.5], [1, 3, 1]))
    }

    func testMaxPooling2dPadding() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MaxPool2d(kernelSize: 3, stride: 2, padding: 1)
        let output = pool(input)

        XCTAssertEqual(output.shape, [1, 2, 2, 1])
        assertEqual(output, MLXArray([5, 7, 13, 15], [1, 2, 2, 1]))
    }

    func testAvgPooling2dPadding() {
        let input = MLXArray(0 ..< 4, [1, 2, 2, 1]).asType(.float32)
        let pool = AvgPool2d(kernelSize: 2, stride: 2, padding: 1)
        let output = pool(input)

        XCTAssertEqual(output.shape, [1, 2, 2, 1])
        assertEqual(output, MLXArray(converting: [0, 0.25, 0.5, 0.75], [1, 2, 2, 1]))
    }

    func testPoolingPaddingKeepsBatchAndChannels() {
        // the padding must not touch the batch or channel dimensions: this shape
        // regressed to [2, 3, 4, 6] when the pad widths were built per pair
        // instead of per dimension
        let input = MLXArray(0 ..< (2 * 8 * 8 * 4), [2, 8, 8, 4]).asType(.float32)

        XCTAssertEqual(
            MaxPool2d(kernelSize: 3, stride: 2, padding: 1)(input).shape, [2, 4, 4, 4])
        XCTAssertEqual(
            AvgPool2d(kernelSize: 3, stride: 2, padding: 1)(input).shape, [2, 4, 4, 4])
        XCTAssertEqual(
            MaxPool2d(kernelSize: 2, stride: 2, padding: 0)(input).shape, [2, 4, 4, 4])
    }

    func testPooling1dAnd3dPaddingShapes() {
        let input1d = MLXArray(0 ..< (2 * 16 * 4), [2, 16, 4]).asType(.float32)
        XCTAssertEqual(
            MaxPool1d(kernelSize: 3, stride: 2, padding: 1)(input1d).shape, [2, 8, 4])
        XCTAssertEqual(
            AvgPool1d(kernelSize: 3, stride: 2, padding: 1)(input1d).shape, [2, 8, 4])

        let input3d = MLXArray(0 ..< (2 * 4 * 8 * 8 * 4), [2, 4, 8, 8, 4]).asType(.float32)
        XCTAssertEqual(
            MaxPool3d(kernelSize: 3, stride: 2, padding: 1)(input3d).shape, [2, 2, 4, 4, 4])
        XCTAssertEqual(
            AvgPool3d(kernelSize: 3, stride: 2, padding: 1)(input3d).shape, [2, 2, 4, 4, 4])
    }

    func testPoolingAsymmetricPaddingPerAxis() {
        let input = MLXArray(0 ..< (1 * 8 * 6 * 1), [1, 8, 6, 1]).asType(.float32)

        // padding only the width axis
        XCTAssertEqual(
            MaxPool2d(kernelSize: 2, stride: 2, padding: [0, 1])(input).shape, [1, 4, 4, 1])
        // and only the height axis
        XCTAssertEqual(
            MaxPool2d(kernelSize: 2, stride: 2, padding: [1, 0])(input).shape, [1, 5, 3, 1])
    }
}
