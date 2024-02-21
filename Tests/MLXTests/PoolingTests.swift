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
}
