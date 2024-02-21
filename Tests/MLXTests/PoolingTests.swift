// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class MLXNNPoolingTests: XCTestCase {
    func testMinPoolingStride1() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MinPool2d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([0, 1, 2, 4, 5, 6, 8, 9, 10], [1, 3, 3, 1]))
    }

    func testMinPoolingStride2() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MinPool2d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([0, 2, 8, 10], [1, 2, 2, 1]))
    }

    func testMaxPoolingStride1() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MaxPool2d(kernelSize: 2, stride: 1)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([5, 6, 7, 9, 10, 11, 13, 14, 15], [1, 3, 3, 1]))
    }

    func testMaxPoolingStride2() {
        let input = MLXArray(0 ..< 16, [1, 4, 4, 1])
        let pool = MaxPool2d(kernelSize: 2, stride: 2)
        let output = pool.callAsFunction(input)
        assertEqual(output, MLXArray([5, 7, 13, 15], [1, 2, 2, 1]))
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
