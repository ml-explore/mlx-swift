// Copyright © 2024 Apple Inc.

import Foundation
import MLX

open class Pool: Module, UnaryLayer {

    public let kernelSize: [Int]
    public let stride: [Int]
    public let axes: [Int]
    public let poolingOp: (MLXArray, [Int]) -> MLXArray

    init(
        kernelSize: [Int],
        stride: [Int],
        poolingOp: @escaping (MLXArray, [Int]) -> MLXArray
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.axes = Array((-1 * kernelSize.count - 1) ..< -1)
        self.poolingOp = poolingOp
    }

    open func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        let shape = x.shape

        var finalShape = [shape[0]]
        finalShape += zip(zip(shape.dropFirst().dropLast(), kernelSize), stride).map {
            (t, stride) in
            let (size, window) = t
            return (size - window) / stride + 1
        }
        finalShape += kernelSize
        finalShape += [shape.last!]

        let strides = (shape + [1]).reversed().reduce([] as [Int]) { partial, a in
            guard let element = partial.last else { return [a] }
            return partial + [a * element]
        }.reversed().dropFirst()
        let middleStrides = strides.dropFirst().dropLast()
        var finalStrides: [Int] = [strides.first!]
        finalStrides += zip(middleStrides, stride).map { $0 * $1 }
        finalStrides += middleStrides
        finalStrides += [1]

        let strided = asStrided(x, finalShape, strides: finalStrides)
        return poolingOp(strided, axes)
    }
}

/// Applies 1-dimensional max pooling.
open class MaxPool1d: Pool {

    /// Applies 1-dimensional max pooling.
    ///
    /// The input is expected to be `NLC`. The output will have the same N/C dimensions with the new L = floor((L - kernel)/stride) + 1
    ///
    /// See [MaxPool1d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool1d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    public init(kernelSize: Int, stride: Int) {
        super.init(kernelSize: [kernelSize], stride: [stride], poolingOp: { $0.max(axes: $1) })
    }
}

/// Applies 2-dimensional max pooling.
open class MaxPool2d: Pool {

    /// Applies 2-dimensional max pooling.
    ///
    /// The input is expected to be `NHWC`. The output will have the same N/C dimensions with the new H/W = floor((H/W - kernel)/stride) + 1
    ///
    /// See [MaxPool2d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    public init(kernelSize: IntOrPair, stride: IntOrPair) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second],
            stride: [stride.first, stride.second],
            poolingOp: { $0.max(axes: $1) })
    }
}

/// Applies 1-dimensional average pooling.
open class AvgPool1d: Pool {

    /// Applies 1-dimensional average pooling.
    ///
    /// The input is expected to be `NLC`. The output will have the same N/C dimensions with the new L = floor((L - kernel)/stride) + 1
    ///
    /// See [AvgPool2d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    public init(kernelSize: Int, stride: Int) {
        super.init(kernelSize: [kernelSize], stride: [stride], poolingOp: { $0.mean(axes: $1) })
    }
}

/// Applies 2-dimensional average pooling.
open class AvgPool2d: Pool {

    /// Applies 2-dimensional average pooling.
    ///
    /// The input is expected to be `NHWC`. The output will have the same N/C dimensions with the new H/W = floor((H/W - kernel)/stride) + 1
    ///
    /// See [AvgPool2d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool2d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    public init(kernelSize: IntOrPair, stride: IntOrPair) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second],
            stride: [stride.first, stride.second],
            poolingOp: { $0.mean(axes: $1) })
    }
}
