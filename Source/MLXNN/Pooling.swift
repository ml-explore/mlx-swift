// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Abstract pooling layer.
///
/// ### See Also
/// - ``MaxPool1d``
/// - ``MaxPool2d``
/// - ``MaxPool3d``
/// - ``AvgPool1d``
/// - ``AvgPool2d``
/// - ``AvgPool3d``
open class Pool: Module, UnaryLayer {

    public let kernelSize: [Int]
    public let stride: [Int]
    public let padding: [Int]
    public let paddingValue: Float
    public let axes: [Int]
    public let poolingOp: (MLXArray, [Int]) -> MLXArray

    init(
        kernelSize: [Int],
        stride: [Int],
        padding: [Int],
        paddingValue: Float,
        poolingOp: @escaping (MLXArray, [Int]) -> MLXArray
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.paddingValue = paddingValue
        self.axes = Array((-1 * kernelSize.count - 1) ..< -1)
        self.poolingOp = poolingOp
    }

    open func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        var input = x

        // Apply padding if any padding value is greater than 0
        if padding.contains(where: { $0 > 0 }) {
            // batch and channel dimension get no padding
            let padWidths: [IntOrPair] =
                [0, 0] + padding.map { .init($0) } + [0, 0]
            let paddingValue = paddingValue.asMLXArray(dtype: input.dtype)
            input = padded(input, widths: padWidths, mode: .constant, value: paddingValue)
        }

        let shape = input.shape

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

        let strided = asStrided(input, finalShape, strides: finalStrides)
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
    ///   - padding: how much negative infinity padding to apply to the input. The padding is applied to both sides of the spatial axis. Default: 0
    public init(kernelSize: Int, stride: Int, padding: Int = 0) {
        super.init(
            kernelSize: [kernelSize],
            stride: [stride],
            padding: [padding],
            paddingValue: -Float.infinity,
            poolingOp: { $0.max(axes: $1) })
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
    ///   - padding: how much negative infinity padding to apply to the input. The padding is applied on both sides of the height and width axis. Default: 0
    public init(kernelSize: IntOrPair, stride: IntOrPair, padding: IntOrPair = 0) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second],
            stride: [stride.first, stride.second],
            padding: [padding.first, padding.second],
            paddingValue: -Float.infinity,
            poolingOp: { $0.max(axes: $1) })
    }
}

/// Applies 3-dimensional max pooling.
open class MaxPool3d: Pool {

    /// Applies 3-dimensional max pooling.
    ///
    /// The input is expected to be `NDHWC`. The output will have the same N/C dimensions with the new D/H/W = floor((D/H/W - kernel)/stride) + 1
    ///
    /// The parameters `kernelSize`, `stride`, and `padding` can either be:
    /// - a single `Int` -- in which case the same value is used for the depth, height, and width axis
    /// - a tuple of three `Int`s -- in which case, the first `Int` is used for the depth axis, the second `Int` for the height axis, and the third `Int` for the width axis
    ///
    /// See [MaxPool3d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool3d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    ///   - padding: how much negative infinity padding to apply to the input. The padding is applied on both sides of the depth, height and width axis. Default: 0
    public init(kernelSize: IntOrTriple, stride: IntOrTriple, padding: IntOrTriple = 0) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second, kernelSize.third],
            stride: [stride.first, stride.second, stride.third],
            padding: [padding.first, padding.second, padding.third],
            paddingValue: -Float.infinity,
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
    ///   - padding: how much zero padding to apply to the input. The padding is applied to both sides of the spatial axis. Default: 0
    public init(kernelSize: Int, stride: Int, padding: Int = 0) {
        super.init(
            kernelSize: [kernelSize],
            stride: [stride],
            padding: [padding],
            paddingValue: 0,
            poolingOp: { $0.mean(axes: $1) })
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
    ///   - padding: how much zero padding to apply to the input. The padding is applied on both sides of the height and width axis. Default: 0
    public init(kernelSize: IntOrPair, stride: IntOrPair, padding: IntOrPair = 0) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second],
            stride: [stride.first, stride.second],
            padding: [padding.first, padding.second],
            paddingValue: 0,
            poolingOp: { $0.mean(axes: $1) })
    }
}

/// Applies 3-dimensional average pooling.
open class AvgPool3d: Pool {

    /// Applies 3-dimensional average pooling.
    ///
    /// The input is expected to be `NDHWC`. The output will have the same N/C dimensions with the new D/H/W = floor((D/H/W - kernel)/stride) + 1
    ///
    /// The parameters `kernelSize`, `stride`, and `padding` can either be:
    /// - a single `Int` -- in which case the same value is used for the depth, height, and width axis
    /// - a tuple of three `Int`s -- in which case, the first `Int` is used for the depth axis, the second `Int` for the height axis, and the third `Int` for the width axis
    ///
    /// See [AvgPool3d python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool3d.html) for more information.
    ///
    /// - Parameters:
    ///   - kernelSize: size of the pooling window
    ///   - stride: stride of the pooling window
    ///   - padding: how much zero padding to apply to the input. The padding is applied on both sides of the depth, height and width axis. Default: 0
    public init(kernelSize: IntOrTriple, stride: IntOrTriple, padding: IntOrTriple = 0) {
        super.init(
            kernelSize: [kernelSize.first, kernelSize.second, kernelSize.third],
            stride: [stride.first, stride.second, stride.third],
            padding: [padding.first, padding.second, padding.third],
            paddingValue: 0,
            poolingOp: { $0.mean(axes: $1) })
    }
}
