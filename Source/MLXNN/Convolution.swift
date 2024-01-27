// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom

public class Conv1d: Module, UnaryModel {

    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let stride: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))

        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding)
        if let bias {
            y = y + bias
        }
        return y
    }
}

public class Conv2d: Module, UnaryModel {

    let weight: MLXArray
    let bias: MLXArray?
    let padding: (Int, Int)
    let stride: (Int, Int)

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize.first * kernelSize.second))

        self.weight = uniform(
            low: -scale, high: scale,
            [outputChannels, kernelSize.first, kernelSize.second, inputChannels])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding.values
        self.stride = stride.values
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: .init(stride), padding: .init(padding))
        if let bias {
            y = y + bias
        }
        return y
    }
}
