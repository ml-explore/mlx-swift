// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
///
/// ### See Also
/// - ``Conv2d``
/// - ``init(inputChannels:outputChannels:kernelSize:stride:padding:bias:)``
open class Conv1d: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let stride: Int

    /// Applies a 1-dimensional convolution over the multi-channel input sequence.
    ///
    /// The channels are expected to be last i.e. the input shape should be `NLC` where:
    ///
    /// - `N` is the batch dimension
    /// - `L` is the sequence length
    /// - `C` is the number of input channels
    ///
    /// - Parameters:
    ///   - inputChannels: number of input channels (`C` from the discussion)
    ///   - outputChannels: number of output channels
    ///   - kernelSize: size of the convolution filters
    ///   - stride: stride when applying the filter
    ///   - padding: many positions to 0-pad the input with
    ///   - bias: if `true` add a learnable bias to the output
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

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding)
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// Applies a 2-dimensional convolution over the multi-channel input image.
///
/// ### See Also
/// - ``Conv1d``
/// - ``init(inputChannels:outputChannels:kernelSize:stride:padding:bias:)``
open class Conv2d: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: (Int, Int)
    public let stride: (Int, Int)

    /// Applies a 2-dimensional convolution over the multi-channel input image.
    ///
    /// The channels are expected to be last i.e. the input shape should be `NHWC` where:
    ///
    /// - `N` is the batch dimension
    /// - `H` is the input image height
    /// - `W` is the input image width
    /// - `C` is the number of input channels
    ///
    /// - Parameters:
    ///   - inputChannels: number of input channels (`C` from the discussion)
    ///   - outputChannels: number of output channels
    ///   - kernelSize: size of the convolution filters
    ///   - stride: stride when applying the filter
    ///   - padding: many positions to 0-pad the input with
    ///   - bias: if `true` add a learnable bias to the output
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

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: .init(stride), padding: .init(padding))
        if let bias {
            y = y + bias
        }
        return y
    }
}
