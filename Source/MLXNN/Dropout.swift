// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Randomly zero a portion of the elements during training.
///
/// The remaining elements are multiplied with `1 / (1-p)` where
/// `p` is the probability of zeroing an element. This is done so the
/// expected value of a given element will remain the same.
///
/// ### See Also
/// - ``Dropout2d``
/// - ``Dropout3d``
open class Dropout: Module, UnaryLayer {

    public let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        if p1 == 1 || !self.training {
            return x
        }

        let mask = MLXRandom.bernoulli(p1, x.shape)
        return (mask * x) * (1 / p1)
    }
}

/// Apply 2D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e. the input shape should be
/// `NWHC` or `WHC` where:`N` is the batch dimension,`H` is the input
/// image height,`W` is the input image width, and`C` is the number of
/// input channels
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// beneficial for early convolution layers where adjacent pixels are
/// correlated. In such case, traditional dropout may not effectively
/// regularize activations. For more details, see [1].
///
/// [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
/// Efficient Object Localization Using Convolutional Networks. CVPR 2015.
///
/// ### See Also
/// - ``Dropout``
/// - ``Dropout3d``
open class Dropout2d: Module, UnaryLayer {

    public let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ndim = x.ndim
        precondition(ndim == 3 || ndim == 4)

        if p1 == 1 || !self.training {
            return x
        }

        // Dropout is applied on the whole channel
        // 3D input: (1, 1, C)
        // 4D input: (B, 1, 1, C)

        var maskShape = x.shape
        maskShape[maskShape.endIndex - 2] = 1
        maskShape[maskShape.endIndex - 3] = 1

        let mask = MLXRandom.bernoulli(p1, maskShape)
        return (mask * x) * (1 / p1)
    }
}

/// Apply 3D channel-wise dropout during training.
///
/// Randomly zero out entire channels independently with probability `p`.
/// This layer expects the channels to be last, i.e., the input shape should be
/// `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
/// `H` is the input image height, `W` is the input image width, and `C` is
/// the number of input channels.
///
/// The remaining channels are scaled by `1 / (1-p)` to
/// maintain the expected value of each element. Unlike traditional dropout,
/// which zeros individual entries, this layer zeros entire channels. This is
/// often beneficial for convolutional layers processing 3D data, like in
/// medical imaging or video processing.
///
/// ### See Also
/// - ``Dropout``
/// - ``Dropout2d``
open class Dropout3d: Module, UnaryLayer {

    public let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ndim = x.ndim
        precondition(ndim == 4 || ndim == 5)

        if p1 == 1 || !self.training {
            return x
        }

        // Dropout is applied on the whole channel
        // 4D input: (1, 1, 1, C)
        // 5D input: (B, 1, 1, 1, C)

        var maskShape = x.shape
        maskShape[maskShape.endIndex - 2] = 1
        maskShape[maskShape.endIndex - 3] = 1
        maskShape[maskShape.endIndex - 4] = 1

        let mask = MLXRandom.bernoulli(p1, maskShape)
        return (mask * x) * (1 / p1)
    }
}
