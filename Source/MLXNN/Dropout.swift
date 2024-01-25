import Foundation
import MLX
import MLXRandom

class Dropout: Module, UnaryModel {

    let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if p1 == 1 || !self.training {
            return x
        }

        let mask = bernoulli(p1, x.shape)
        return (1 / p1) * mask * x
    }
}

class Dropout2d: Module, UnaryModel {

    let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
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

        let mask = bernoulli(p1, maskShape)
        return (1 / p1) * mask * x
    }
}

class Dropout3d: Module, UnaryModel {

    let p1: Float

    public init(p: Float = 0.5) {
        precondition((0 ..< 1).contains(p))
        self.p1 = 1 - p
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ndim = x.ndim
        precondition(ndim == 3 || ndim == 4)

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

        let mask = bernoulli(p1, maskShape)
        return (1 / p1) * mask * x
    }
}
