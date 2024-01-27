// Copyright © 2024 Apple Inc.

import Foundation
import MLX

public func relu(_ x: MLXArray) -> MLXArray {
    maximum(x, 0)
}

public func leakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    maximum(negativeSlope * x, x)
}

public func logSoftMax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    x - logSumExp(x, axis: axis, keepDims: true)
}

public func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    MLX.where(x .> 0, x, alpha * (exp(x) - 1))
}

public func relu6(_ x: MLXArray) -> MLXArray {
    minimum(maximum(x, 0), 6)
}

public func softPlus(_ x: MLXArray) -> MLXArray {
    logAddExp(x, 0)
}

public func softSign(_ x: MLXArray) -> MLXArray {
    x / (1 + abs(x))
}

public func celu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    maximum(x, 0.0) + alpha * (exp(minimum(x, 0.0) / alpha) - 1)
}

public func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}

public func logSigmoid(_ x: MLXArray) -> MLXArray {
    -softPlus(-x)
}

public func gelu(_ x: MLXArray) -> MLXArray {
    x * (1 + erf(x / sqrt(2))) / 2
}

public func geluApproximate(_ x: MLXArray) -> MLXArray {
    x * sigmoid(1.60033 * x * (1 + 0.0433603 * x.square()))
}

public func geluFastApproximate(_ x: MLXArray) -> MLXArray {
    x * sigmoid(1.773 * x)
}

public func glu(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let pieces = split(x, parts: 2, axis: axis)
    return pieces[0] * sigmoid(pieces[1])
}

public func step(_ x: MLXArray, threshold: Float = 0.0) -> MLXArray {
    MLX.where(x .> threshold, 1, 0)
}

public func selu(_ x: MLXArray) -> MLXArray {
    elu(x, alpha: 1.67326) * 1.0507
}

public func prelu(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    maximum(0, x) + alpha * minimum(0, x)
}

public func mish(_ x: MLXArray) -> MLXArray {
    x * tanh(softPlus(x))
}

public func hardSwish(_ x: MLXArray) -> MLXArray {
    let maxXPlus3 = maximum(x + 3, 0)
    return x * minimum(maxXPlus3, 6) / 6
}

public class GLU: Module, UnaryModel {
    let axis: Int

    public init(axis: Int = -1) {
        self.axis = axis
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        glu(x, axis: axis)
    }
}

public class Sigmoid: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        sigmoid(x)
    }
}

public class Mish: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        mish(x)
    }
}

public class ReLU: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(x)
    }
}

public class LeakyReLU: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        leakyRelu(x)
    }
}

public class Relu6: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu6(x)
    }
}

public class SoftMax: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        softMax(x)
    }
}

public class SoftPlus: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        softPlus(x)
    }
}

public class SoftSign: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        softSign(x)
    }
}

public class CELU: Module, UnaryModel {
    let alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        celu(x, alpha: alpha)
    }
}

public class SiLU: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(x)
    }
}

public class LogSoftMax: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSoftMax(x)
    }
}

public class LogSigmoid: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSigmoid(x)
    }
}

public class PReLU: Module, UnaryModel {

    let weight: MLXArray

    public init(count: Int = 1, value: Float = 0.25) {
        self.weight = MLXArray.full([count], values: MLXArray(value))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        prelu(x, alpha: weight)
    }
}

public class GELU: Module, UnaryModel {

    public enum Approximation {
        case none
        case precise
        case fast
    }

    let approximation: Approximation

    public init(approximation: Approximation = .none) {
        self.approximation = approximation
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch approximation {
        case .none:
            gelu(x)
        case .precise:
            geluApproximate(x)
        case .fast:
            geluFastApproximate(x)
        }
    }
}

/// Applies the hyperbolic tangent function
///
/// See ``tanh(_:stream:)``
public class Tanh: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        tanh(x)
    }
}

/// Applies the hardswish function, element-wise.
///
/// See ``hardSwish(_:)``
public class HardSwish: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        hardSwish(x)
    }
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// See ``step(_:threshold:)``
public class Step: Module, UnaryModel {

    let threshold: Float

    public init(threshold: Float = 0.0) {
        self.threshold = threshold
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        step(x, threshold: threshold)
    }
}

/// Applies the Scaled Exponential Linear Unit.
///
/// See ``selu(_:)``.
public class SELU: Module, UnaryModel {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        selu(x)
    }
}
