// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import XCTest

// not tests that run -- these are tests to make sure we can
// compile subclasses outside the package they are defined in

public class AdjustedRMSNorm: MLXNN.RMSNorm {
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x) + 1
    }
}

public class TrivialOptimizer: MLXOptimizers.OptimizerBase<MLXArray> {

    public override func newState(parameter: MLXArray) -> MLXArray {
        ones(like: parameter)
    }

    public override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        (gradient, state)
    }
}

public class AdjustedSGD: MLXOptimizers.SGD {

    public override func applySingle(gradient: MLXArray, parameter: MLXArray, state: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        super.applySingle(gradient: gradient + 1, parameter: parameter, state: state)
    }

}
