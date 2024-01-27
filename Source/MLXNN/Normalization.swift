// Copyright © 2024 Apple Inc.

import Foundation
import MLX

public class RMSNorm: Module, UnaryModel {

    let weight: MLXArray
    let eps: Float

    public init(_ dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public override func describeExtra(_ indent: Int) -> String {
        "(dimensions=\(weight.dim(0)), eps=\(self.eps))"
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // S is 1/sqrt(N) where N is the size of the features of x and is used
        // to compute a numerically more stable RMS of x by multiplying with S
        // first and summing.
        //
        // This way we prefer underflow over overflow which is controlled with
        // the parameter epsilon anyway.

        let S = 1.0 / sqrt(Float(x.dim(-1)))

        var n = (x * S).square().sum(axis: -1, keepDims: true)
        n = rsqrt(n + eps)

        return weight * x * n
    }
}
