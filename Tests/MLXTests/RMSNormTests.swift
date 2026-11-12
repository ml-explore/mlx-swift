// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct RMSNormTests {

    private let eps: Float = 1e-5

    /// Reference implementation: `x * rsqrt(mean(x^2) + eps)` over the last axis.
    private func reference(_ x: MLXArray, eps: Float) -> MLXArray {
        x * rsqrt(mean(square(x), axis: -1, keepDims: true) + eps)
    }

    @Test(.randomSeed(42)) func nilWeightMatchesReference() {
        let x = normal([2, 8, 16])

        let result = rmsNorm(x, eps: eps)

        expectEqual(result, reference(x, eps: eps), rtol: 1e-5, atol: 1e-5)
    }

    @Test(.randomSeed(42)) func nilWeightMatchesOnesWeight() {
        let x = normal([2, 8, 16])

        let noWeight = MLXFast.rmsNorm(x, eps: eps)
        let onesWeight = MLXFast.rmsNorm(x, weight: MLXArray.ones([16]), eps: eps)

        expectEqual(noWeight, onesWeight)
    }

    @Test(.randomSeed(42)) func explicitWeightStillWorks() {
        let x = normal([2, 8, 16])
        let weight = normal([16])

        let result = rmsNorm(x, weight: weight, eps: eps)

        expectEqual(result, reference(x, eps: eps) * weight, rtol: 1e-5, atol: 1e-5)
    }

    /// Without a weight the result keeps the dtype of `x`.  A float32 identity
    /// weight would promote half-precision inputs to float32, see
    /// https://github.com/ml-explore/mlx-swift-lm/issues/751
    @Test(.randomSeed(42), arguments: [DType.float16, .bfloat16, .float32])
    func nilWeightPreservesDType(dtype: DType) {
        let x = normal([2, 8, 16]).asType(dtype)

        let result = rmsNorm(x, eps: eps)

        #expect(result.dtype == dtype)
        expectEqual(
            result.asType(.float32), reference(x.asType(.float32), eps: eps),
            rtol: 1e-2, atol: 1e-2)
    }

    @Test(.randomSeed(42)) func nilWeightGradient() {
        let x = normal([4, 16])

        let gradNoWeight = grad { (x: MLXArray) in
            (rmsNorm(x, eps: eps) * arange(16, dtype: .float32)).sum()
        }(x)
        let gradReference = grad { (x: MLXArray) in
            (reference(x, eps: eps) * arange(16, dtype: .float32)).sum()
        }(x)

        expectEqual(gradNoWeight, gradReference, rtol: 1e-4, atol: 1e-4)
    }
}
