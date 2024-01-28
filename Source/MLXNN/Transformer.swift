// Copyright © 2024 Apple Inc.

import Foundation
import MLX

public class MultiHeadAttention: Module {

    let numHeads: Int

    @ModuleInfo(key: "query_proj") var queryProjection: Linear
    @ModuleInfo(key: "key_proj") var keyProjection: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    public init(
        dimensions: Int,
        numHeads: Int,
        queryInputDimensions: Int? = nil,
        keyInputDimensions: Int? = nil,
        valueInputDimensions: Int? = nil,
        valueDimensions: Int? = nil,
        valueOutputDimensions: Int? = nil,
        bias: Bool = false
    ) {
        precondition(dimensions % numHeads == 0)

        let queryInputDimensions = queryInputDimensions ?? dimensions
        let keyInputDimensions = keyInputDimensions ?? dimensions
        let valueInputDimensions = valueInputDimensions ?? dimensions
        let valueDimensions = valueDimensions ?? dimensions
        let valueOutputDimensions = valueOutputDimensions ?? dimensions

        self.numHeads = numHeads
        
        self._queryProjection.wrappedValue = Linear(queryInputDimensions, dimensions, bias: bias)
        self._keyProjection.wrappedValue = Linear(keyInputDimensions, dimensions, bias: bias)
        self._valueProjection.wrappedValue = Linear(valueInputDimensions, valueDimensions, bias: bias)
        self._outProjection.wrappedValue = Linear(valueDimensions, valueOutputDimensions, bias: bias)
    }

    public func callAsFunction(
        _ queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var queries = queryProjection(queries)
        var keys = keyProjection(keys)
        let values = valueProjection(values)

        let (B, L) = (queries.dim(0), queries.dim(1))
        let S = keys.dim(1)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 3, 1)
        queries = queries.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

        // Dimensions are [batch x num heads x sequence x hidden dim]
        let scale = sqrt(1 / Float(queries.dim(-1)))
        var scores = (queries * scale).matmul(keys)
        if let mask {
            scores = scores + mask.asType(scores.dtype)
        }
        scores = softMax(scores, axis: -1)
        let valuesHat = matmul(scores, values).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return outProjection(valuesHat)
    }

    public static func createAdditiveCausalMask(_ n: Int, dtype: DType = .float32) -> MLXArray {
        let indices = MLXArray(0 ..< n)
        var mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        mask = mask.asType(dtype) * -1e9
        return mask
    }
}
