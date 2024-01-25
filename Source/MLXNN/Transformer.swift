import Foundation
import MLX

public class MultiHeadAttention: Module {

    let numHeads: Int

    let queryProjection: Linear
    let keyProjection: Linear
    let valueProjection: Linear
    let outProjection: Linear

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
        self.queryProjection = Linear(queryInputDimensions, dimensions, bias: bias)
        self.keyProjection = Linear(keyInputDimensions, dimensions, bias: bias)
        self.valueProjection = Linear(valueInputDimensions, valueDimensions, bias: bias)
        self.outProjection = Linear(valueDimensions, valueOutputDimensions, bias: bias)
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
        var mask = indices.reshaped(-1, 1) < indices.reshaped(1, -1)
        mask = mask.asType(dtype) * -1e9
        return mask
    }
}
