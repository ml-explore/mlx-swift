// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

public struct ArrayAt {

    let array: MLXArray

    public subscript(indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default)
        -> ArrayAtIndices
    {
        get {
            ArrayAtIndices(
                array: array,
                indexOperations: indices.map { $0.mlxArrayIndexOperation },
                stream: stream)
        }
    }

    public subscript(indices: [MLXArrayIndex], stream stream: StreamOrDevice = .default)
        -> ArrayAtIndices
    {
        get {
            ArrayAtIndices(
                array: array,
                indexOperations: indices.map { $0.mlxArrayIndexOperation },
                stream: stream)
        }
    }
}

public struct ArrayAtIndices {

    let array: MLXArray
    let indexOperations: [MLXArrayIndexOperation]
    let stream: StreamOrDevice

    public func add(_ values: ScalarOrArray) -> MLXArray {
        let values = values.asMLXArray(dtype: array.dtype)
        let (indices, update, axes) = scatterArguments(
            src: array, operations: indexOperations, update: values, stream: stream)

        if !indices.isEmpty {
            let indices_vector = new_mlx_vector_array(indices)
            defer { mlx_vector_array_free(indices_vector) }

            var result = mlx_array_new()
            mlx_scatter_add(
                &result, array.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)

            return MLXArray(result)
        } else {
            return array + update
        }
    }

    public func subtract(_ values: ScalarOrArray) -> MLXArray {
        add(-values.asMLXArray(dtype: array.dtype))
    }

    public func multiply(_ values: ScalarOrArray) -> MLXArray {
        let values = values.asMLXArray(dtype: array.dtype)
        let (indices, update, axes) = scatterArguments(
            src: array, operations: indexOperations, update: values, stream: stream)

        if !indices.isEmpty {
            let indices_vector = new_mlx_vector_array(indices)
            defer { mlx_vector_array_free(indices_vector) }

            var result = mlx_array_new()
            mlx_scatter_prod(
                &result, array.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)

            return MLXArray(result)
        } else {
            return array * update
        }
    }

    public func divide(_ values: ScalarOrArray) -> MLXArray {
        multiply(1 / values.asMLXArray(dtype: array.dtype))
    }

    public func minimum(_ values: ScalarOrArray) -> MLXArray {
        let values = values.asMLXArray(dtype: array.dtype)
        let (indices, update, axes) = scatterArguments(
            src: array, operations: indexOperations, update: values, stream: stream)

        if !indices.isEmpty {
            let indices_vector = new_mlx_vector_array(indices)
            defer { mlx_vector_array_free(indices_vector) }

            var result = mlx_array_new()
            mlx_scatter_min(
                &result, array.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)

            return MLXArray(result)
        } else {
            return MLX.minimum(array, update)
        }
    }

    public func maximum(_ values: ScalarOrArray) -> MLXArray {
        let values = values.asMLXArray(dtype: array.dtype)
        let (indices, update, axes) = scatterArguments(
            src: array, operations: indexOperations, update: values, stream: stream)

        if !indices.isEmpty {
            let indices_vector = new_mlx_vector_array(indices)
            defer { mlx_vector_array_free(indices_vector) }

            var result = mlx_array_new()
            mlx_scatter_max(
                &result, array.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)

            return MLXArray(result)
        } else {
            return MLX.maximum(array, update)
        }
    }

}
