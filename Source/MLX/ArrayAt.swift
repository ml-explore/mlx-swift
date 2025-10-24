// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Intermediate type for ``MLXArray/at``.
///
/// This type isn't typically used directly, rather it is the return value from the `at` property on MLXArray
/// and provides the subscript.
///
/// ```swift
/// let idx = MLXArray([0, 1, 0, 1])
/// var a2 = MLXArray([0, 0])
/// a2 = a2.at[idx].add(1)
/// ```
///
/// ### See Also
///     - ``MLXArray/at``
///     - ``ArrayAtIndices``
public struct ArrayAt {

    let array: MLXArray

    /// Provide indices for the `at` property:
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([0, 0])
    /// a2 = a2.at[idx].add(1)
    /// ```
    ///
    /// This is specifically the `a2.at[idx]` part.
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    ///     - ``ArrayAtIndices``
    public subscript(indices: any MLXArrayIndex..., stream stream: StreamOrDevice = .default)
        -> ArrayAtIndices
    {
        get {
            ArrayAtIndices(
                array: array,
                indexOperations: indices.map { $0.mlxArrayIndexOperation },
                stream: stream)
        }
    }

    /// Provide indices for the `at` property:
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([0, 0])
    /// a2 = a2.at[idx].add(1)
    /// ```
    ///
    /// This is specifically the `a2.at[idx]` part.
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    ///     - ``ArrayAtIndices``
    public subscript(indices: some Sequence<any MLXArrayIndex>,
        stream stream: StreamOrDevice = .default
    )
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

/// Intermediate type for ``MLXArray/at``.
///
/// This type allows update operations when using `array.at[indices]`, e.g.:
///
/// ```swift
/// let idx = MLXArray([0, 1, 0, 1])
/// var a2 = MLXArray([0, 0])
/// a2 = a2.at[idx].add(1)
/// ```
///
/// ### See Also
///     - ``MLXArray/at``
///     - ``ArrayAt``
public struct ArrayAtIndices {

    let array: MLXArray
    let indexOperations: [MLXArrayIndexOperation]
    let stream: StreamOrDevice

    /// Add values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([0, 0])
    /// a2 = a2.at[idx].add(1)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func add(_ values: some ScalarOrArray) -> MLXArray {
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

    /// Subtract values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([0, 0])
    /// a2 = a2.at[idx].subtract(1)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func subtract(_ values: some ScalarOrArray) -> MLXArray {
        add(-values.asMLXArray(dtype: array.dtype))
    }

    /// Multiply values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([1, 1])
    /// a2 = a2.at[idx].multiply(2)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func multiply(_ values: some ScalarOrArray) -> MLXArray {
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

    /// Divide values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([1, 1])
    /// a2 = a2.at[idx].divide(2)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func divide(_ values: some ScalarOrArray) -> MLXArray {
        multiply(values.asMLXArray(dtype: array.dtype).reciprocal())
    }

    /// Update to minimum values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([1, 1])
    /// a2 = a2.at[idx].minimum(2)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func minimum(_ values: some ScalarOrArray) -> MLXArray {
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

    /// Update to maximum values via `at[]` operator.
    ///
    /// ```swift
    /// let idx = MLXArray([0, 1, 0, 1])
    /// var a2 = MLXArray([1, 1])
    /// a2 = a2.at[idx].maximum(2)
    /// ```
    ///
    /// ### See Also
    ///     - ``MLXArray/at``
    public func maximum(_ values: some ScalarOrArray) -> MLXArray {
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
