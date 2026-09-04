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

    /// The `mlx_slice_update_*` functions all share this signature.
    private typealias SliceUpdateReduce = (
        UnsafeMutablePointer<mlx_array>?, mlx_array, mlx_array,
        UnsafePointer<Int32>?, Int, UnsafePointer<Int32>?, Int, UnsafePointer<Int32>?, Int,
        mlx_stream
    ) -> Int32

    /// Apply `values` to the indexed region, preferring a slice update over a scatter.
    ///
    /// Mirrors the structure of `mlx_add_item` and friends: if the indices describe a pure
    /// slice, use the corresponding reducing slice update, otherwise fall back to a scatter.
    ///
    /// - Parameters:
    ///   - values: the update value
    ///   - sliceUpdate: the `mlx_slice_update_*` function for this reduction
    ///   - scatter: the `mlx_scatter_*` function for this reduction
    ///   - elementwise: the whole-array fallback when the indices select everything
    private func update(
        _ values: some ScalarOrArray,
        sliceUpdate: SliceUpdateReduce,
        scatter: (
            UnsafeMutablePointer<mlx_array>?, mlx_array, mlx_vector_array, mlx_array,
            UnsafePointer<Int32>?, Int, mlx_stream
        ) -> Int32,
        elementwise: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let values = values.asMLXArray(dtype: array.dtype)

        switch sliceUpdateArguments(
            src: array, operations: indexOperations, update: values, stream: stream)
        {
        case .slice(let update, let starts, let ends, let strides):
            var result = mlx_array_new()
            _ = sliceUpdate(
                &result, array.ctx, update.ctx, starts, starts.count, ends, ends.count, strides,
                strides.count, stream.ctx)
            return MLXArray(result)

        case .broadcast(let update):
            return elementwise(array, update)

        case nil:
            let (indices, update, axes) = scatterArguments(
                src: array, operations: indexOperations, update: values, stream: stream)

            if !indices.isEmpty {
                let indices_vector = new_mlx_vector_array(indices)
                defer { mlx_vector_array_free(indices_vector) }

                var result = mlx_array_new()
                _ = scatter(
                    &result, array.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)

                return MLXArray(result)
            } else {
                return elementwise(array, update)
            }
        }
    }

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
        update(
            values, sliceUpdate: mlx_slice_update_add, scatter: mlx_scatter_add,
            elementwise: { $0 + $1 })
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
        update(
            values, sliceUpdate: mlx_slice_update_prod, scatter: mlx_scatter_prod,
            elementwise: { $0 * $1 })
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
        update(
            values, sliceUpdate: mlx_slice_update_min, scatter: mlx_scatter_min,
            elementwise: { MLX.minimum($0, $1) })
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
        update(
            values, sliceUpdate: mlx_slice_update_max, scatter: mlx_scatter_max,
            elementwise: { MLX.maximum($0, $1) })
    }

}
