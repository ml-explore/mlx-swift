// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

extension MLXArray {

    /// Construct an array of zeros.
    ///
    /// Example:
    ///
    /// ```swift
    /// let z = MLXArray.zeros([5, 10], type: Int.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - type: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``zeros(like:stream:)``
    /// - ``ones(_:type:stream:)``
    static public func zeros(
        _ shape: some Collection<Int>, type: (some HasDType).Type = Float.self,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.zeros(shape, type: type, stream: stream)
    }

    /// Construct an array of zeros with a given ``DType``
    ///
    /// Example:
    ///
    /// ```swift
    /// let z = MLXArray.zeros([5, 10], dtype: .int32)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - dtype: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``zeros(like:stream:)``
    /// - ``ones(_:type:stream:)``
    static public func zeros(
        _ shape: some Collection<Int>, dtype: DType, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.zeros(shape, dtype: dtype, stream: stream)
    }

    /// Construct an array of zeros.
    ///
    /// Example:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [4, 3])
    /// let z = MLXArray.zeros(like: array)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to copy shape and dtype from
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``zeros(_:type:stream:)``
    /// - ``ones(_:type:stream:)``
    static public func zeros(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.zeros(like: array, stream: stream)
    }

    /// Construct an array of ones.
    ///
    /// Example:
    ///
    /// ```swift
    /// let r = MLXArray.ones([5, 10], type: Int.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - type: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ones(like:stream:)``
    /// - ``zeros(_:type:stream:)``
    static public func ones(
        _ shape: some Collection<Int>, type: (some HasDType).Type = Float.self,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.ones(shape, type: type, stream: stream)
    }

    /// Construct an array of ones with a given ``DType``
    ///
    /// Example:
    ///
    /// ```swift
    /// let r = MLXArray.ones([5, 10], dtype: .int32)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - dtype: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ones(like:stream:)``
    /// - ``zeros(_:type:stream:)``
    static public func ones(
        _ shape: some Collection<Int>, dtype: DType, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.ones(shape, dtype: dtype, stream: stream)
    }

    /// Construct an array of ones.
    ///
    /// Example:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [4, 3])
    /// let r = MLXArray.ones(like: array)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to copy shape and dtype from
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ones(_:type:stream:)``
    /// - ``zeros(_:type:stream:)``
    static public func ones(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.ones(like: array, stream: stream)
    }

    /// Create an identity matrix or a general diagonal matrix.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.eye(10)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``identity(_:type:stream:)``
    static public func eye(
        _ n: Int, m: Int? = nil, k: Int = 0, type: (some HasDType).Type = Float.self,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.eye(n, m: m, k: k, type: type, stream: stream)
    }

    /// Create an identity matrix or a general diagonal matrix given a ``DType``.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.eye(10, dtype: .int32)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``identity(_:type:stream:)``
    static public func eye(
        _ n: Int, m: Int? = nil, k: Int = 0, dtype: DType, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.eye(n, m: m, k: k, dtype: dtype, stream: stream)
    }

    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `vals`. If `vals`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [5, 4] array filled with 7
    /// let r = MLXArray.full([5, 4], values: 7, type: Float.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: shape of the output array
    ///     - values: values to be bradcast into the array
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``full(_:values:stream:)``
    /// - ``repeated(_:count:axis:stream:)``
    static public func full(
        _ shape: some Collection<Int>, values: MLXArray, type: (some HasDType).Type = Float.self,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.full(shape, values: values, type: type, stream: stream)
    }

    /// Construct an array with the given value and a given ``DType``.
    ///
    /// Constructs an array of size `shape` filled with `vals`. If `vals`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [5, 4] array filled with 7
    /// let r = MLXArray.full([5, 4], values: 7, dtype: .float32)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: shape of the output array
    ///     - values: values to be bradcast into the array
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``full(_:values:stream:)``
    /// - ``repeated(_:count:axis:stream:)``
    static public func full(
        _ shape: some Collection<Int>, values: MLXArray, dtype: DType,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.full(shape, values: values, dtype: dtype, stream: stream)
    }

    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `vals`. If `vals`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [5, 4] array filled with 7
    /// let r = MLXArray.full([5, 4], values: 7)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: shape of the output array
    ///     - values: values to be bradcast into the array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``full(_:values:type:stream:)``
    /// - ``repeated(_:count:axis:stream:)``
    static public func full(
        _ shape: some Collection<Int>, values: MLXArray, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        MLX.full(shape, values: values, stream: stream)
    }

    /// Create a square identity matrix.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.identity(10)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows and columns in the output
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``eye(_:m:k:type:stream:)``
    static public func identity(
        _ n: Int, type: (some HasDType).Type = Float.self, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.identity(n, type: type, stream: stream)
    }

    /// Create a square identity matrix with a given ``DType``.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.identity(10, dtype: .int32)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows and columns in the output
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``eye(_:m:k:type:stream:)``
    static public func identity(_ n: Int, dtype: DType, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        MLX.identity(n, dtype: dtype, stream: stream)
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` for `BinaryInteger`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = MLXArray.linSpace(0, 50)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - count: number of samples
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``linspace(_:_:count:stream:)-92x6l``
    static public func linspace<T: HasDType>(
        _ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default
    ) -> MLXArray where T: BinaryInteger {
        MLX.linspace(start, stop, count: count, stream: stream)
    }

    /// Generate `num` evenly spaced numbers over interval `[start, stop]` for `BinaryFloatingPoint`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create a 50 element 1-D array with values from 0 to 1
    /// let r = MLXArray.linSpace(0.0, 1.0)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - count: number of samples
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``linspace(_:_:count:stream:)-7m7eg``
    static public func linspace<T: HasDType>(
        _ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default
    ) -> MLXArray where T: BinaryFloatingPoint {
        MLX.linspace(start, stop, count: count, stream: stream)
    }

    /// Generate values in the half-open interval `[0, stop)`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    /// let r = MLXArray.arange(10)
    /// ```
    ///
    /// - Parameters:
    ///     - stop: stop value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:_:step:stream:)``
    static public func arange(_ stop: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.arange(0, stop, stream: stream)
    }

    /// Generate values in the half-open interval `[start, stop)` spaced by `step`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [2, 4, 6, 8]
    /// let r = MLXArray.arange(2, 10, step: 2)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - step: step size (default: 1)
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:_:step:dtype:stream:)-(Int,_,_,_,_)``
    static public func arange(
        _ start: Int, _ stop: Int, step: Int = 1, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.arange(start, stop, step: step, stream: stream)
    }

    /// Generate values in the half-open interval `[0, stop)` with a given ``DType``.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [0.0, 1.0, 2.0, ...] as float32
    /// let r = MLXArray.arange(10, dtype: .float32)
    /// ```
    ///
    /// - Parameters:
    ///     - stop: stop value
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:_:step:dtype:stream:)-(Int,_,_,_,_)``
    static public func arange(_ stop: Int, dtype: DType, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        MLX.arange(0, stop, dtype: dtype, stream: stream)
    }

    /// Generate values in the half-open interval `[start, stop)` spaced by `step` with a given ``DType``.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [2.0, 4.0, 6.0, 8.0] as float32
    /// let r = MLXArray.arange(2, 10, step: 2, dtype: .float32)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - step: step size (default: 1)
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:_:step:stream:)``
    static public func arange(
        _ start: Int, _ stop: Int, step: Int = 1, dtype: DType, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.arange(start, stop, step: step, dtype: dtype, stream: stream)
    }

    /// Generate values in the half-open interval `[0, stop)` (floating point version).
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [0.0, 1.0, 2.0, 3.0, 4.0]
    /// let r = MLXArray.arange(5.0)
    /// ```
    ///
    /// - Parameters:
    ///     - stop: stop value
    ///     - dtype: data type of the output array (default: .float32)
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:_:step:dtype:stream:)-(Double,_,_,_,_)``
    static public func arange(
        _ stop: Double, dtype: DType = .float32, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.arange(stop, dtype: dtype, stream: stream)
    }

    /// Generate values in the half-open interval `[start, stop)` spaced by `step` (floating point version).
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create [0.0, 0.5, 1.0, 1.5, 2.0, ...]
    /// let r = MLXArray.arange(0.0, 5.0, step: 0.5)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - step: step size (default: 1.0)
    ///     - dtype: data type of the output array (default: .float32)
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``arange(_:dtype:stream:)-(Double,_,_)``
    static public func arange(
        _ start: Double, _ stop: Double, step: Double = 1.0, dtype: DType = .float32,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.arange(start, stop, step: step, dtype: dtype, stream: stream)
    }

    /// Repeat an array along a specified axis.
    ///
    /// > Deprected in favor of the more consistently named ``repeated(_:count:axis:stream:)``
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeated(_:count:stream:)``
    /// - ``full(_:values:stream:)``
    @available(*, deprecated, renamed: "repeated(_:count:axis:stream:)")
    static public func `repeat`(
        _ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.repeated(array, count: count, axis: axis, stream: stream)
    }

    /// Repeat a flattened array along axis 0.
    ///
    /// > Deprected in favor of the more consistently named ``repeated(_:count:stream:)``
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeated(_:count:axis:stream:)``
    /// - ``full(_:values:stream:)``
    @available(*, deprecated, renamed: "repeated(_:count:stream:)")
    static public func `repeat`(_ array: MLXArray, count: Int, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        MLX.repeated(array, count: count, stream: stream)
    }

    /// Repeat an array along a specified axis.
    ///
    /// Example:
    ///
    /// ```swift
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let r = MLXArray.repeated(MLXArray(0 ..< 4, [2, 2]), count: 4, axis: 1)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to repeat
    ///     - count: number of times to repeat
    ///     - axis: axis to repeat along
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeated(_:count:stream:)``
    /// - ``full(_:values:stream:)``
    static public func repeated(
        _ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.repeated(array, count: count, axis: axis, stream: stream)
    }

    /// Repeat a flattened array along axis 0.
    ///
    /// Example:
    ///
    /// ```swift
    /// // repeat a 4 element array 4 times along axis 0
    /// let r = MLXArray.repeated(MLXArray(0 ..< 4, [2, 2]), count: 4)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to repeat
    ///     - count: number of times to repeat
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeated(_:count:axis:stream:)``
    /// - ``full(_:values:stream:)``
    static public func repeated(_ array: MLXArray, count: Int, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        MLX.repeated(array, count: count, stream: stream)
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// Example:
    ///
    /// ```swift
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = MLXArray.triangle(5)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    static public func tri(
        _ n: Int, m: Int? = nil, k: Int = 0, type: (some HasDType).Type = Float.self,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.tri(n, m: m, k: k, type: type, stream: stream)
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere and a given ``DType``.
    ///
    /// Example:
    ///
    /// ```swift
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = MLXArray.triangle(5, dtype: .int32)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - dtype: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    static public func tri(
        _ n: Int, m: Int? = nil, k: Int = 0, dtype: DType, stream: StreamOrDevice = .default
    ) -> MLXArray {
        MLX.tri(n, m: m, k: k, dtype: dtype, stream: stream)
    }

}

/// Construct an array of zeros.
///
/// Example:
///
/// ```swift
/// let z = MLXArray.zeros([5, 10], type: Int.self)
/// ```
///
/// - Parameters:
///     - shape: desired shape
///     - type: dtype of the values
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``zeros(like:stream:)``
/// - ``ones(_:type:stream:)``
public func zeros(
    _ shape: some Collection<Int>, type: (some HasDType).Type = Float.self,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_zeros(&result, shape.map { Int32($0) }, shape.count, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array of zeros with a given ``DType``
///
/// Example:
///
/// ```swift
/// let z = MLXArray.zeros([5, 10], dtype: .int32)
/// ```
///
/// - Parameters:
///     - shape: desired shape
///     - dtype: dtype of the values
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``zeros(like:stream:)``
/// - ``ones(_:type:stream:)``
public func zeros(
    _ shape: some Collection<Int>, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_zeros(&result, shape.map { Int32($0) }, shape.count, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array of zeros.
///
/// Example:
///
/// ```swift
/// let array = MLXArray(0 ..< 12, [4, 3])
/// let z = MLXArray.zeros(like: array)
/// ```
///
/// - Parameters:
///     - array: array to copy shape and dtype from
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``zeros(_:type:stream:)``
/// - ``ones(_:type:stream:)``
public func zeros(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_zeros_like(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Construct an array of ones.
///
/// Example:
///
/// ```swift
/// let r = MLXArray.ones([5, 10], type: Int.self)
/// ```
///
/// - Parameters:
///     - shape: desired shape
///     - type: dtype of the values
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``ones(like:stream:)``
/// - ``zeros(_:type:stream:)``
public func ones(
    _ shape: some Collection<Int>, type: (some HasDType).Type = Float.self,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_ones(&result, shape.map { Int32($0) }, shape.count, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array of ones with a given ``DType``
///
/// Example:
///
/// ```swift
/// let r = MLXArray.ones([5, 10], dtype: .int32)
/// ```
///
/// - Parameters:
///     - shape: desired shape
///     - dtype: dtype of the values
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``zeros(like:stream:)``
/// - ``ones(_:type:stream:)``
public func ones(
    _ shape: some Collection<Int>, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_ones(&result, shape.map { Int32($0) }, shape.count, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array of ones.
///
/// Example:
///
/// ```swift
/// let array = MLXArray(0 ..< 12, [4, 3])
/// let r = MLXArray.ones(like: array)
/// ```
///
/// - Parameters:
///     - array: array to copy shape and dtype from
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``ones(_:type:stream:)``
/// - ``zeros(_:type:stream:)``
public func ones(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_ones_like(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Create an identity matrix or a general diagonal matrix.
///
/// Example:
///
/// ```swift
/// //  create [10, 10] array with 1's on the diagonal.
/// let r = MLXArray.eye(10)
/// ```
///
/// - Parameters:
///     - n: number of rows in the output
///     - m: number of columns in the output -- equal to `n` if not specified
///     - k: index of the diagonal
///     - type: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``identity(_:type:stream:)``
public func eye(
    _ n: Int, m: Int? = nil, k: Int = 0, type: (some HasDType).Type = Float.self,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_eye(&result, n.int32, (m ?? n).int32, k.int32, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Create an identity matrix or a general diagonal matrix given a ``DType``.
///
/// Example:
///
/// ```swift
/// //  create [10, 10] array with 1's on the diagonal.
/// let r = MLXArray.eye(10, dtype: .int32)
/// ```
///
/// - Parameters:
///     - n: number of rows in the output
///     - m: number of columns in the output -- equal to `n` if not specified
///     - k: index of the diagonal
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``identity(_:type:stream:)``
public func eye(
    _ n: Int, m: Int? = nil, k: Int = 0, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_eye(&result, n.int32, (m ?? n).int32, k.int32, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array with the given value.
///
/// Constructs an array of size `shape` filled with `vals`. If `vals`
/// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
///
/// Example:
///
/// ```swift
/// //  create [5, 4] array filled with 7
/// let r = MLXArray.full([5, 4], values: 7, type: Float.self)
/// ```
///
/// - Parameters:
///     - shape: shape of the output array
///     - values: values to be bradcast into the array
///     - type: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``full(_:values:stream:)``
/// - ``repeated(_:count:axis:stream:)``
public func full(
    _ shape: some Collection<Int>, values: some ScalarOrArray, type: (some HasDType).Type,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    let values = values.asMLXArray(dtype: nil)
    mlx_full(&result, shape.asInt32, shape.count, values.ctx, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array with the given value and a given ``DType``.
///
/// Constructs an array of size `shape` filled with `vals`. If `vals`
/// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
///
/// Example:
///
/// ```swift
/// //  create [5, 4] array filled with 7
/// let r = MLXArray.full([5, 4], values: 7, dtype: .float32)
/// ```
///
/// - Parameters:
///     - shape: shape of the output array
///     - values: values to be bradcast into the array
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``full(_:values:stream:)``
/// - ``repeated(_:count:axis:stream:)``
public func full(
    _ shape: some Collection<Int>, values: MLXArray, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_full(&result, shape.asInt32, shape.count, values.ctx, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Construct an array with the given value.
///
/// Constructs an array of size `shape` filled with `vals`. If `vals`
/// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
///
/// Example:
///
/// ```swift
/// //  create [5, 4] array filled with 7
/// let r = MLXArray.full([5, 4], values: 7)
/// ```
///
/// - Parameters:
///     - shape: shape of the output array
///     - values: values to be bradcast into the array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``full(_:values:type:stream:)``
/// - ``repeated(_:count:axis:stream:)``
public func full(
    _ shape: some Collection<Int>, values: some ScalarOrArray, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    let values = values.asMLXArray(dtype: nil)
    mlx_full(&result, shape.asInt32, shape.count, values.ctx, values.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Create a square identity matrix.
///
/// Example:
///
/// ```swift
/// //  create [10, 10] array with 1's on the diagonal.
/// let r = MLXArray.identity(10)
/// ```
///
/// - Parameters:
///     - n: number of rows and columns in the output
///     - type: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``eye(_:m:k:type:stream:)``
public func identity(
    _ n: Int, type: (some HasDType).Type = Float.self, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_identity(&result, n.int32, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Create a square identity matrix with a given ``DType``.
///
/// Example:
///
/// ```swift
/// //  create [10, 10] array with 1's on the diagonal.
/// let r = MLXArray.identity(10, dtype: .int32)
/// ```
///
/// - Parameters:
///     - n: number of rows and columns in the output
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``eye(_:m:k:type:stream:)``
public func identity(_ n: Int, dtype: DType, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_identity(&result, n.int32, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate `num` evenly spaced numbers over interval `[start, stop]`.
///
/// Example:
///
/// ```swift
/// // Create a 50 element 1-D array with values from 0 to 50
/// let r = MLXArray.linSpace(0, 50)
/// ```
///
/// - Parameters:
///     - start: start value
///     - stop: stop value
///     - count: number of samples
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``linspace(_:_:count:stream:)-7vj0o``
public func linspace<T: HasDType>(
    _ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default
) -> MLXArray where T: BinaryInteger {
    var result = mlx_array_new()
    mlx_linspace(&result, Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate `num` evenly spaced numbers over interval `[start, stop]`.
///
/// Example:
///
/// ```swift
/// // Create a 50 element 1-D array with values from 0 to 1
/// let r = MLXArray.linSpace(0.0, 1.0)
/// ```
///
/// - Parameters:
///     - start: start value
///     - stop: stop value
///     - count: number of samples
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``linspace(_:_:count:stream:)-6w959``
public func linspace<T: HasDType>(
    _ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default
) -> MLXArray where T: BinaryFloatingPoint {
    var result = mlx_array_new()
    mlx_linspace(&result, Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[0, stop)`.
///
/// Example:
///
/// ```swift
/// // Create [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
/// let r = arange(10)
/// ```
///
/// - Parameters:
///     - stop: stop value
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:_:step:stream:)``
public func arange(_ stop: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(&result, 0, Double(stop), 1, DType.int32.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[start, stop)` spaced by `step`.
///
/// Example:
///
/// ```swift
/// // Create [2, 4, 6, 8]
/// let r = arange(2, 10, step: 2)
/// ```
///
/// - Parameters:
///     - start: start value
///     - stop: stop value
///     - step: step size (default: 1)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:_:step:dtype:stream:)-(Int,_,_,_,_)``
public func arange(
    _ start: Int, _ stop: Int, step: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(
        &result, Double(start), Double(stop), Double(step), DType.int32.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[0, stop)` with a given ``DType``.
///
/// Example:
///
/// ```swift
/// // Create [0.0, 1.0, 2.0, ...] as float32
/// let r = arange(10, dtype: .float32)
/// ```
///
/// - Parameters:
///     - stop: stop value
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:_:step:dtype:stream:)-(Int,_,_,_,_)``
public func arange(_ stop: Int, dtype: DType, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(&result, 0, Double(stop), 1, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[start, stop)` spaced by `step` with a given ``DType``.
///
/// Example:
///
/// ```swift
/// // Create [2.0, 4.0, 6.0, 8.0] as float32
/// let r = arange(2, 10, step: 2, dtype: .float32)
/// ```
///
/// - Parameters:
///     - start: start value
///     - stop: stop value
///     - step: step size (default: 1)
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:_:step:stream:)``
public func arange(
    _ start: Int, _ stop: Int, step: Int = 1, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(&result, Double(start), Double(stop), Double(step), dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[0, stop)` (floating point version).
///
/// Example:
///
/// ```swift
/// // Create [0.0, 1.0, 2.0, 3.0, 4.0]
/// let r = arange(5.0)
/// ```
///
/// - Parameters:
///     - stop: stop value
///     - dtype: data type of the output array (default: .float32)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:_:step:dtype:stream:)-(Double,_,_,_,_)``
public func arange(
    _ stop: Double, dtype: DType = .float32, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(&result, 0, stop, 1, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Generate values in the half-open interval `[start, stop)` spaced by `step` (floating point version).
///
/// Example:
///
/// ```swift
/// // Create [0.0, 0.5, 1.0, 1.5, 2.0, ...]
/// let r = arange(0.0, 5.0, step: 0.5)
/// ```
///
/// - Parameters:
///     - start: start value
///     - stop: stop value
///     - step: step size (default: 1.0)
///     - dtype: data type of the output array (default: .float32)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``arange(_:dtype:stream:)-(Double,_,_)``
public func arange(
    _ start: Double, _ stop: Double, step: Double = 1.0, dtype: DType = .float32,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_arange(&result, start, stop, step, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// Repeat an array along a specified axis.
///
/// > Deprected in favor of the more consistently named ``repeated(_:count:axis:stream:)``
///
/// ### See Also
/// - <doc:initialization>
/// - ``repeated(_:count:stream:)``
/// - ``full(_:values:stream:)``
@available(*, deprecated, renamed: "repeated(_:count:axis:stream:)")
public func `repeat`(_ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_repeat_axis(&result, array.ctx, count.int32, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Repeat a flattened array along axis 0.
///
/// > Deprected in favor of the more consistently named ``repeated(_:count:stream:)``
///
/// ### See Also
/// - <doc:initialization>
/// - ``repeated(_:count:axis:stream:)``
/// - ``full(_:values:stream:)``
@available(*, deprecated, renamed: "repeated(_:count:stream:)")
public func `repeat`(_ array: MLXArray, count: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_repeat(&result, array.ctx, count.int32, stream.ctx)
    return MLXArray(result)
}

/// Repeat an array along a specified axis.
///
/// Example:
///
/// ```swift
/// // repeat a [2, 2] array 4 times along axis 1
/// let r = MLXArray.repeated(MLXArray(0 ..< 4, [2, 2]), count: 4, axis: 1)
/// ```
///
/// - Parameters:
///     - array: array to repeat
///     - count: number of times to repeat
///     - axis: axis to repeat along
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``repeated(_:count:stream:)``
/// - ``full(_:values:stream:)``
public func repeated(_ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_repeat_axis(&result, array.ctx, count.int32, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Repeat a flattened array along axis 0.
///
/// Example:
///
/// ```swift
/// // repeat a 4 element array 4 times along axis 0
/// let r = MLXArray.repeated(MLXArray(0 ..< 4, [2, 2]), count: 4)
/// ```
///
/// - Parameters:
///     - array: array to repeat
///     - count: number of times to repeat
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
/// - ``repeated(_:count:axis:stream:)``
/// - ``full(_:values:stream:)``
public func repeated(_ array: MLXArray, count: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_repeat(&result, array.ctx, count.int32, stream.ctx)
    return MLXArray(result)
}

/// An array with ones at and below the given diagonal and zeros elsewhere.
///
/// Example:
///
/// ```swift
/// // [5, 5] array with the lower triangle filled with 1s
/// let r = MLXArray.triangle(5)
/// ```
///
/// - Parameters:
///     - n: number of rows in the output
///     - m: number of columns in the output -- equal to `n` if not specified
///     - k: index of the diagonal
///     - type: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
public func tri(
    _ n: Int, m: Int? = nil, k: Int = 0, type: (some HasDType).Type = Float.self,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_tri(&result, n.int32, (m ?? n).int32, k.int32, type.dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}

/// An array with ones at and below the given diagonal and zeros elsewhere and a given ``DType``.
///
/// Example:
///
/// ```swift
/// // [5, 5] array with the lower triangle filled with 1s
/// let r = MLXArray.triangle(5, dtype: .int32)
/// ```
///
/// - Parameters:
///     - n: number of rows in the output
///     - m: number of columns in the output -- equal to `n` if not specified
///     - k: index of the diagonal
///     - dtype: data type of the output array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:initialization>
public func tri(
    _ n: Int, m: Int? = nil, k: Int = 0, dtype: DType, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_tri(&result, n.int32, (m ?? n).int32, k.int32, dtype.cmlxDtype, stream.ctx)
    return MLXArray(result)
}
