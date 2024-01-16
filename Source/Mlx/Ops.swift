import Foundation
import Cmlx

// MARK: - Internal Ops

/// Broadcast a vector of arrays against one another.
func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let result = mlx_broadcast_arrays(arrays.map { $0.ctx }, arrays.count, stream.ctx)
    defer { mlx_vector_array_free(result) }
    
    return (0 ..< result.size).map { MLXArray(result.arrays[$0]!) }
}

// MARK: - Public Ops

/// An `and` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray(0 ..< 12, [3, 4])
///
/// // will produce a scalar MLXArray with false -- not all of the values are non-zero
/// let all = array.all()
///
/// // produces an MLXArray([false, true, true, true]) -- the first row has a zero
/// let allRows = array.all(axes: [0])
///
/// // equivalent
/// let allRows2 = array.all(axis: 0)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``all(axis:keepDims:stream:)``
/// - ``all(keepDims:stream:)``
/// - ``allTrue(stream:)``
public func all(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_all_axes(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// An `and` reduction over the given axes.
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``all(axes:keepDims:stream:)``
/// - ``all(keepDims:stream:)``
/// - ``allTrue(stream:)``
public func all(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_all_axis(array.ctx,  axis.int32, keepDims, stream.ctx))
}

/// An `and` reduction over the given axes.
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``all(axes:keepDims:stream:)``
/// - ``all(axis:keepDims:stream:)``
/// - ``allTrue(stream:)``
public func all(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_all_all(array.ctx,  keepDims, stream.ctx))
}

/// Approximate comparison of two arrays.
///
/// The arrays are considered equal if:
///
/// ```swift
/// all(abs(a - b) <= (atol + rtol * abs(b)))
/// ```
///
/// Note unlike ``arrayEqual(_:equalNAN:stream:)``, this function supports <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 4).sqrt()
/// let b: MLXArray(0 ..< 4) ** 0.5
///
/// if a.allClose(b).allTrue() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - array: input array
///     - other: array to compare to
///     - rtol: relative tolerance (see discussion)
///     - atol: absolute tolerance (see discussion)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``arrayEqual(_:equalNAN:stream:)``
public func allClose(_ array: MLXArray, _ other: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_allclose(array.ctx, other.ctx, rtol, atol, stream.ctx))
}

/// Return `true` if all contents are `true` (in the mlx-sense where true is != 0).
///
/// Equivalent to:
///
/// ```swift
/// let allTrue = array.all().item(Bool.self)
/// ```
///
/// Use this as:
///
/// ```swift
/// if (a < b || a.allClose(c)).allTrue() {
///     ...
/// }
/// ```
///
/// Note: this is equivalent to using an array in a boolean context in python.
///
/// - Parameters:
///     - array: input array
///
/// ### See Also
/// - <doc:reduction>
/// - ``all(axes:keepDims:stream:)``
/// - ``any(axes:keepDims:stream:)``
public func allTrue(_ array: MLXArray, stream: StreamOrDevice = .default) -> Bool {
    if array.ndim > 1 || array.dtype != .bool {
        let all = mlx_all_all(array.ctx,  false, stream.ctx)!
        let bool = mlx_array_item_bool(all)
        mlx_free(all)
        return bool
    } else {
        let bool = mlx_array_item_bool(array.ctx)
        return bool
    }
}

/// An `or` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray(0 ..< 12, [3, 4])
///
/// // will produce a scalar MLXArray with true -- some of the values are non-zero
/// let all = array.any()
///
/// // produces an MLXArray([true, true, true, true]) -- all rows have non-zeros
/// let allRows = array.any(axes: [0])
///
/// // equivalent
/// let allRows2 = array.any(axis: 0)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``any(axis:keepDims:stream:)``
/// - ``any(keepDims:stream:)``
/// - ``allTrue(stream:)``
public func any(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_any(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// An `or` reduction over the given axes.
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``any(axes:keepDims:stream:)``
/// - ``any(keepDims:stream:)``
/// - ``allTrue(stream:)``
public func any(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_any(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// An `or` reduction over the given axes.
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``any(axes:keepDims:stream:)``
/// - ``any(axis:keepDims:stream:)``
/// - ``allTrue(stream:)``
public func any(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_any_all(array.ctx,  keepDims, stream.ctx))
}

/// Indices of the maximum values along the axis.
///
/// ```swift
/// let array = MLXArray(4 ..< 16, [4, 3])
///
/// // this will produce [3, 3, 3] -- the index in each column for the maximum value
/// let i = array.argMax(axis=0)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``argMax(keepDims:stream:)``
/// - ``argMin(axis:keepDims:stream:)``
public func argMax(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argmax(array.ctx,  axis.int32, keepDims, stream.ctx))
}

/// Indices of the maximum value over the entire array.
///
/// ```swift
/// let array = MLXArray(4 ..< 16, [4, 3])
///
/// // this will produce [11] -- the index in the flattened array of the largest value
/// let i = array.argMax()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``argMax(axis:keepDims:stream:)``
/// - ``argMin(axis:keepDims:stream:)``
public func argMax(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argmax_all(array.ctx,  keepDims, stream.ctx))
}

/// Indices of the minimum values along the axis.
///
/// ```swift
/// let array = MLXArray(4 ..< 16, [4, 3])
///
/// // this will produce [0, 0, 0] -- the index in each column for the minimum value
/// let i = array.argMin(axis=0)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``argMin(keepDims:stream:)``
/// - ``argMax(axis:keepDims:stream:)``
public func argMin(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argmin(array.ctx,  axis.int32, keepDims, stream.ctx))
}

/// Indices of the minimum value over the entire array.
///
/// ```swift
/// let array = MLXArray(4 ..< 16, [4, 3])
///
/// // this will produce [0] -- the index in the flattened array of the smallest value
/// let i = array.argMin()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``argMin(axis:keepDims:stream:)``
/// - ``argMax(axis:keepDims:stream:)``
public func argMin(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argmin_all(array.ctx,  keepDims, stream.ctx))
}

/// Array equality check.
///
/// Compare two arrays for equality. Returns `True` if and only if the arrays
/// have the same shape and their values are equal. The arrays need not have
/// the same type to be considered equal.
///
/// ```swift
/// let a1 = MLXArray([0, 1, 2, 3])
/// let a2 = MLXArray([0.0, 1.0, 2.0, 3.0])
///
/// print(a1.arrayEqual(a2))
/// // prints the scalar array `true`
/// ```
///
/// ### See Also
/// - ``allClose(_:rtol:atol:stream:)``
/// - ``item(_:)``
/// - ``allTrue(stream:)``
/// - ``MLXArray/==(lhs:rhs:)``
public func arrayEqual(_ array: MLXArray, _ other: MLXArray, equalNAN: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_array_equal(array.ctx,  other.ctx, equalNAN, stream.ctx))
}

/// Element-wise cosine.
///
/// ### See Also
/// - <doc:element-wise>
public func cos(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cos(array.ctx,  stream.ctx))
}

/// Return the cumulative maximum of the elements along the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
/// let result = array.cummax(axis: 0)
/// ```
///
/// ### See Also
/// - ``cummax(reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cummax(_ array: MLXArray, axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cummax(array.ctx,  axis.int32, reverse, inclusive, stream.ctx))
}

/// Return the cumulative maximum of the elements over the flattened array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 8, 8, 9]
/// let result = array.cummax()
/// ```
///
/// ### See Also
/// - ``cummax(axis:reverse:inclusive:stream:)``
/// - ``cummax(reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cummax(_ array: MLXArray, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    let flat = mlx_reshape(array.ctx,  [-1], 1, stream.ctx)!
    defer { mlx_free(flat) }
    return MLXArray(mlx_cummax(array.ctx,  0, reverse, inclusive, stream.ctx))
}

/// Return the cumulative minimum of the elements along the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
/// let result = array.cummin(axis: 0)
/// ```
///
/// ### See Also
/// - ``cummin(reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cummin(_ array: MLXArray, axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cummin(array.ctx,  axis.int32, reverse, inclusive, stream.ctx))
}

/// Return the cumulative minimum of the elements over the flattened array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 5, 4, 4]
/// let result = array.cummin()
/// ```
///
/// ### See Also
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cummin(_ array: MLXArray, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    let flat = mlx_reshape(array.ctx,  [-1], 1, stream.ctx)!
    defer { mlx_free(flat) }
    return MLXArray(mlx_cummin(array.ctx,  0, reverse, inclusive, stream.ctx))
}

/// Return the cumulative product of the elements along the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [[5, 8], [20, 72]] -- cumulative product along the columns
/// let result = array.cumprod(axis: 0)
/// ```
///
/// ### See Also
/// - ``cumprod(reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cumprod(_ array: MLXArray, axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cumprod(array.ctx,  axis.int32, reverse, inclusive, stream.ctx))
}

/// Return the cumulative product of the elements over the flattened array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 40, 160, 1440]
/// let result = array.cumprod()
/// ```
///
/// ### See Also
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
/// - ``cumsum(axis:reverse:inclusive:stream:)``
public func cumprod(_ array: MLXArray, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    let flat = mlx_reshape(array.ctx,  [-1], 1, stream.ctx)!
    defer { mlx_free(flat) }
    return MLXArray(mlx_cumprod(array.ctx,  0, reverse, inclusive, stream.ctx))
}

/// Return the cumulative sum of the elements along the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [[5, 8], [9, 17]] -- cumulative sum along the columns
/// let result = array.cumsum(axis: 0)
/// ```
///
/// ### See Also
/// - ``cumsum(reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
public func cumsum(_ array: MLXArray, axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cumsum(array.ctx,  axis.int32, reverse, inclusive, stream.ctx))
}

/// Return the cumulative sum of the elements over the flattened array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 13, 17, 26]
/// let result = array.cumsum()
/// ```
///
/// ### See Also
/// - ``cumsum(axis:reverse:inclusive:stream:)``
/// - ``cumprod(axis:reverse:inclusive:stream:)``
/// - ``cummin(axis:reverse:inclusive:stream:)``
/// - ``cummax(axis:reverse:inclusive:stream:)``
public func cumsum(_ array: MLXArray, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    let flat = mlx_reshape(array.ctx,  [-1], 1, stream.ctx)!
    defer { mlx_free(flat) }
    return MLXArray(mlx_cumsum(array.ctx,  0, reverse, inclusive, stream.ctx))
}

/// Element-wise exponential.
///
/// ### See Also
/// - <doc:element-wise>
public func exp(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_exp(array.ctx,  stream.ctx))
}

/// Flatten an array.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< (8 * 4 * 3), [8, 4, 3])
///
/// // f1 is shape [8 * 4 * 3] = [96]
/// let f1 = a.flatten()
///
/// // f2 is [8, 4 * 3] = [8, 12]
/// let f2 = a.flatten(start: 1)
/// ```
///
/// - Parameters:
///     - array: input array
///     - start: first dimension to flatten
///     - end: last dimension to flatten
///
/// ### See Also
/// - <doc:shapes>
public func flatten(_ array: MLXArray, start: Int = 0, end: Int = -1, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_flatten(array.ctx,  start.int32, end.int32, stream.ctx))
}

/// Element-wise floor.
///
/// ### See Also
/// - ``round(decimals:stream:)``
/// - ``floorDivide(_:stream:)``
/// - <doc:element-wise>
public func floor(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_floor(array.ctx,  stream.ctx))
}

/// Element-wise integer division..
///
/// Divide two arrays with <doc:broadcasting>.
///
/// If either array is a floating point type then it is equivalent to calling ``floor(stream:)`` after `/`.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// let r = a.floorDivide(b)
/// ```
///
/// ### See Also
/// - ``/%(lhs:rhs:)``
/// - ``floor(stream:)``
/// - <doc:element-wise>
public func floorDivide(_ array: MLXArray, _ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_floor_divide(array.ctx,  other.ctx, stream.ctx))
}

/// Element-wise natural logarithm.
///
/// ### See Also
/// - <doc:element-wise>
public func log(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_log(array.ctx,  stream.ctx))
}

/// Element-wise base-2 logarithm.
///
/// ### See Also
/// - ``log(stream:)``
/// - ``log10(stream:)``
/// - ``log1p(stream:)``
/// - <doc:element-wise>
public func log2(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_log2(array.ctx,  stream.ctx))
}

/// Element-wise base-10 logarithm.
///
/// ### See Also
/// - ``log(stream:)``
/// - ``log2(stream:)``
/// - ``log1p(stream:)``
/// - <doc:element-wise>
public func log10(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_log10(array.ctx,  stream.ctx))
}

/// Element-wise natural log of one plus the array.
///
/// ### See Also
/// - ``log(stream:)``
/// - ``log2(stream:)``
/// - ``log10(stream:)``
/// - <doc:element-wise>
public func log1p(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_log1p(array.ctx,  stream.ctx))
}

/// A `log-sum-exp` reduction over the given axes.
///
/// The log-sum-exp reduction is a numerically stable version of:
///
/// ```swift
/// log(sum(exp(a), [axes]]))
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``logSumExp(axis:keepDims:stream:)``
/// - ``logSumExp(keepDims:stream:)``
public func logSumExp(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_logsumexp(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// A `log-sum-exp` reduction over the given axis.
///
/// The log-sum-exp reduction is a numerically stable version of:
///
/// ```swift
/// log(sum(exp(a), axis))
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``logSumExp(axes:keepDims:stream:)``
/// - ``logSumExp(keepDims:stream:)``
public func logSumExp(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_logsumexp(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// A `log-sum-exp` reduction over the entire array.
///
/// The log-sum-exp reduction is a numerically stable version of:
///
/// ```swift
/// log(sum(exp(a)))
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``logSumExp(axes:keepDims:stream:)``
/// - ``logSumExp(axis:keepDims:stream:)``
public func logSumExp(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_logsumexp_all(array.ctx,  keepDims, stream.ctx))
}

/// Matrix multiplication.
///
/// Perform the (possibly batched) matrix multiplication of two arrays. This function supports
/// broadcasting for arrays with more than two dimensions.
///
/// - If the first array is 1-D then a 1 is prepended to its shape to make it
///   a matrix. Similarly if the second array is 1-D then a 1 is appended to its
///   shape to make it a matrix. In either case the singleton dimension is removed
///   from the result.
/// - A batched matrix multiplication is performed if the arrays have more than
///   2 dimensions.  The matrix dimensions for the matrix product are the last
///   two dimensions of each input.
/// - All but the last two dimensions of each input are broadcast with one another using
///   standard <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray([1, 2, 3, 4], [2, 2])
/// let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])
///
/// // produces a [2, 3] result
/// let r = a.matmul(b)
/// ```
///
/// ### See Also
/// - ``***(lhs:rhs:)``
public func matmul(_ array: MLXArray, _ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_matmul(array.ctx,  other.ctx, stream.ctx))
}

/// A `max` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 9]
/// let result = array.max(axis=[0])
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``max(axis:keepDims:stream:)``
/// - ``max(keepDims:stream:)``
public func max(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_max(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// A `max` reduction over the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [8, 9]
/// let result = array.max(axis=1)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``max(axes:keepDims:stream:)``
/// - ``max(keepDims:stream:)``
public func max(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_max(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// A `max` reduction over the entire array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [9]
/// let result = array.max()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``max(axes:keepDims:stream:)``
/// - ``max(axis:keepDims:stream:)``
public func max(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_max_all(array.ctx,  keepDims, stream.ctx))
}

/// A `mean` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [4.5, 8.5]
/// let result = array.mean(axis=[0])
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``mean(axis:keepDims:stream:)``
/// - ``mean(keepDims:stream:)``
public func mean(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_mean(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// A `mean` reduction over the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [6.5, 6.5]
/// let result = array.mean(axis=1)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``mean(axes:keepDims:stream:)``
/// - ``mean(keepDims:stream:)``
public func mean(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_mean(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// A `mean` reduction over the entire array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [6.5]
/// let result = array.mean()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``mean(axes:keepDims:stream:)``
/// - ``mean(axis:keepDims:stream:)``
public func mean(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_mean_all(array.ctx,  keepDims, stream.ctx))
}

/// A `min` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [4, 8]
/// let result = array.min(axis=[0])
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``min(axis:keepDims:stream:)``
/// - ``min(keepDims:stream:)``
public func min(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_min(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// A `min` reduction over the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5, 4]
/// let result = array.min(axis=1)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``min(axes:keepDims:stream:)``
/// - ``min(keepDims:stream:)``
public func min(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_min(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// A `min` reduction over the entire array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [5]
/// let result = array.min()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``min(axes:keepDims:stream:)``
/// - ``min(axis:keepDims:stream:)``
public func min(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_min_all(array.ctx,  keepDims, stream.ctx))
}

/// Move an axis to a new position.
///
/// ```swift
/// let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
/// print(array)
/// // array([[[[0, 1],
/// //          [2, 3]],
/// //         [[4, 5],
/// //          [6, 7]]],
/// //        [[[8, 9],
/// //          [10, 11]],
/// //         [[12, 13],
/// //          [14, 15]]]], dtype=int64)
///
/// let r = array.moveAxis(source: 0, destination: 3)
/// print(r)
/// // array([[[[0, 8],
/// //          [1, 9]],
/// //         [[2, 10],
/// //          [3, 11]]],
/// //        [[[4, 12],
/// //          [5, 13]],
/// //         [[6, 14],
/// //          [7, 15]]]], dtype=int64)
/// ```
///
/// ### See Also
/// - <doc:shapes>
/// - ``swapAxes(_:_:stream:)``
public func moveAxis(_ array: MLXArray, source: Int, destination: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_moveaxis(array.ctx,  source.int32, destination.int32, stream.ctx))
}

/// Element-wise power operation.
///
/// Raise the elements of `self` to the powers in elements of `other` with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// // same as a ** b
/// let r = a.pow(b)
/// ```
///
/// ### See Also
/// - <doc:element-wise>
/// - ``**(lhs:rhs:)``
public func pow(_ array: MLXArray, _ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_power(array.ctx,  other.ctx, stream.ctx))
}

/// A `product` reduction over the given axes.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [20, 72]
/// let result = array.product(axis=[0])
/// ```
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``product(axis:keepDims:stream:)``
/// - ``product(keepDims:stream:)``
public func product(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_prod(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// A `product` reduction over the given axis.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [40, 36]
/// let result = array.product(axis=1)
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``product(axes:keepDims:stream:)``
/// - ``product(keepDims:stream:)``
public func product(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_prod(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// A `product` reduction over the entire array.
///
/// ```swift
/// let array = MLXArray([5, 8, 4, 9], [2, 2])
///
/// // result is [1440]
/// let result = array.product()
/// ```
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``product(axes:keepDims:stream:)``
/// - ``product(axis:keepDims:stream:)``
public func product(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_prod_all(array.ctx,  keepDims, stream.ctx))
}

/// Element-wise reciprocal.
///
/// ### See Also
/// - <doc:element-wise>
public func reciprocal(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_reciprocal(array.ctx,  stream.ctx))
}

/// Reshape an array while preserving the size.
///
/// ### See Also
/// - <doc:shapes>
public func reshape(_ array: MLXArray, _ newShape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_reshape(array.ctx,  newShape.asInt32, newShape.count, stream.ctx))
}

/// Round to the given number of decimals.
///
/// Roughly equivalent to:
///
/// ```swift
/// let array: MLXArray
///
/// let s = 10 ** decimals
/// let result = round(array * s) / s
/// ```
///
/// ### See Also
/// - <doc:element-wise>
/// - ``floor(stream:)``
public func round(_ array: MLXArray, decimals: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_round(array.ctx,  decimals.int32, stream.ctx))
}

/// Element-wise reciprocal and square root.
///
/// ### See Also
/// - <doc:element-wise>
public func rsqrt(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_rsqrt(array.ctx,  stream.ctx))
}

/// Element-wise sine.
///
/// ### See Also
/// - <doc:element-wise>
public func sin(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sin(array.ctx,  stream.ctx))
}

/// Split an array into equal size pieces along a given axis.
///
/// Splits the array into equal size pieces along a given axis and returns an array of `MLXArray`:
///
/// ```swift
/// let array = MLXArray(0 ..< 12, (4, 3))
///
/// let halves = array.split(2)
/// print(halves)
///
/// [array([[0, 1, 2],
///         [3, 4, 5]], dtype=int64),
///  array([[6, 7, 8],
///         [9, 10, 11]], dtype=int64)]
/// ```
///
/// - Parameters:
///     - array: input array
///     - parts: array is split into that many sections of equal size. It is a fatal error if this is not possible
///     - axis: axis to split along
///
/// ### See Also
/// - <doc:shapes>
/// - ``split(indices:axis:stream:)``
public func split(_ array: MLXArray, parts: Int, axis: Int = 0, stream: StreamOrDevice = .default) -> [MLXArray] {
    let vec = mlx_split_equal_parts(array.ctx,  parts.int32, axis.int32, stream.ctx)
    defer { mlx_vector_array_free(vec) }
    return MLXArray.fromVector(vec)
}

/// Split an array along a given axis.
///
/// - Parameters:
///     - array: input array
///     - indices: the indices of the start of each subarray along the given axis
///     - axis: axis to split along
///
/// ### See Also
/// - <doc:shapes>
/// - ``split(parts:axis:stream:)``
public func split(_ array: MLXArray, indices: [Int], axis: Int = 0, stream: StreamOrDevice = .default) -> [MLXArray] {
    let vec = mlx_split(array.ctx, indices.asInt32, indices.count, axis.int32, stream.ctx)
    defer { mlx_vector_array_free(vec) }
    return MLXArray.fromVector(vec)
}

/// Element-wise square root
///
/// ### See Also
/// - <doc:element-wise>
public func sqrt(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sqrt(array.ctx,  stream.ctx))
}

/// Element-wise square.
///
/// ### See Also
/// - <doc:element-wise>
public func square(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_square(array.ctx,  stream.ctx))
}

/// Remove length one axes from an array.
///
/// - Parameters:
///     - array: input array
///     - axes: axes to remove
///
/// ### See Also
/// - <doc:shapes>
/// - ``squeeze(axis:stream:)``
/// - ``squeeze(stream:)``
public func squeeze(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_squeeze(array.ctx,  axes.asInt32, axes.count, stream.ctx))
}

/// Remove length one axes from an array.
///
/// - Parameters:
///     - array: input array
///     - axis: axis to remove
///
/// ### See Also
/// - <doc:shapes>
/// - ``squeeze(axes:stream:)``
/// - ``squeeze(stream:)``
public func squeeze(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_squeeze(array.ctx,  [axis.int32], 1, stream.ctx))
}

/// Remove all length one axes from an array.
///
/// ### See Also
/// - <doc:shapes>
/// - ``squeeze(axes:stream:)``
/// - ``squeeze(axis:stream:)``
public func squeeze(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_squeeze_all(array.ctx,  stream.ctx))
}

/// Sum reduce the array over the given axes.
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true` keep the reduces axes as singleton dimensions
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``sum(axis:keepDims:stream:)``
/// - ``sum(keepDims:stream:)``
public func sum(_ array: MLXArray, axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sum(array.ctx,  axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// Sum reduce the array over the given axis.
///
/// - Parameters:
///     - array: input array
///     - axis: axis to reduce over
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``sum(axes:keepDims:stream:)``
/// - ``sum(keepDims:stream:)``
public func sum(_ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sum(array.ctx,  [axis.int32], 1, keepDims, stream.ctx))
}

/// Sum reduce the array over all axes.
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true`keep reduced axis as singleton dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``sum(axes:keepDims:stream:)``
/// - ``sum(axis:keepDims:stream:)``
public func sum(_ array: MLXArray, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sum_all(array.ctx,  keepDims, stream.ctx))
}

/// Swap two axes of an array.
///
/// ```swift
/// let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
/// print(array)
/// // array([[[[0, 1],
/// //          [2, 3]],
/// //         [[4, 5],
/// //          [6, 7]]],
/// //        [[[8, 9],
/// //          [10, 11]],
/// //         [[12, 13],
/// //          [14, 15]]]], dtype=int64)
///
/// let r = array.swapAxes(2, 1)
/// print(r)
/// // array([[[[0, 8],
/// //          [2, 10]],
/// //         [[4, 12],
/// //          [6, 14]]],
/// //        [[[1, 9],
/// //          [3, 11]],
/// //         [[5, 13],
/// //          [7, 15]]]], dtype=int64)
/// ```
///
/// ### See Also
/// - <doc:shapes>
public func swapAxes(_ array: MLXArray, _ axis1: Int, _ axis2: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_swapaxes(array.ctx,  axis1.int32, axis2.int32, stream.ctx))
}

/// Take elements along an axis.
///
/// The elements are taken from `indices` along the specified axis.
///
/// ```swift
/// let array = MLXArray(0 ..< 12, [3, 4])
///
/// /// produces a [3, 2] result with the index 0 and 2 columns from the array
/// let col = array.take([0, 2], axis: 1)
/// ```
///
/// ### See Also
/// - ``take(_:stream:)``
public func take(_ array: MLXArray, _ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_take(array.ctx, indices.ctx, axis.int32, stream.ctx))
}

/// Take elements from flattened 1-D array.
///
/// ### See Also
/// - ``take(_:axis:stream:)``
public func take(_ array: MLXArray, _ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_take_all(array.ctx, indices.ctx, stream.ctx))
}

/// Transpose the dimensions of the array.
///
/// - Parameters:
///     - array: input array
///     - axes: Specifies the source axis for each axis in the new array
///
/// ### See Also
/// - <doc:shapes>
/// - ``transpose(axis:stream:)``
/// - ``transpose(stream:)``
public func transpose(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_transpose(array.ctx,  axes.asInt32, axes.count, stream.ctx))
}

/// Transpose the dimensions of the array.
///
/// This swaps the position of the first dimension with the given axis.
///
/// ### See Also
/// - <doc:shapes>
/// - ``transpose(axes:stream:)``
/// - ``transpose(stream:)``
public func transpose(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_transpose(array.ctx,  [axis.int32], 1, stream.ctx))
}

/// Transpose the dimensions of the array.
///
/// With no axes specified this will reverse the axes in the array.
///
/// ### See Also
/// - <doc:shapes>
/// - ``transpose(axes:stream:)``
/// - ``transpose(axis:stream:)``
public func transpose(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_transpose_all(array.ctx,  stream.ctx))
}

/// Transpose the dimensions of the array.
///
/// Cover for ``transpose(stream:)``
public func T(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    transpose(array, stream: stream)
}

/// Compute the variance(s) over the given axes
///
/// - Parameters:
///     - array: input array
///     - axes: axes to reduce over
///     - keepDims: if `true` keep the reduces axes as singleton dimensions
///     - ddof: the divisor to compute the variance is `N - ddof`
///
/// ### See Also
/// - <doc:reduction>
/// - ``variance(axis:keepDims:ddof:stream:)``
/// - ``variance(keepDims:ddof:stream:)``
public func variance(_ array: MLXArray, axes: [Int], keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_var(array.ctx,  axes.asInt32, axes.count, keepDims, ddof.int32, stream.ctx))
}

/// Compute the variance(s) over the given axes
///
/// - Parameters:
///     - array: input array
///     - axis: axes to reduce over
///     - keepDims: if `true` keep the reduces axis as singleton dimensions
///     - ddof: the divisor to compute the variance is `N - ddof`
///
/// ### See Also
/// - <doc:reduction>
/// - ``variance(axes:keepDims:ddof:stream:)``
/// - ``variance(keepDims:ddof:stream:)``
public func variance(_ array: MLXArray, axis: Int, keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_var(array.ctx,  [axis.int32], 1, keepDims, ddof.int32, stream.ctx))
}

/// Compute the variance(s) over the given axes
///
/// - Parameters:
///     - array: input array
///     - keepDims: if `true` keep the reduces axes as singleton dimensions
///     - ddof: the divisor to compute the variance is `N - ddof`
///
///
/// ### See Also
/// - <doc:reduction>
/// - ``variance(axes:keepDims:ddof:stream:)``
/// - ``variance(axis:keepDims:ddof:stream:)``
public func variance(_ array: MLXArray, keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_var_all(array.ctx,  keepDims, ddof.int32, stream.ctx))
}
