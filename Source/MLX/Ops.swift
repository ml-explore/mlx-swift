// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// MARK: - Internal Ops

/// Broadcast a vector of arrays against one another.
func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(vector_array) }

    var result = mlx_vector_array_new()
    mlx_broadcast_arrays(&result, vector_array, stream.ctx)
    defer { mlx_vector_array_free(result) }

    return mlx_vector_array_values(result)
}

/// Element-wise addition.
///
/// Add two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// // equivalent to a + b + 7
/// let r = add(add(a, b), 7)
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``MLXArray/+(_:_:)-1rv98``
public func add<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_add(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

@available(*, deprecated, renamed: "addMM(_:_:_:alpha:beta:stream:)")
@_documentation(visibility: internal)
public func addmm<A: ScalarOrArray, B: ScalarOrArray, C: ScalarOrArray>(
    _ c: C, _ a: A, _ b: B, alpha: Float = 1.0, beta: Float = 1.0, stream: StreamOrDevice = .default
) -> MLXArray {
    addMM(c, a, b, alpha: alpha, beta: beta, stream: stream)
}

/// Matrix multiplication with addition and optional scaling.
///
/// Perform the (possibly batched) matrix multiplication of two arrays and add to the result
/// with optional scaling factors.
///
/// Equivalent to:
///
/// ```swift
/// alpha * matmul(a, b) + beta * c
/// ```
///
/// > Note the ordering of the parameters
///
/// - Parameters:
///   - c: input array or scalar
///   - a: input array or scalar
///   - b: input array or scalar
///   - alpha: optional scaling for the matrix product of `a` and `b`
///   - beta: optional scaling factor for `c`
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``matmul(_:_:stream:)``
/// - ``blockMaskedMM(_:_:blockSize:maskOut:maskLHS:maskRHS:stream:)``
public func addMM<A: ScalarOrArray, B: ScalarOrArray, C: ScalarOrArray>(
    _ c: C, _ a: A, _ b: B, alpha: Float = 1.0, beta: Float = 1.0, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    let (_, c) = toArrays(a, c)
    var result = mlx_array_new()
    mlx_addmm(&result, c.ctx, a.ctx, b.ctx, alpha, beta, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse cosine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``cos(_:stream:)``
public func acos(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arccos(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse hyperbolic cosine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``cosh(_:stream:)``
public func acosh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arccosh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse sine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``sin(_:stream:)``
public func asin(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arcsin(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse hyperbolic sine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``sinh(_:stream:)``
public func asinh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arcsinh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse tangent.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``tan(_:stream:)``
public func atan(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arctan(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse tangent of the ratio of two arrays.
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``atan(_:stream:)``
public func atan2(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arctan2(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse hyperbolic tangent.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``tanh(_:stream:)``
public func atanh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_arctanh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Convert array to have at least 1 dimension.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast1D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_atleast_1d(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Convert array to have at least 2 dimensions.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast2D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_atleast_2d(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Convert array to have at least 3 dimensions.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast3D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_atleast_3d(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Returns the indices that partition the array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// For example:
///
/// ```swift
/// // array with values in random order
/// let array = MLXRandom.randInt(0 ..< 100, [10])
///
/// let partitionIndexes = argPartition(array, kth: 3)
///
/// // the partitioned array.  the pivot is at partitioned[3] and all values
/// // with lower indexes will be less than (in undefined order)
/// let partitioned = array[sortIndexes]
/// ```
///
/// - Parameters:
///     - array: input array
///     - kth: element index at the `kth` position in the output will give the sorted position.  All indices before the`kth` position will be of elements less than or equal to the element at the `kth` index and all indices after will be elemenents greater than or equal to the element at the `kth` position.
///     - axis: axis to partition over
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``argPartition(_:kth:stream:)``
/// - ``partitioned(_:kth:axis:stream:)``
public func argPartition(_ array: MLXArray, kth: Int, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_argpartition_axis(&result, array.ctx, kth.int32, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Returns the indices that partition the flattened array.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// - Parameters:
///     - array: input array
///     - kth: element index at the `kth` position in the output will give the sorted position.  All indices before the`kth` position will be of elements less than or equal to the element at the `kth` index and all indices after will be elemenents greater than or equal to the element at the `kth` position.
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``argPartition(_:kth:axis:stream:)``
/// - ``partitioned(_:kth:axis:stream:)``
public func argPartition(_ array: MLXArray, kth: Int, stream: StreamOrDevice = .default) -> MLXArray
{
    var result = mlx_array_new()
    mlx_argpartition(&result, array.ctx, kth.int32, stream.ctx)
    return MLXArray(result)
}

/// Returns the indices that sort the array.
///
/// ```swift
/// // array with values in random order
/// let array = MLXRandom.randInt(0 ..< 100, [10])
///
/// let sortIndexes = argSort(array, axis: -1)
///
/// // the array in sorted order
/// let sorted = array[sortIndexes]
/// ```
///
/// - Parameters:
///     - array: input array
///     - axis: axis to sort over
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``argSort(_:stream:)``
public func argSort(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_argsort_axis(&result, array.ctx, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Returns the indices that sort the array.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``argSort(_:axis:stream:)``
public func argSort(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_argsort(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Create a view into the array with the given shape and strides.
///
/// The resulting array will always be as if the provided array was row
/// contiguous regardless of the provided arrays storage order and current strides.
///
/// > Caution! This function should be used with caution as it changes
/// the shape and strides of the array directly. This can lead to the
/// resulting array pointing to invalid memory locations which can
/// result into crashes.
///
/// Here are two examples of use:
///
/// ```swift
/// // strides in the reverse order is a transpose
/// let a = MLXArray(0 ..< 12, [4, 3])
///
/// let transposed = asStrided(a, [3, 4], strides: [1, 3])
/// ```
///
/// and:
///
/// ```swift
/// // negative strides and an offset produce a reversed array
/// let a = MLXArray(0 ..< 16, [4, 4])
///
/// let b = asStrided(a, [4, 4], strides: [-4, -1], offset: 15)
/// let same = MLXArray((0 ..< 16).reversed(), [4, 4])
/// ```
///
/// - Parameters:
///     - array: input array
///     - shape: shape of the resulting array.  If not specified it will keep the same shape
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
public func asStrided(
    _ array: MLXArray, _ shape: [Int]? = nil, strides: [Int]? = nil, offset: Int = 0,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let shape = shape ?? array.shape

    let resolvedStrides: [Int64]
    if let strides {
        resolvedStrides = strides.map { .init($0) }
    } else {
        var result = [Int64]()
        var cum = 1
        for v in shape.reversed() {
            result.append(Int64(cum))
            cum = cum * v
        }
        resolvedStrides = result.reversed()
    }

    var result = mlx_array_new()
    mlx_as_strided(
        &result,
        array.ctx, shape.asInt32, shape.count, resolvedStrides, resolvedStrides.count,
        offset,
        stream.ctx)
    return MLXArray(result)
}

/// Matrix multiplication with block masking.
///
/// Perform the (possibly batched) matrix multiplication of two arrays and with blocks
/// of size `blockSize x blockSize` optionally masked out.
///
/// Assuming `a` with shape (..., `M`, `K`) and b with shape (..., `K`, `N`)
///
/// * `maskLHS` must have shape (..., ceil(`M` / `blockSize`), ceil(`K` / `blockSize`))
///
/// * `maskRHS` must have shape (..., ceil(`K` / `blockSize`), ceil(`N` / `blockSize`))
///
/// * `maskOut` must have shape (..., ceil(`M` / `blockSize`), ceil(`N` / `blockSize`))
///
/// > Note: Only `block_size=64` and `block_size=32` are currently supported
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - blockSize: Size of blocks to be masked. Must be `32` or `64`
///   - maskOut: Boolean mask for output
///   - maskLHS: Boolean mask for a
///   - maskRHS: Boolean mask for b
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``multiply(_:_:stream:)``
/// - ``addMM(_:_:_:alpha:beta:stream:)``
/// - ``MLXArray/matmul(_:stream:)``
/// - ``matmul(_:_:stream:)``
public func blockMaskedMM(
    _ a: MLXArray, _ b: MLXArray, blockSize: Int = 64, maskOut: MLXArray? = nil,
    maskLHS: MLXArray? = nil, maskRHS: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()

    mlx_block_masked_mm(
        &result,
        a.ctx, b.ctx, blockSize.int32, (maskOut ?? .mlxNone).ctx, (maskLHS ?? .mlxNone).ctx,
        (maskRHS ?? .mlxNone).ctx, stream.ctx)

    return MLXArray(result)
}

/// Broadcast an array to the given shape.
///
/// - Parameters:
///     - array: input array
///     - shape: shape to broadcast to
///
/// ### See Also
/// - <doc:broadcasting>
public func broadcast(_ array: MLXArray, to shape: [Int], stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_broadcast_to(&result, array.ctx, shape.asInt32, shape.count, stream.ctx)
    return MLXArray(result)
}

/// Element-wise ceil.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``floor(_:stream:)``
public func ceil(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_ceil(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Clip the values of the array between the given minimum and maximum.
///
/// - Parameters:
///     - array: input array
///     - min: minimum value (must broadcast to `array`)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``clip(_:max:stream:)``
/// - ``clip(_:min:max:stream:)``
public func clip<A: ScalarOrArray>(
    _ array: MLXArray, min: A, stream: StreamOrDevice = .default
) -> MLXArray {
    let (array, min) = toArrays(array, min)
    var result = mlx_array_new()
    let max = mlx_array_new()
    defer { mlx_array_free(max) }
    mlx_clip(&result, array.ctx, min.ctx, max, stream.ctx)
    return MLXArray(result)
}

/// Clip the values of the array between the given minimum and maximum.
///
/// - Parameters:
///     - array: input array
///     - min: minimum value (must broadcast to `array`)
///     - max: maximum value (must broadcast to `array`).  If omitted only the `min` will be honored.
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``clip(_:max:stream:)``
public func clip<A: ScalarOrArray, B: ScalarOrArray>(
    _ array: MLXArray, min: A, max: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (array, min) = toArrays(array, min)
    let (_, max) = toArrays(array, max)
    var result = mlx_array_new()
    mlx_clip(&result, array.ctx, min.ctx, max.ctx, stream.ctx)
    return MLXArray(result)
}

/// Clip the values of the array up to the given maximum.
///
/// - Parameters:
///     - array: input array
///     - max: maximum value (must broadcast to `array`)
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``clip(_:min:stream:)``
/// - ``clip(_:min:max:stream:)``
public func clip<A: ScalarOrArray>(_ array: MLXArray, max: A, stream: StreamOrDevice = .default)
    -> MLXArray
{
    let (array, max) = toArrays(array, max)
    var result = mlx_array_new()
    let min = mlx_array_new()
    defer { mlx_array_free(min) }
    mlx_clip(&result, array.ctx, min, max.ctx, stream.ctx)
    return MLXArray(result)
}

/// Concatenate the arrays along the given axis.
///
/// - Parameters:
///     - array: input array
///     - axis: the axis along which to concatenate
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
public func concatenated(_ arrays: [MLXArray], axis: Int = 0, stream: StreamOrDevice = .default)
    -> MLXArray
{
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(vector_array) }

    var result = mlx_array_new()
    mlx_concatenate_axis(&result, vector_array, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// 1D convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// - Parameters:
///     - array: input array of shape `[N, H, C_in]`
///     - weight: weight array of shape `[C_out, H, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``conv3d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
public func conv1d(
    _ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0, dilation: Int = 1,
    groups: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv1d(
        &result,
        array.ctx, weight.ctx, stride.int32, padding.int32, dilation.int32, groups.int32,
        stream.ctx)
    return MLXArray(result)
}

/// 2D convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// ```swift
/// padding: 1
/// ```
///
/// This will produce a padding of `(1, 1)`.  You can also give an array:
///
/// ```swift
/// padding: [2, 3]
/// ```
///
/// See ``IntOrPair`` for more information.
///
/// - Parameters:
///     - array: input array of shape `[N, H, W, C_in]`
///     - weight: weight array of shape `[C_out, H, W, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrPair``
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``conv3d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-9t1sj``
public func conv2d(
    _ array: MLXArray, _ weight: MLXArray, stride: IntOrPair = 1, padding: IntOrPair = 0,
    dilation: IntOrPair = 1, groups: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv2d(
        &result,
        array.ctx, weight.ctx, stride.first.int32, stride.second.int32, padding.first.int32,
        padding.second.int32, dilation.first.int32, dilation.second.int32, groups.int32,
        stream.ctx)
    return MLXArray(result)
}

/// 3D convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// ```swift
/// padding: 1
/// ```
///
/// This will produce a padding of `(1, 1, 1)`.  You can also give an array:
///
/// ```swift
/// padding: [2, 3, 3]
/// ```
///
/// See ``IntOrTriple`` for more information.
///
/// - Parameters:
///     - array: input array of shape `[N, D, H, W, C_in]`
///     - weight: weight array of shape `[C_out, D, H, W, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrTriple``
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-9t1sj``
public func conv3d(
    _ array: MLXArray, _ weight: MLXArray, stride: IntOrTriple = 1, padding: IntOrTriple = 0,
    dilation: IntOrTriple = 1, groups: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv3d(
        &result,
        array.ctx, weight.ctx,
        stride.first.int32, stride.second.int32, stride.third.int32,
        padding.first.int32, padding.second.int32, padding.third.int32,
        dilation.first.int32, dilation.second.int32, dilation.third.int32,
        groups.int32, stream.ctx)
    return MLXArray(result)
}

/// General convolution over an input with several channels.
///
/// > Only 1d and 2d convolutions are supported at the moment
///
/// > the default `groups: 1` is currently supported
///
/// - Parameters:
///   - array: Input array of shape `(N, ..., C_in)`
///   - weight: Weight array of shape `(C_out, ..., C_in)`
///   - strides: `Int` or `[Int]` with kernel strides.  All dimensions get the
///   same stride if only one number is specified.
///   - padding: `Int` or `[Int]` with input padding.  All dimensions get the
///   same padding if only one number is specified.
///   - kernelDilation: `Int` or `[Int]` with kernel dilation.  All dimensions get the
///   same dilation if only one number is specified.
///   - inputDilation: `Int` or `[Int]` with input dilation.  All dimensions get the
///   same dilation if only one number is specified.
///   - groups: input feature groups
///   - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrArray``
/// - ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-6j1nr``
public func convGeneral(
    _ array: MLXArray, _ weight: MLXArray, strides: IntOrArray = 1, padding: IntOrArray = 0,
    kernelDilation: IntOrArray = 1, inputDilation: IntOrArray = 1, groups: Int = 1,
    flip: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv_general(
        &result,
        array.ctx, weight.ctx,
        strides.asInt32Array, strides.count,
        padding.asInt32Array, padding.count,
        padding.asInt32Array, padding.count,
        kernelDilation.asInt32Array, kernelDilation.count,
        inputDilation.asInt32Array, inputDilation.count,
        groups.int32, flip, stream.ctx)
    return MLXArray(result)
}

/// General convolution over an input with several channels with a padding pair.
///
/// > Only 1d and 2d convolutions are supported at the moment
///
/// > the default `groups: 1` is currently supported
///
/// - Parameters:
///   - array: Input array of shape `(N, ..., C_in)`
///   - weight: Weight array of shape `(C_out, ..., C_in)`
///   - strides: `Int` or `[Int]` with kernel strides.  All dimensions get the
///   same stride if only one number is specified.
///   - padding: pair of padding values to apply to all dimensions
///   - kernelDilation: `Int` or `[Int]` with kernel dilation.  All dimensions get the
///   same dilation if only one number is specified.
///   - inputDilation: `Int` or `[Int]` with input dilation.  All dimensions get the
///   same dilation if only one number is specified.
///   - groups: input feature groups
///   - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrArray``
/// - ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-6j1nr``
public func convGeneral(
    _ array: MLXArray, _ weight: MLXArray, strides: IntOrArray = 1, padding: (Int, Int),
    kernelDilation: IntOrArray = 1, inputDilation: IntOrArray = 1, groups: Int = 1,
    flip: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv_general(
        &result,
        array.ctx, weight.ctx,
        strides.asInt32Array, strides.count,
        [padding.0.int32], 1,
        [padding.1.int32], 1,
        kernelDilation.asInt32Array, kernelDilation.count,
        inputDilation.asInt32Array, inputDilation.count,
        groups.int32, flip, stream.ctx)
    return MLXArray(result)
}

/// 1D transposed convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// - Parameters:
///     - array: input array of shape `[N, H, C_in]`
///     - weight: weight array of shape `[C_out, H, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - outputPadding: output padding
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed2d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed3d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
public func convTransposed1d(
    _ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0,
    dilation: Int = 1, outputPadding: Int = 0, groups: Int = 1,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv_transpose1d(
        &result,
        array.ctx, weight.ctx, stride.int32, padding.int32,
        dilation.int32, outputPadding.int32, groups.int32,
        stream.ctx)
    return MLXArray(result)
}

/// 2D transposed convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// ```swift
/// padding: 1
/// ```
///
/// This will produce a padding of `(1, 1)`.  You can also give an array:
///
/// ```swift
/// padding: [2, 3]
/// ```
///
/// See ``IntOrPair`` for more information.
///
/// - Parameters:
///     - array: input array of shape `[N, H, W, C_in]`
///     - weight: weight array of shape `[C_out, H, W, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - outputPadding: output padding
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrPair``
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed3d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-9t1sj``
public func convTransposed2d(
    _ array: MLXArray, _ weight: MLXArray, stride: IntOrPair = 1, padding: IntOrPair = 0,
    dilation: IntOrPair = 1, outputPadding: IntOrPair = 0, groups: Int = 1,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv_transpose2d(
        &result,
        array.ctx, weight.ctx, stride.first.int32, stride.second.int32, padding.first.int32,
        padding.second.int32, dilation.first.int32, dilation.second.int32,
        outputPadding.first.int32, outputPadding.second.int32, groups.int32,
        stream.ctx)
    return MLXArray(result)
}

/// 3D transposed convolution over an input with several channels.
///
/// > Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// ```swift
/// padding: 1
/// ```
///
/// This will produce a padding of `(1, 1, 1)`.  You can also give an array:
///
/// ```swift
/// padding: [2, 3, 3]
/// ```
///
/// See ``IntOrTriple`` for more information.
///
/// - Parameters:
///     - array: input array of shape `[N, D, H, W, C_in]`
///     - weight: weight array of shape `[C_out, D, H, W, C_in]`
///     - stride: kernel stride
///     - padding: input padding
///     - dilation: kernel dilation
///     - outputPadding: output padding
///     - groups: input feature groups
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``IntOrTriple``
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convTransposed3d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``convolve(_:_:mode:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-9t1sj``
public func convTransposed3d(
    _ array: MLXArray, _ weight: MLXArray, stride: IntOrTriple = 1, padding: IntOrTriple = 0,
    dilation: IntOrTriple = 1, outputPadding: IntOrTriple = 0, groups: Int = 1,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_conv_transpose3d(
        &result,
        array.ctx, weight.ctx,
        stride.first.int32, stride.second.int32, stride.third.int32,
        padding.first.int32, padding.second.int32, padding.third.int32,
        dilation.first.int32, dilation.second.int32, dilation.third.int32,
        outputPadding.first.int32, outputPadding.second.int32, outputPadding.third.int32,
        groups.int32, stream.ctx)
    return MLXArray(result)
}

/// Mode for ``convolve(_:_:mode:stream:)``
public enum ConvolveMode: Sendable {
    case full
    case valid
    case same
}

/// The discrete convolution of 1D arrays.
///
/// - Parameters:
///     - a: 1D input array
///     - b: 1D input array
///     - mode: padding mode
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:convolution>
/// - ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
/// - ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
public func convolve<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, mode: ConvolveMode = .full, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)

    precondition(a.ndim == 1, "inputs must be 1d (a)")
    precondition(b.ndim == 1, "inputs must be 1d (b)")

    var (input, weight) = a.size < b.size ? (b, a) : (a, b)

    var slice = mlx_array_new()
    mlx_slice(
        &slice,
        weight.ctx, [weight.dim(0) - 1].asInt32, 1, [-weight.dim(0) - 1].asInt32, 1, [-1], 1,
        stream.ctx)
    weight = MLXArray(slice)

    weight = weight.reshaped([1, -1, 1], stream: stream)
    input = input.reshaped([1, -1, 1], stream: stream)

    let weightSize = weight.size
    var padding = 0

    switch mode {
    case .full:
        padding = weightSize - 1
    case .valid:
        padding = 0
    case .same:
        if weightSize % 2 == 1 {
            padding = weightSize / 2
        } else {
            let padLeft = weightSize / 2
            let padRight = max(0, padLeft / 2 - 1)

            input = padded(input, widths: [0, [padLeft, padRight], 0], stream: stream)
        }
    }

    var result = mlx_array_new()
    mlx_conv1d(&result, input.ctx, weight.ctx, 1, padding.int32, 1, 1, stream.ctx)
    return MLXArray(result).reshaped(-1, stream: stream)
}

/// Element-wise hyperbolic cosine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``cos(_:stream:)``
public func cosh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_cosh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Convert angles from radians to degrees.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``radians(_:stream:)``
public func degrees(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_degrees(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Dequantize the matrix `w` using the provided `scales` and
/// `biases` and the `group_size` and `bits` configuration.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
///
/// ### See Also
/// - ``quantized(_:groupSize:bits:stream:)``
/// - ``quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)``
public func dequantized(
    _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int = 64, bits: Int = 4,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_dequantize(&result, w.ctx, scales.ctx, biases.ctx, groupSize.int32, bits.int32, stream.ctx)
    return MLXArray(result)
}

/// Element-wise division.
///
/// Divide two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// let r = a / b / 7
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``divmod(_:_:stream:)``
public func divide<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_divide(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise quotient and remainder.
///
/// The fuction `divmod(a, b)` is equivalent to but faster than
/// `(a // b, a % b)`. The function uses numpy-style broadcasting
/// semantics. Either or both input arrays can also be scalars.
///
/// - Parameters:
///   - a: input array or scalar
///   - b: input array or scalar
///   - stream: stream or device to evaluate on
/// - Returns: The quotient `a / b` and remainder `a % b`
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``divide(_:_:stream:)``
/// - ``remainder(_:_:stream:)``
public func divmod<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> (MLXArray, MLXArray) {
    let (a, b) = toArrays(a, b)
    var vec = mlx_vector_array_new()
    mlx_divmod(&vec, a.ctx, b.ctx, stream.ctx)
    defer { mlx_vector_array_free(vec) }
    let result = mlx_vector_array_values(vec)
    return (result[0], result[1])
}

/// Perform the Einstein summation convention on the operands.
///
/// - Parameters:
///   - subscripts: Einstein summation convention equation
///   - operands: input arrays
///   - stream: stream or device to evaluate on
public func einsum(_ subscripts: String, _ operands: MLXArray..., stream: StreamOrDevice = .default)
    -> MLXArray
{
    einsum(subscripts, operands: operands, stream: stream)
}

/// Perform the Einstein summation convention on the operands.
///
/// - Parameters:
///   - subscripts: Einstein summation convention equation
///   - operands: input arrays
///   - stream: stream or device to evaluate on
public func einsum(_ subscripts: String, operands: [MLXArray], stream: StreamOrDevice = .default)
    -> MLXArray
{
    let operands = new_mlx_vector_array(operands)
    defer { mlx_vector_array_free(operands) }

    var result = mlx_array_new()
    mlx_einsum(&result, subscripts, operands, stream.ctx)
    return MLXArray(result)
}

/// Element-wise equality.
///
/// Equality comparison on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// if (a .== b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func equal<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_equal(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise error function.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html)
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``erfInverse(_:stream:)``
public func erf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_erf(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise inverse of ``erf(_:stream:)``.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html)
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``erf(_:stream:)``
public func erfInverse(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_erfinv(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Add a size one dimension at the given axis.
///
/// - Parameters:
///     - array: input array
///     - axes: indexes of the inserted dimensions
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``expandedDimensions(_:axis:stream:)``
public func expandedDimensions(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_expand_dims_axes(&result, array.ctx, axes.asInt32, axes.count, stream.ctx)
    return MLXArray(result)
}

/// Add a size one dimension at the given axis.
///
/// - Parameters:
///     - array: input array
///     - axis: index of the inserted dimension
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``expandedDimensions(_:axes:stream:)``
public func expandedDimensions(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_expand_dims_axes(&result, array.ctx, [axis.int32], 1, stream.ctx)
    return MLXArray(result)
}

/// Element-wise exponential minus 1.
///
/// Computes `exp(x) - 1` with greater precision for small `x`.
///
/// - Parameters:
///   - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``exp(_:stream:)``
public func expm1(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_expm1(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Matrix multiplication with matrix-level gather.
///
/// Performs a gather of the operands with the given indices followed by a
/// (possibly batched) matrix multiplication of two arrays.  This operation
/// is more efficient than explicitly applying a `take` followed by a
/// `matmul`.
///
/// The indices `lhsIndices` and `rhsIndices` contain flat indices
/// along the batch dimensions (i.e. all but the last two dimensions) of
/// `a` and `b` respectively.
///
/// For `a` with shape `(A1, A2, ..., AS, M, K)`, `lhsIndices`
/// contains indices from the range `[0, A1 * A2 * ... * AS)`
///
/// For `b` with shape `(B1, B2, ..., BS, M, K)`, `rhsIndices`
/// contains indices from the range `[0, B1 * B2 * ... * BS)`
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``matmul(_:_:stream:)``
public func gatherMatmul(
    _ a: MLXArray, _ b: MLXArray, lhsIndices: MLXArray? = nil, rhsIndices: MLXArray? = nil,
    sortedIndices: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()

    mlx_gather_mm(
        &result, a.ctx, b.ctx, (lhsIndices ?? .mlxNone).ctx, (rhsIndices ?? .mlxNone).ctx,
        sortedIndices, stream.ctx)

    return MLXArray(result)
}

/// Perform quantized matrix multiplication with matrix-level gather.
///
/// This operation is the quantized equivalent to ``gatherMatmul(_:_:lhsIndices:rhsIndices:stream:)``
///
/// Note that ``scales`` and ``biases`` must have the same batch dimensions
/// as ``w`` since they represent the same quantized matrix.
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)``
public func gatherQuantizedMatmul(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray,
    lhsIndices: MLXArray? = nil, rhsIndices: MLXArray? = nil,
    transpose: Bool = true, groupSize: Int = 64, bits: Int = 4,
    sortedIndices: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()

    mlx_gather_qmm(
        &result,
        x.ctx, w.ctx, scales.ctx, biases.ctx, (lhsIndices ?? .mlxNone).ctx,
        (rhsIndices ?? .mlxNone).ctx, transpose,
        groupSize.int32, bits.int32, sortedIndices, stream.ctx)

    return MLXArray(result)
}

/// Element-wise greater than.
///
/// Greater than on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// if (a .> b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func greater<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_greater(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise less greater than or equal.
///
/// Greater than or equal on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// if (a .>= b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func greaterEqual<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_greater_equal(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Perform the Walsh-Hadamard transform along the final axis.
///
/// Supports sizes `n = m*2^k` for `m` in `(1, 12, 20, 28)` and `2^k <= 8192`
/// for ``DType/float32`` and `2^k <= 16384` for ``DType/float16`` and ``DType/bfloat16``.
///
/// - Parameters:
///   - array: input array
///   - scale: scale the output by this factor -- default is `1.0/sqrt(array.dim(-1))`
///   - stream: stream to evaluate on
public func hadamardTransform(
    _ array: MLXArray, scale: Float? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    let scale = mlx_optional_float(value: scale ?? 0, has_value: scale != nil)
    var result = mlx_array_new()
    mlx_hadamard_transform(&result, array.ctx, scale, stream.ctx)
    return MLXArray(result)
}

/// Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - stream: stream or device to evaluate on
/// - Returns: inner product
///
/// ### See Also
/// - <doc:arithmetic>
public func inner(
    _ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_inner(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Returns a boolean array where two arrays are element-wise equal within a tolerance.
///
/// Infinite values are considered equal if they have the same sign, NaN values are not equal unless
/// `equalNAN` is `true`.
///
/// Two values are considered close if:
///
/// ```swift
/// abs(a - b) <= (atol + rtol * abs(b))
/// ```
///
/// Unlike ``arrayEqual(_:_:equalNAN:stream:)`` this function supports <doc:broadcasting>.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - rtol: relative tolerance (see discussion)
///   - atol: absolute tolerance (see discussion)
///   - equalNaN: if `true` treat NaN values as equal to each other
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``allClose(_:_:rtol:atol:equalNaN:stream:)``
/// - ``arrayEqual(_:_:equalNAN:stream:)``
public func isClose(
    _ a: MLXArray, _ b: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, equalNaN: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_isclose(&result, a.ctx, b.ctx, rtol, atol, equalNaN, stream.ctx)
    return MLXArray(result)
}

/// Return a boolean array indicating which elements are NaN.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The boolean array indicating which elements are NaN.
///
/// ### See Also
/// - <doc:arithmetic>
public func isNaN(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_isnan(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Return a boolean array indicating which elements are infinity.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The boolean array indicating which elements are infinity.
///
/// ### See Also
/// - <doc:arithmetic>
public func isInf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_isinf(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Return a boolean array indicating which elements are finite.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The boolean array indicating which elements are infinity.
///
/// ### See Also
/// - <doc:arithmetic>
public func isFinite(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_isfinite(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Return a boolean array indicating which elements are negative infinity.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The boolean array indicating which elements are negative infinity.
///
/// ### See Also
/// - <doc:arithmetic>
public func isNegInf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_isneginf(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Return a boolean array indicating which elements are positive infinity.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The boolean array indicating which elements are positive infinity.
///
/// ### See Also
/// - <doc:arithmetic>
public func isPosInf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_isposinf(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise less than.
///
/// Less than on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// if (a .< b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func less<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_less(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise less than or equal.
///
/// Less than or equal on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// if (a .<= b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func lessEqual<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_less_equal(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise log-add-exp.
///
/// This is a numerically stable log-add-exp of two arrays with numpy-style
/// broadcasting semantics. Either or both input arrays can also be scalars.
///
/// The computation is is a numerically stable version of `log(exp(a) + exp(b))`.
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func logAddExp<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_logaddexp(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise logical and.
///
/// Logical and on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// // equivalent
/// let r = (a .< b) .&& ((a + 1) .> b)
/// let r2 = logicalAnd((a .< b), ((a + 1) .> b))
/// ```
///
/// - Parameters:
///   - a: input array or scalar
///   - b: input array or scalar
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - <doc:logical>
/// - ``MLXArray/.&&(_:_:)``
public func logicalAnd<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_logical_and(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise logical not.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
/// let r = !(a == b)
/// ```
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - <doc:logical>
public func logicalNot(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_logical_not(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise logical or.
///
/// Logical or on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// // equivalent
/// let r = (a .< b) .|| ((a + 1) .> b)
/// let r2 = logicalOr((a .< b), ((a + 1) .> b))
/// ```
///
/// - Parameters:
///   - a: input array or scalar
///   - b: input array or scalar
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - <doc:logical>
/// - ``MLXArray/.||(_:_:)``
public func logicalOr<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_logical_or(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Indexing mode for ``meshGrid(_:sparse:indexing:stream:)``.
public enum MeshGridIndexing: String, Sendable {
    /// cartesian indexing
    case xy

    /// matrix indexing
    case ij
}

/// Generate multidimensional coordinate grids from 1-D coordinate arrays
///
/// - Parameters:
///   - arrays: input arrays
///   - sparse: if `true` a parse grid is returned in which each output array has a single
///     non-zero element, otherwise a dense grid is returned.
///   - indexing: indexing mode
///   - stream: stream or device to evaluate on
public func meshGrid(
    _ arrays: [MLXArray], sparse: Bool = false, indexing: MeshGridIndexing = .xy,
    stream: StreamOrDevice = .default
) -> [MLXArray] {
    let mlxArrays = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(mlxArrays) }

    var vec = mlx_vector_array_new()

    mlx_meshgrid(&vec, mlxArrays, sparse, indexing.rawValue.cString(using: .utf8), stream.ctx)
    defer { mlx_vector_array_free(vec) }

    return mlx_vector_array_values(vec)
}

/// Element-wise maximum.
///
/// Take the element-wise max of two arrays with <doc:broadcasting>
/// semantics.
///
/// - Parameters:
///     - a: the first array
///     - b: the second array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``minimum(_:_:stream:)``
public func maximum<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_maximum(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise minimum.
///
/// Take the element-wise min of two arrays with <doc:broadcasting>
/// semantics.
///
/// - Parameters:
///     - a: the first array
///     - b: the second array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``maximum(_:_:stream:)``
public func minimum<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_minimum(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise multiplication.
///
/// Multiply two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// let r = a * b * 7
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func multiply<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_multiply(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Replace NaN and Inf values with finite numbers.
///
/// - Parameters:
///   - array: input array
///   - nan: value to replace NaN with
///   - posInf: value to replace positive inifinites with.  If not specified will use
///     the largest finite value for the given dtype.
///   - negInf: value to replace negative inifinites with.  If not specified will use
///     the negative of the largest finite value for the given dtype.
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func nanToNum(
    _ array: MLXArray,
    nan: Float = 0, posInf: Float? = 0, negInf: Float? = 0,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let posInf = mlx_optional_float(value: posInf ?? 0, has_value: posInf != nil)
    let negInf = mlx_optional_float(value: negInf ?? 0, has_value: negInf != nil)
    var result = mlx_array_new()
    mlx_nan_to_num(&result, array.ctx, nan, posInf, negInf, stream.ctx)
    return MLXArray(result)
}

/// Element-wise negation.
///
/// Negate the values in the array.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let r = negative(a) // e.g. -a
/// ```
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func negative(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_negative(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise not equal.
///
/// Not equal on two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = a + 1
///
/// // equivalent to if (a .!= b).all().item() {
/// if notEqual(a, b).all().item() {
///     ...
/// }
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
public func notEqual<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_not_equal(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - stream: stream or device to evaluate on
/// - Returns: outer product
///
/// ### See Also
/// - <doc:arithmetic>
public func outer(
    _ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_outer(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Mode for ``padded(_:width:value:stream:)``
public enum PadMode: String {
    /// pads with constant value
    case constant
    /// pads with the edge values of the array
    case edge
}

/// Pad an array with a constant value.
///
/// - Parameters:
///     - array: the array to pad
///     - width: either an `Int` number of values to pad before AND after each axis or an array of 2 giving the
///             before and after counts
///     - mode: padding mode, see ``PadMode``
///     - value: constant value to pad the edges with
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``padded(_:widths:value:stream:)``
public func padded(
    _ array: MLXArray, width: IntOrPair, mode: PadMode = .constant, value: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = (0 ..< ndim).map { _ in width.first.int32 }
    let highPads = (0 ..< ndim).map { _ in width.second.int32 }
    let value = value ?? MLXArray(0, dtype: array.dtype)

    var result = mlx_array_new()
    mlx_pad(
        &result,
        array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx,
        mode.rawValue.cString(using: .utf8), stream.ctx)
    return MLXArray(result)
}

/// Pad an array with a constant value.
///
/// - Parameters:
///     - array: the array to pad
///     - widths: array of int or pairs giving the before/after amounts for each axis
///     - mode: padding mode, see ``PadMode``
///     - value: constant value to pad the edges with
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``padded(_:width:value:stream:)``
public func padded(
    _ array: MLXArray, widths: [IntOrPair], mode: PadMode = .constant, value: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = widths.map { $0.first.int32 }
    let highPads = widths.map { $0.second.int32 }
    let value = value ?? MLXArray(0, dtype: array.dtype)

    var result = mlx_array_new()
    mlx_pad(
        &result,
        array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx,
        mode.rawValue.cString(using: .utf8), stream.ctx)
    return MLXArray(result)
}

/// Returns a partitioned copy of the array such that the smaller `kth`
/// elements are first.
///
/// The ordering of the elements in partitions is undefined.
///
/// - Parameters:
///     - array: input array
///     - kth: Element at the `kth` index will be in its sorted
///                   position in the output. All elements before the kth index will
///                   be less or equal to the `kth` element and all elements after
///                   will be greater or equal to the `kth` element in the output.
///     - axis: axis to partition over
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``partitioned(_:kth:stream:)``
/// - ``argPartition(_:kth:axis:stream:)``
public func partitioned(_ array: MLXArray, kth: Int, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_partition_axis(&result, array.ctx, kth.int32, axis.int32, stream.ctx)
    return MLXArray(result)
}

///
/// Returns a partitioned copy of the flattened array such that the smaller `kth`
/// elements are first.
///
/// The ordering of the elements in partitions is undefined.
///
/// - Parameters:
///     - array: input array
///     - kth: Element at the `kth` index will be in its sorted
///                   position in the output. All elements before the kth index will
///                   be less or equal to the `kth` element and all elements after
///                   will be greater or equal to the `kth` element in the output.
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``partitioned(_:kth:axis:stream:)``
/// - ``argPartition(_:kth:axis:stream:)``
public func partitioned(_ array: MLXArray, kth: Int, stream: StreamOrDevice = .default) -> MLXArray
{
    var result = mlx_array_new()
    mlx_partition(&result, array.ctx, kth.int32, stream.ctx)
    return MLXArray(result)
}

/// Put values along an axis at the specified indices.
///
/// - Parameters:
///     - array: destination array
///     - indices: Indices array. These should be broadcastable with the input array excluding the `axis` dimension.
///     - values: Values array. These should be broadcastable with the indices.
///     - axis: Axis in the destination to put the values to
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``takeAlong(_:_:stream:)``
public func putAlong(
    _ array: MLXArray, _ indices: MLXArray, values: MLXArray, axis: Int,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_put_along_axis(&result, array.ctx, indices.ctx, values.ctx, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Put values along an axis at the specified indices in a flattened array.
///
/// - Parameters:
///     - array: destination array
///     - indices: Indices array. These should be broadcastable with the flattened input array
///     - values: Values array. These should be broadcastable with the flattened input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``takeAlong(_:_:axis:stream:)
public func putAlong(
    _ array: MLXArray, _ indices: MLXArray, values: MLXArray, stream: StreamOrDevice = .default
)
    -> MLXArray
{
    let input = array.reshaped([-1], stream: stream)
    var result = mlx_array_new()
    mlx_put_along_axis(&result, input.ctx, indices.ctx, values.ctx, 0, stream.ctx)
    return MLXArray(result).reshaped(array.shape, stream: stream)
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized
/// together. Hence, number of columns of `w` should be divisible by
/// `group_size`. In particular, the rows of `w` are divided into groups of
/// size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// ### See Also
/// - ``dequantized(_:scales:biases:groupSize:bits:stream:)``
/// - ``quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)``
public func quantized(
    _ w: MLXArray, groupSize: Int = 64, bits: Int = 4, stream: StreamOrDevice = .default
) -> (wq: MLXArray, scales: MLXArray, biases: MLXArray) {
    var r1 = mlx_array_new()
    var r2 = mlx_array_new()
    var r3 = mlx_array_new()
    mlx_quantize(&r1, &r2, &r3, w.ctx, groupSize.int32, bits.int32, stream.ctx)

    return (MLXArray(r1), MLXArray(r2), MLXArray(r3))
}

/// Perform the matrix multiplication with the quantized matrix `w`. The
/// quantization uses one floating point scale and bias per `group_size` of
/// elements. Each element in `w` takes `bits` bits and is packed in an
/// unsigned 32 bit integer.
///
/// ### See Also
/// - ``dequantized(_:scales:biases:groupSize:bits:stream:)``
/// - ``quantized(_:groupSize:bits:stream:)``
public func quantizedMatmul(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray, transpose: Bool = true,
    groupSize: Int = 64, bits: Int = 4, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_quantized_matmul(
        &result,
        x.ctx, w.ctx, scales.ctx, biases.ctx, transpose, groupSize.int32, bits.int32, stream.ctx
    )
    return MLXArray(result)
}

/// Convert angles from degrees to radians.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``degrees(_:stream:)``
public func radians(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_radians(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise remainder of division.
///
/// Computes the remainder of dividing `lhs` with `rhs` with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
///
/// let r = remainder(a, 2) // e.g. a % 2
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func remainder<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_remainder(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Roll array elements along a given axis.
///
/// Elements that are rolled beyond the end of the array are introduced at the beggining and vice-versa.
///
/// - Parameters:
///   - a: input array
///   - shift: The number of places by which elements
///     are shifted. If positive the array is rolled to the right, if
///     negative it is rolled to the left.
///   - axis: the axis along which to roll the elements
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
public func roll(_ a: MLXArray, shift: Int, axis: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_roll_axis(&result, a.ctx, [shift.int32], 1, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Roll array elements along a given axis.
///
/// Elements that are rolled beyond the end of the array are introduced at the beggining and vice-versa.
///
/// - Parameters:
///   - a: input array
///   - shift: The number of places by which elements
///     are shifted. If positive the array is rolled to the right, if
///     negative it is rolled to the left.
///   - axes: the axes along which to roll the elements, or all if omitted
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
public func roll(_ a: MLXArray, shift: Int, axes: [Int]? = nil, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    if let axes {
        mlx_roll_axes(&result, a.ctx, [shift.int32], 1, axes.asInt32, axes.count, stream.ctx)
    } else {
        mlx_roll(&result, a.ctx, [shift.int32], 1, stream.ctx)
    }
    return MLXArray(result)
}

/// Element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func sigmoid(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_sigmoid(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise sign.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func sign(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_sign(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise hyperbolic sine.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``sin(_:stream:)``
public func sinh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_sinh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

@available(*, deprecated, renamed: "softmax(_:axes:precise:stream:)")
@_documentation(visibility: internal)
public func softMax(
    _ array: MLXArray, axes: [Int], precise: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    softmax(array, axes: axes, precise: precise, stream: stream)
}

/// Perform the softmax along the given axis.
///
/// This operation is a numerically stable version of:
///
/// ```swift
///exp(a) / sum(exp(a), axis, keepdims: true)
/// ```
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``softmax(_:axis:precise:stream:)``
/// - ``softmax(_:precise:stream:)``
public func softmax(
    _ array: MLXArray, axes: [Int], precise: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_softmax_axes(&result, array.ctx, axes.asInt32, axes.count, precise, stream.ctx)
    return MLXArray(result)
}

@available(*, deprecated, renamed: "softmax(_:axis:precise:stream:)")
@_documentation(visibility: internal)
public func softMax(
    _ array: MLXArray, axis: Int, precise: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    softmax(array, axis: axis, precise: precise, stream: stream)
}

/// Perform the softmax along the given axis.
///
/// This operation is a numerically stable version of:
///
/// ```swift
///exp(a) / sum(exp(a), axis, keepdims: true)
/// ```
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``softmax(_:axes:precise:stream:)``
/// - ``softmax(_:precise:stream:)``
public func softmax(
    _ array: MLXArray, axis: Int, precise: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_softmax_axis(&result, array.ctx, axis.int32, precise, stream.ctx)
    return MLXArray(result)
}

@available(*, deprecated, renamed: "softmax(_:axis:precise:stream:)")
@_documentation(visibility: internal)
public func softMax(_ array: MLXArray, precise: Bool = false, stream: StreamOrDevice = .default)
    -> MLXArray
{
    softmax(array, precise: precise, stream: stream)
}

/// Perform the softmax along the given axis.
///
/// This operation is a numerically stable version of:
///
/// ```swift
///exp(a) / sum(exp(a), axis, keepdims: true)
/// ```
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``softmax(_:axes:precise:stream:)``
/// - ``softmax(_:axis:precise:stream:)``
public func softmax(_ array: MLXArray, precise: Bool = false, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_softmax(&result, array.ctx, precise, stream.ctx)
    return MLXArray(result)
}

/// Returns a sorted copy of the array.
///
/// - Parameters:
///     - array: input array
///     - axis: axis to sort over
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``sorted(_:stream:)``
/// - ``argSort(_:axis:stream:)``
public func sorted(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_sort_axis(&result, array.ctx, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Returns a sorted copy of the flattened array.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``sorted(_:axis:stream:)``
/// - ``argSort(_:axis:stream:)``
public func sorted(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_sort(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Compute the standard deviation(s) over the given axes.
///
/// - Parameters:
///   - array: input array
///   - axes: axes to reduce over
///   - keepDims: if `true`keep reduced axis as singleton dimension
///   - ddof: the divisor to compute the varian is `N - ddof`
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``std(_:axis:keepDims:ddof:stream:)``
/// - ``std(_:keepDims:ddof:stream:)``
public func std(
    _ array: MLXArray, axes: [Int], keepDims: Bool = false, ddof: Int = 0,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_std_axes(&result, array.ctx, axes.asInt32, axes.count, keepDims, ddof.int32, stream.ctx)
    return MLXArray(result)
}

/// Compute the standard deviation over the given axis.
///
/// - Parameters:
///   - array: input array
///   - axis: axis to reduce over
///   - keepDims: if `true`keep reduced axis as singleton dimension
///   - ddof: the divisor to compute the varian is `N - ddof`
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``std(_:axes:keepDims:ddof:stream:)``
/// - ``std(_:keepDims:ddof:stream:)``
public func std(
    _ array: MLXArray, axis: Int, keepDims: Bool = false, ddof: Int = 0,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_std_axis(&result, array.ctx, axis.int32, keepDims, ddof.int32, stream.ctx)
    return MLXArray(result)
}

/// Compute the standard deviations over all axes.
///
/// - Parameters:
///   - array: input array
///   - keepDims: if `true`keep reduced axis as singleton dimension
///   - ddof: the divisor to compute the varian is `N - ddof`
///   - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:reduction>
/// - ``std(_:axes:keepDims:ddof:stream:)``
/// - ``std(_:axis:keepDims:ddof:stream:)``
public func std(
    _ array: MLXArray, keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_std(&result, array.ctx, keepDims, ddof.int32, stream.ctx)
    return MLXArray(result)
}

/// Stacks the arrays along a new axis.
///
/// ### See Also
/// - <doc:shapes>
public func stacked(_ arrays: [MLXArray], axis: Int = 0, stream: StreamOrDevice = .default)
    -> MLXArray
{
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(vector_array) }
    var result = mlx_array_new()
    mlx_stack_axis(&result, vector_array, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Stop gradients from being computed.
///
///The operation is the identity but it prevents gradients from flowing
/// through the array.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
public func stopGradient(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_stop_gradient(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise subtraction.
///
/// Subtract two arrays with <doc:broadcasting>.
///
/// For example:
///
/// ```swift
/// let a = MLXArray(0 ..< 12, [4, 3])
/// let b = MLXArray([4, 5, 6])
///
/// let r = subtract(a, b) // e.g. a - b
/// ```
///
/// - Parameters:
///     - a: the left hand side array
///     - b: the right hand side array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func subtract<A: ScalarOrArray, B: ScalarOrArray>(
    _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_subtract(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Take values along an axis at the specified indices.
///
/// - Parameters:
///     - array: the left hand side array
///     - indices: should be broadcastable to `array` excluding the `axis` dimension
///     - axis: axis in the input to take the values from
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``takeAlong(_:_:stream:)``
public func takeAlong(
    _ array: MLXArray, _ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_take_along_axis(&result, array.ctx, indices.ctx, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Take values along an axis at the specified indices from a flattened array.
///
/// - Parameters:
///     - array: the left hand side array
///     - indices: should be broadcastable to the flattened `array`
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:indexes>
/// - ``takeAlong(_:_:axis:stream:)
public func takeAlong(_ array: MLXArray, _ indices: MLXArray, stream: StreamOrDevice = .default)
    -> MLXArray
{
    let array = array.reshaped([-1], stream: stream)
    var result = mlx_array_new()
    mlx_take_along_axis(&result, array.ctx, indices.ctx, 0, stream.ctx)
    return MLXArray(result)
}

/// Element-wise tangent.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func tan(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_tan(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Element-wise hyperbolic tangent.
///
/// - Parameters:
///     - array: input array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:arithmetic>
public func tanh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_tanh(&result, array.ctx, stream.ctx)
    return MLXArray(result)
}

/// Compute tensor dot product.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - axes: sum over the last `axes` dimensions
///   - stream: stream or device to evaluate on
/// - Returns: tensor dot product
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``tensordot(_:_:axes:stream:)-8yqyi``
public func tensordot(
    _ a: MLXArray, _ b: MLXArray, axes: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_tensordot_axis(&result, a.ctx, b.ctx, axes.int32, stream.ctx)
    return MLXArray(result)
}

/// Compute tensor dot product.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - axes: two ranges for the `a` and `b` dimensions
///   - stream: stream or device to evaluate on
/// - Returns: tensor dot product
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``tensordot(_:_:axes:stream:)-3qkgq``
public func tensordot(
    _ a: MLXArray, _ b: MLXArray, axes: ((Int, Int), (Int, Int)), stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_tensordot(
        &result,
        a.ctx, b.ctx, [axes.0.0, axes.0.1].asInt32, 2, [axes.1.0, axes.1.1].asInt32, 2,
        stream.ctx)
    return MLXArray(result)
}

/// Compute tensor dot product.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - axes: multiple ranges for the `a` and `b` dimensions
///   - stream: stream or device to evaluate on
/// - Returns: tensor dot product
///
/// ### See Also
/// - <doc:arithmetic>
/// - ``tensordot(_:_:axes:stream:)``
public func tensordot(
    _ a: MLXArray, _ b: MLXArray, axes: ([Int], [Int]), stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_tensordot(
        &result,
        a.ctx, b.ctx, axes.0.asInt32, axes.0.count, axes.1.asInt32, axes.1.count,
        stream.ctx)
    return MLXArray(result)
}

/// Construct array by repeating given array the number of times given by `repetitions`.
///
/// - Parameters:
///   - array: input array
///   - repetitions: number of repetitions for each axis
///   - stream: stream or device to evaluate on
/// - Returns: tiled array
///
/// ### See Also
/// - <doc:shapes>
/// - ``tiled(_:repetitions:stream:)-eouf``
public func tiled(_ array: MLXArray, repetitions: [Int], stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_tile(&result, array.ctx, repetitions.asInt32, repetitions.count, stream.ctx)
    return MLXArray(result)
}

/// Construct array by repeating given array the number of times given by `repetitions`.
///
/// - Parameters:
///   - array: input array
///   - repetitions: number of repetitions for all axes
///   - stream: stream or device to evaluate on
/// - Returns: tiled array
///
/// ### See Also
/// - <doc:shapes>
/// - ``tiled(_:repetitions:stream:)-72ntc``
public func tiled(_ array: MLXArray, repetitions: Int, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_tile(&result, array.ctx, [repetitions.int32], 1, stream.ctx)
    return MLXArray(result)
}

/// Returns the `k` largest elements from the input along a given axis.
///
/// The elements will not necessarily be in sorted order.
///
/// - Parameters:
///     - array: input array
///     - k: how many values
///     - axis: axis to select over
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``top(_:k:stream:)``
public func top(_ array: MLXArray, k: Int, axis: Int = -1, stream: StreamOrDevice = .default)
    -> MLXArray
{
    var result = mlx_array_new()
    mlx_topk_axis(&result, array.ctx, k.int32, axis.int32, stream.ctx)
    return MLXArray(result)
}

/// Returns the `k` largest elements from the flattened input along a given axis.
///
/// The elements will not necessarily be in sorted order.
///
/// - Parameters:
///     - array: input array
///     - k: how many values
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:sorting>
/// - ``top(_:k:axis:stream:)``
public func top(_ array: MLXArray, k: Int, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_topk(&result, array.ctx, k.int32, stream.ctx)
    return MLXArray(result)
}

/// Return the sum along a specified diagonal in the given array.
///
/// - Parameters:
///   - array: input array
///   - offset: Offset of the diagonal from the main diagonal
///   - axis1: The first axis of the 2-D sub-arrays from which the diagonals should be taken
///   - axis2: The second axis of the 2-D sub-arrays from which the diagonals should be taken
///   - dtype: Data type of the output array. If unspecified the output type is inferred from the input array.
///   - stream: stream or device to evaluate on
/// - Returns: sum of specified diagonal.
///
/// ### See Also
/// - <doc:arithmetic>
public func trace(
    _ array: MLXArray, offset: Int = 0, axis1: Int = 0, axis2: Int = 1, dtype: DType? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_trace(
        &result,
        array.ctx, offset.int32, axis1.int32, axis2.int32, (dtype ?? array.dtype).cmlxDtype,
        stream.ctx)
    return MLXArray(result)
}

/// Zeros the array above the given diagonal.
///
/// - Parameters:
///     - array: input array
///     - k: the diagonal of the 2-D array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``triu(_:k:stream:)``
public func tril(_ array: MLXArray, k: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_tril(&result, array.ctx, k.int32, stream.ctx)
    return MLXArray(result)
}

/// Zeros the array below the given diagonal.
///
/// - Parameters:
///     - array: input array
///     - k: the diagonal of the 2-D array
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``tril(_:k:stream:)``
public func triu(_ array: MLXArray, k: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_triu(&result, array.ctx, k.int32, stream.ctx)
    return MLXArray(result)
}

/// Select from `x` or `y` according to `condition`.
///
/// The condition and input arrays must be the same shape or <doc:broadcasting>
/// with each another.
///
/// > ``which(_:_:_:stream:)`` may be easier to use (`where` is a Swift keyword).
///
/// - Parameters:
///     - condition: condition array
///     - a: input selected from where condiiton is non-zero or `true`
///     - b: input selected from where condiiton is zero or `false`
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
/// - ``which(_:_:_:stream:)``
public func `where`<A: ScalarOrArray, B: ScalarOrArray>(
    _ condition: MLXArray, _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_where(&result, condition.ctx, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Alias for ``where(_:_:_:stream:)`` -- select from `x` or `y` according to `condition`.
///
/// The condition and input arrays must be the same shape or <doc:broadcasting>
/// with each another.
///
/// - Parameters:
///     - condition: condition array
///     - a: input selected from where condiiton is non-zero or `true`
///     - b: input selected from where condiiton is zero or `false`
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:logical>
/// - ``where(_:_:_:stream:)``
public func which<A: ScalarOrArray, B: ScalarOrArray>(
    _ condition: MLXArray, _ a: A, _ b: B, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    var result = mlx_array_new()
    mlx_where(&result, condition.ctx, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Compute the Kronecker product of two arrays ``a`` and ``b``.
///
/// - Parameters:
///     - a: input array
///     - b: input array
///     - stream: stream or device to evaluate on
public func kron(
    _ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_kron(&result, a.ctx, b.ctx, stream.ctx)
    return MLXArray(result)
}

/// Flatten an array.
///
/// The axes flattened will be between ``start_axis`` and ``end_axis``,
/// inclusive. Negative axes are supported. After converting negative axis to
/// positive, axes outside the valid range will be clamped to a valid value,
/// ``start_axis`` to ``0`` and ``end_axis`` to ``ndim - 1``.
///
/// - Parameters:
///     - a: input array
///     - start_axis: first dim to flatten
///     - end_axis: last dim to flatten
///     - stream: stream or device to evaluate on
public func flatten(
    _ a: MLXArray, startAxis: Int, endAxis: Int = -1, stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_flatten(&result, a.ctx, startAxis.int32, endAxis.int32, stream.ctx)
    return MLXArray(result)
}

/// Unflatten an axis of an array to a shape.
///
/// - Parameters:
///     - a: input array
///     - axis: axis to unflatten
///     - shape: shape to unflatten into
///     - stream: stream or device to evaluate on
public func unflatten(
    _ a: MLXArray, axis: Int, shape: [Int], stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    mlx_unflatten(&result, a.ctx, axis.int32, shape.map { Int32($0) }, shape.count, stream.ctx)
    return MLXArray(result)
}
