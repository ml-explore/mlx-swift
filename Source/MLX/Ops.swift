// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// MARK: - Internal Ops

/// Broadcast a vector of arrays against one another.
func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_free(vector_array) }

    let result = mlx_broadcast_arrays(vector_array, stream.ctx)!
    defer { mlx_free(result) }

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
    return MLXArray(mlx_add(a.ctx, b.ctx, stream.ctx))
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
public func addmm<A: ScalarOrArray, B: ScalarOrArray, C: ScalarOrArray>(
    _ c: C, _ a: A, _ b: B, alpha: Float = 1.0, beta: Float = 1.0, stream: StreamOrDevice = .default
) -> MLXArray {
    let (a, b) = toArrays(a, b)
    let (_, c) = toArrays(a, c)
    return MLXArray(mlx_addmm(c.ctx, a.ctx, b.ctx, alpha, beta, stream.ctx))
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
    MLXArray(mlx_arccos(array.ctx, stream.ctx))
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
    MLXArray(mlx_arccosh(array.ctx, stream.ctx))
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
    MLXArray(mlx_arcsin(array.ctx, stream.ctx))
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
    MLXArray(mlx_arcsinh(array.ctx, stream.ctx))
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
    MLXArray(mlx_arctan(array.ctx, stream.ctx))
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
    MLXArray(mlx_arctanh(array.ctx, stream.ctx))
}

/// Convert array to have at least 1 dimension.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast1D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_atleast_1d(array.ctx, stream.ctx))
}

/// Convert array to have at least 2 dimensions.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast2D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_atleast_2d(array.ctx, stream.ctx))
}

/// Convert array to have at least 3 dimensions.
///
/// ### See Also
/// - <doc:shapes>
public func atLeast3D(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_atleast_3d(array.ctx, stream.ctx))
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
    MLXArray(mlx_argpartition(array.ctx, kth.int32, axis.int32, stream.ctx))
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
    MLXArray(mlx_argpartition_all(array.ctx, kth.int32, stream.ctx))
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
    MLXArray(mlx_argsort(array.ctx, axis.int32, stream.ctx))
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
    MLXArray(mlx_argsort_all(array.ctx, stream.ctx))
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

    let resolvedStrides: [Int]
    if let strides {
        resolvedStrides = strides
    } else {
        var result = [Int]()
        var cum = 1
        for v in shape.reversed() {
            result.append(cum)
            cum = cum * v
        }
        resolvedStrides = result.reversed()
    }

    return MLXArray(
        mlx_as_strided(
            array.ctx, shape.asInt32, shape.count, resolvedStrides, resolvedStrides.count, offset,
            stream.ctx))
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
    MLXArray(mlx_broadcast_to(array.ctx, shape.asInt32, shape.count, stream.ctx))
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
    MLXArray(mlx_ceil(array.ctx, stream.ctx))
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
public func clip(
    _ array: MLXArray, min: MLXArray, max: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_clip(array.ctx, min.ctx, max?.ctx, stream.ctx))
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
/// - ``clip(_:min:max:stream:)``
public func clip(_ array: MLXArray, max: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_clip(array.ctx, nil, max.ctx, stream.ctx))
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
    defer { mlx_free(vector_array) }

    return MLXArray(mlx_concatenate(vector_array, axis.int32, stream.ctx))
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
/// - ``convolve(_:_:mode:stream:)``
public func conv1d(
    _ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0, dilation: Int = 1,
    groups: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        mlx_conv1d(
            array.ctx, weight.ctx, stride.int32, padding.int32, dilation.int32, groups.int32,
            stream.ctx))
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
/// - ``convolve(_:_:mode:stream:)``
/// - ``convGeneral(_:_:strides:padding:kernelDilation:inputDilation:groups:flip:stream:)-9t1sj``
public func conv2d(
    _ array: MLXArray, _ weight: MLXArray, stride: IntOrPair = 1, padding: IntOrPair = 0,
    dilation: IntOrPair = 1, groups: Int = 1, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        mlx_conv2d(
            array.ctx, weight.ctx, stride.first.int32, stride.second.int32, padding.first.int32,
            padding.second.int32, dilation.first.int32, dilation.second.int32, groups.int32,
            stream.ctx))
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
    MLXArray(
        mlx_conv_general(
            array.ctx, weight.ctx,
            strides.asInt32Array, strides.count,
            padding.asInt32Array, padding.count,
            padding.asInt32Array, padding.count,
            kernelDilation.asInt32Array, kernelDilation.count,
            inputDilation.asInt32Array, inputDilation.count,
            groups.int32, flip, stream.ctx))
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
    MLXArray(
        mlx_conv_general(
            array.ctx, weight.ctx,
            strides.asInt32Array, strides.count,
            [padding.0.int32], 1,
            [padding.1.int32], 1,
            kernelDilation.asInt32Array, kernelDilation.count,
            inputDilation.asInt32Array, inputDilation.count,
            groups.int32, flip, stream.ctx))
}

/// Mode for ``convolve(_:_:mode:stream:)``
public enum ConvolveMode {
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

    weight = MLXArray(
        mlx_slice(
            weight.ctx, [weight.dim(0) - 1].asInt32, 1, [-weight.dim(0) - 1].asInt32, 1, [-1], 1,
            stream.ctx))

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

    return MLXArray(mlx_conv1d(input.ctx, weight.ctx, 1, padding.int32, 1, 1, stream.ctx)).reshaped(
        -1, stream: stream)
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
    MLXArray(mlx_cosh(array.ctx, stream.ctx))
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
    MLXArray(mlx_dequantize(w.ctx, scales.ctx, biases.ctx, groupSize.int32, bits.int32, stream.ctx))
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
    return MLXArray(mlx_divide(a.ctx, b.ctx, stream.ctx))
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
    let arrays = mlx_divmod(a.ctx, b.ctx, stream.ctx)!
    defer { mlx_free(arrays) }
    let result = mlx_vector_array_values(arrays)
    return (result[0], result[1])
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
    return MLXArray(mlx_equal(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_erf(array.ctx, stream.ctx))
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
    MLXArray(mlx_erfinv(array.ctx, stream.ctx))
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
    MLXArray(mlx_expand_dims(array.ctx, axes.asInt32, axes.count, stream.ctx))
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
    MLXArray(mlx_expand_dims(array.ctx, [axis.int32], 1, stream.ctx))
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
    return MLXArray(mlx_greater(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_greater_equal(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_inner(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_isclose(a.ctx, b.ctx, rtol, atol, equalNaN, stream.ctx))
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
    MLXArray(mlx_isnan(array.ctx, stream.ctx))
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
    MLXArray(mlx_isinf(array.ctx, stream.ctx))
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
    MLXArray(mlx_isneginf(array.ctx, stream.ctx))
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
    MLXArray(mlx_isposinf(array.ctx, stream.ctx))
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
    return MLXArray(mlx_less(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_less_equal(a.ctx, b.ctx, stream.ctx))
}

enum LoadSaveError: Error {
    case unableToOpen(URL, String)
    case unknownExtension(String)
}

/// Load array from a binary file in `.npy`format.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``save(array:url:stream:)``
/// - ``save(arrays:metadata:url:stream:)``
public func loadArray(url: URL, stream: StreamOrDevice = .default) throws -> MLXArray {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    if let fp = fopen(path, "r") {
        defer { fclose(fp) }

        switch url.pathExtension {
        case "npy":
            return MLXArray(mlx_load_file(fp, stream.ctx))

        default:
            throw LoadSaveError.unknownExtension(url.pathExtension)
        }

    } else {
        let message = String(cString: strerror(errno))
        throw LoadSaveError.unableToOpen(url, message)
    }
}

/// Load dictionary of ``MLXArray`` from a `safetensors` file.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArray(url:stream:)``
/// - ``loadArraysAndMetadata(url:stream:)``
/// - ``save(array:url:stream:)``
/// - ``save(arrays:metadata:url:stream:)``
public func loadArrays(url: URL, stream: StreamOrDevice = .default) throws -> [String: MLXArray] {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)
    let filename = mlx_string_new(path.cString(using: .utf8))!
    defer { mlx_free(filename) }

    switch url.pathExtension {
    case "safetensors":
        let mlx_safetensors = mlx_load_safetensors(filename, stream.ctx)!
        defer { mlx_free(mlx_safetensors) }

        let mlx_arrays = mlx_safetensors_data(mlx_safetensors)!
        defer { mlx_free(mlx_arrays) }

        return mlx_map_array_values(mlx_arrays)
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``loadArray(url:stream:)``
public func loadArraysAndMetadata(url: URL, stream: StreamOrDevice = .default) throws -> (
    [String: MLXArray], [String: String]
) {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)
    let filename = mlx_string_new(path.cString(using: .utf8))!
    defer { mlx_free(filename) }

    switch url.pathExtension {
    case "safetensors":
        let mlx_safetensors = mlx_load_safetensors(filename, stream.ctx)!
        defer { mlx_free(mlx_safetensors) }

        let mlx_arrays = mlx_safetensors_data(mlx_safetensors)!
        defer { mlx_free(mlx_arrays) }

        let mlx_metadata = mlx_safetensors_metadata(mlx_safetensors)!
        defer { mlx_free(mlx_metadata) }

        return (mlx_map_array_values(mlx_arrays), mlx_map_string_values(mlx_metadata))
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
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
    return MLXArray(mlx_logaddexp(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_logical_and(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_logical_not(array.ctx, stream.ctx))
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
    return MLXArray(mlx_logical_or(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_maximum(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_minimum(a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_multiply(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_negative(array.ctx, stream.ctx))
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
    return MLXArray(mlx_not_equal(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_outer(a.ctx, b.ctx, stream.ctx))
}

/// Pad an array with a constant value.
///
/// - Parameters:
///     - array: the array to pad
///     - width: either an `Int` number of values to pad before AND after each axis or an array of 2 giving the
///             before and after counts
///     - value: constant value to pad the edges with
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``padded(_:widths:value:stream:)``
public func padded(
    _ array: MLXArray, width: IntOrPair, value: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = (0 ..< ndim).map { _ in width.first.int32 }
    let highPads = (0 ..< ndim).map { _ in width.second.int32 }
    let value = value ?? MLXArray(0, dtype: array.dtype)

    return MLXArray(
        mlx_pad(array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx, stream.ctx))
}

/// Pad an array with a constant value.
///
/// - Parameters:
///     - array: the array to pad
///     - widths: array of int or pairs giving the before/after amounts for each axis
///     - value: constant value to pad the edges with
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - <doc:shapes>
/// - ``padded(_:width:value:stream:)``
public func padded(
    _ array: MLXArray, widths: [IntOrPair], value: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = widths.map { $0.first.int32 }
    let highPads = widths.map { $0.second.int32 }
    let value = value ?? MLXArray(0, dtype: array.dtype)

    return MLXArray(
        mlx_pad(array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx, stream.ctx))
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
    MLXArray(mlx_partition(array.ctx, kth.int32, axis.int32, stream.ctx))
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
    MLXArray(mlx_partition_all(array.ctx, kth.int32, stream.ctx))
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
    let result = mlx_quantize(w.ctx, groupSize.int32, bits.int32, stream.ctx)!
    defer { mlx_free(result) }

    let arrays = mlx_vector_array_values(result)
    return (arrays[0], arrays[1], arrays[2])
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
    MLXArray(
        mlx_quantized_matmul(
            x.ctx, w.ctx, scales.ctx, biases.ctx, transpose, groupSize.int32, bits.int32, stream.ctx
        ))
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
    return MLXArray(mlx_remainder(a.ctx, b.ctx, stream.ctx))
}

/// Save array to a binary file in `.npy`format.
///
/// - Parameters:
///     - a: array to save
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``save(arrays:metadata:url:stream:)``
/// - ``loadArray(url:stream:)``
/// - ``loadArrays(url:stream:)``
public func save(array: MLXArray, url: URL, stream: StreamOrDevice = .default) throws {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)
    if let fp = fopen(path, "w") {
        defer { fclose(fp) }

        switch url.pathExtension {
        case "npy":
            mlx_save_file(fp, array.ctx)

        default:
            throw LoadSaveError.unknownExtension(url.pathExtension)
        }
    } else {
        let message = String(cString: strerror(errno))
        throw LoadSaveError.unableToOpen(url, message)
    }
}

/// Save dictionary of arrays in `safetensors` format.
///
/// - Parameters:
///     - a: array to save
///     - metadata: metadata to save
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``save(arrays:metadata:url:stream:)``
/// - ``loadArray(url:stream:)``
/// - ``loadArrays(url:stream:)``
public func save(
    arrays: [String: MLXArray], metadata: [String: String] = [:], url: URL,
    stream: StreamOrDevice = .default
) throws {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    let mlx_arrays = new_mlx_array_map(arrays)
    defer { mlx_free(mlx_arrays) }

    let mlx_metadata = new_mlx_string_map(metadata)
    defer { mlx_free(mlx_metadata) }

    switch url.pathExtension {
    case "safetensors":
        if let fp = fopen(path, "r") {
            defer { fclose(fp) }

            mlx_save_safetensors_file(fp, mlx_arrays, mlx_metadata)

        } else {
            let message = String(cString: strerror(errno))
            throw LoadSaveError.unableToOpen(url, message)
        }

    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
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
    MLXArray(mlx_sigmoid(array.ctx, stream.ctx))
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
    MLXArray(mlx_sign(array.ctx, stream.ctx))
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
    MLXArray(mlx_sinh(array.ctx, stream.ctx))
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
/// - ``softMax(_:axis:stream:)``
/// - ``softMax(_:stream:)``
public func softMax(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax(array.ctx, axes.asInt32, axes.count, stream.ctx))
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
/// - ``softMax(_:axes:stream:)``
/// - ``softMax(_:stream:)``
public func softMax(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax(array.ctx, [axis.int32], 1, stream.ctx))
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
/// - ``softMax(_:axes:stream:)``
/// - ``softMax(_:axis:stream:)``
public func softMax(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax_all(array.ctx, stream.ctx))
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
    MLXArray(mlx_sort(array.ctx, axis.int32, stream.ctx))
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
    MLXArray(mlx_sort_all(array.ctx, stream.ctx))
}

/// Stacks the arrays along a new axis.
///
/// ### See Also
/// - <doc:shapes>
public func stacked(_ arrays: [MLXArray], axis: Int = 0, stream: StreamOrDevice = .default)
    -> MLXArray
{
    let vector_array = new_mlx_vector_array(arrays)
    defer { mlx_free(vector_array) }
    return MLXArray(mlx_stack(vector_array, axis.int32, stream.ctx))
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
    MLXArray(mlx_stop_gradient(array.ctx, stream.ctx))
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
    return MLXArray(mlx_subtract(a.ctx, b.ctx, stream.ctx))
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
    MLXArray(mlx_take_along_axis(array.ctx, indices.ctx, axis.int32, stream.ctx))
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
    return MLXArray(mlx_take_along_axis(array.ctx, indices.ctx, 0, stream.ctx))
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
    MLXArray(mlx_tan(array.ctx, stream.ctx))
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
    MLXArray(mlx_tanh(array.ctx, stream.ctx))
}

/// Computer tensor dot product.
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
    MLXArray(mlx_tensordot_along_axis(a.ctx, b.ctx, axes.int32, stream.ctx))
}

/// Computer tensor dot product.
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
    MLXArray(
        mlx_tensordot(
            a.ctx, b.ctx, [axes.0.0, axes.0.1].asInt32, 2, [axes.1.0, axes.1.1].asInt32, 2,
            stream.ctx))
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
    MLXArray(mlx_tile(array.ctx, repetitions.asInt32, repetitions.count, stream.ctx))
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
    MLXArray(mlx_tile(array.ctx, [repetitions.int32], 1, stream.ctx))
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
    MLXArray(mlx_topk(array.ctx, k.int32, axis.int32, stream.ctx))
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
    MLXArray(mlx_topk_all(array.ctx, k.int32, stream.ctx))
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
    MLXArray(mlx_tril(array.ctx, k.int32, stream.ctx))
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
    MLXArray(mlx_triu(array.ctx, k.int32, stream.ctx))
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
    return MLXArray(mlx_where(condition.ctx, a.ctx, b.ctx, stream.ctx))
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
    return MLXArray(mlx_where(condition.ctx, a.ctx, b.ctx, stream.ctx))
}
