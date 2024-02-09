// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX

/// One dimensional discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n`.  If not specified `array.dim(axis)` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axis
///
/// ### See Also
/// - <doc:MLXFFT>
public func fft(_ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default)
    -> MLXArray
{
    MLXArray(
        mlx_fft_fftn(array.ctx, [(n ?? array.dim(axis)).int32], 1, [axis.int32], 1, stream.ctx))
}

/// One dimensional inverse discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n`.  If not specified `array.dim(axis)` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse DFT of the input along the given axis
///
/// ### See Also
/// - <doc:MLXFFT>
public func ifft(
    _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_fft_ifft(array.ctx, (n ?? array.dim(axis)).int32, axis.int32, stream.ctx))
}

/// Two dimensional discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axis)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axes
///
/// ### See Also
/// - <doc:MLXFFT>
public func fft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    fftn(array, s: s, axes: axes, stream: stream)
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axis)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse DFT of the input along the given axes
///
/// ### See Also
/// - <doc:MLXFFT>
public func ifft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    ifftn(array, s: s, axes: axes, stream: stream)
}

/// n-dimensional discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axis)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axes
///
/// ### See Also
/// - <doc:MLXFFT>
public func fftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    if let s, let axes {
        // both supplied
        return MLXArray(
            mlx_fft_fft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else if let axes {
        // no n, compute from dim()
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_fft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    } else if let s {
        // axes are the rightmost dimensions matching the number of dimensions of n
        let axes = Array(-s.count ..< 0)
        return MLXArray(
            mlx_fft_fft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else {
        let axes = Array(0 ..< array.ndim)
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_fft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    }
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axes)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse DFT of the input along the given axes
///
/// ### See Also
/// - <doc:MLXFFT>
public func ifftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    if let s, let axes {
        // both supplied
        return MLXArray(
            mlx_fft_ifft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else if let axes {
        // no n, compute from dim()
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_ifft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    } else if let s {
        // axes are the rightmost dimensions matching the number of dimensions of n
        let axes = Array(-s.count ..< 0)
        return MLXArray(
            mlx_fft_ifft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else {
        let axes = Array(0 ..< array.ndim)
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_ifft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    }
}

/// One dimensional discrete Fourier Transform on a real input.
///
/// The output has the same shape as the input except along `axis` in
/// which case it has size `n / 2 + 1`.
///
/// - Parameters:
///   - array: input array.  If the array is complex it will be silently cast to a real type
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n`.  If not specified `array.dim(axis)` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axis.  The output data type will be complex.
///
/// ### See Also
/// - <doc:MLXFFT>
public func rfft(
    _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        mlx_fft_rfftn(array.ctx, [(n ?? array.dim(axis)).int32], 1, [axis.int32], 1, stream.ctx))
}

/// Inverse one dimensional discrete Fourier Transform on a real input.
///
/// The output has the same shape as the input except along `axis` in
/// which case it has size `n`.
///
/// - Parameters:
///   - array: input array.  If the array is complex it will be silently cast to a real type
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfft(_:n:axis:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
public func irfft(
    _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
) -> MLXArray {
    let n = n ?? (array.dim(axis) - 1) * 2
    return MLXArray(mlx_fft_irfft(array.ctx, n.int32, axis.int32, stream.ctx))
}

/// Two dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in
/// `axes` in which case it has sizes from `s`. The last axis in `axes` is
/// treated as the real axis and will have size `s[s.lastIndex] / 2 + 1`.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axes)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axes.  The output data type will be complex.
///
/// ### See Also
/// - <doc:MLXFFT>
public func rfft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    rfftn(array, s: s, axes: axes, stream: stream)
}

/// Inverse two dimensional discrete Fourier Transform on a real input.
///
/// Note the input is generally complex. The dimensions of the input
/// specified in `axes` are padded or truncated to match the sizes
/// from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.lastIndex] / 2 + 1`.
///
/// - Parameters:
///   - array: input array
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfft2(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
public func irfft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    irfftn(array, s: s, axes: axes, stream: stream)
}

/// n-dimensional real discrete Fourier Transform.
///
/// The output has the same shape as the input except along the dimensions in
/// `axes` in which case it has sizes from `s`. The last axis in `axes` is
/// treated as the real axis and will have size `s[s.lastIndex] / 2 + 1`.
///
/// - Parameters:
///   - array: input array
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axes)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: DFT of the input along the given axes.  The output data type will be complex.
///
/// ### See Also
/// - <doc:MLXFFT>
public func rfftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    if let s, let axes {
        // both supplied
        return MLXArray(
            mlx_fft_rfft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else if let axes {
        // no n, compute from dim()
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_rfft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    } else if let s {
        // axes are the rightmost dimensions matching the number of dimensions of n
        let axes = Array(-s.count ..< 0)
        return MLXArray(
            mlx_fft_rfft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else {
        let axes = Array(0 ..< array.ndim)
        let n = axes.map { array.dim($0) }
        return MLXArray(
            mlx_fft_rfft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    }
}

/// Inverse n-dimensional discrete Fourier Transform on a real input.
///
/// Note the input is generally complex. The dimensions of the input
/// specified in `axes` are padded or truncated to match the sizes
/// from `s`. The last axis in `axes` is treated as the real axis
/// and will have size `s[s.lastIndex] / 2 + 1`.
///
/// - Parameters:
///   - array: input array
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfftn(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
public func irfftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    if let s, let axes {
        // both supplied
        return MLXArray(
            mlx_fft_irfft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else if let axes {
        // no n, compute from dim()
        var n = axes.map { array.dim($0) }
        n[n.count - 1] = (n[n.count - 1] - 1) * 2
        return MLXArray(
            mlx_fft_irfft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    } else if let s {
        // axes are the rightmost dimensions matching the number of dimensions of n
        let axes = Array(-s.count ..< 0)
        return MLXArray(
            mlx_fft_irfft2(array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx))
    } else {
        let axes = Array(0 ..< array.ndim)
        var n = axes.map { array.dim($0) }
        n[n.count - 1] = (n[n.count - 1] - 1) * 2
        return MLXArray(
            mlx_fft_irfft2(array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx))
    }
}
