// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

public enum MLXFFT {

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
    public static func fft(
        _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()

        mlx_fft_fftn(
            &result, array.ctx, [(n ?? array.dim(axis)).int32], 1, [axis.int32], 1, stream.ctx)
        return MLXArray(result)
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
    public static func ifft(
        _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fft_ifft(&result, array.ctx, (n ?? array.dim(axis)).int32, axis.int32, stream.ctx)
        return MLXArray(result)
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
    public static func fft2(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
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
    public static func ifft2(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
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
    public static func fftn(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let s, let axes {
            // both supplied

            mlx_fft_fft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let axes {
            // no n, compute from dim()
            let n = axes.map { array.dim($0) }

            mlx_fft_fft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let s {
            // axes are the rightmost dimensions matching the number of dimensions of n
            let axes = Array(-s.count ..< 0)

            mlx_fft_fft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else {
            let axes = Array(0 ..< array.ndim)
            let n = axes.map { array.dim($0) }

            mlx_fft_fft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
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
    public static func ifftn(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let s, let axes {
            // both supplied

            mlx_fft_ifft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let axes {
            // no n, compute from dim()
            let n = axes.map { array.dim($0) }

            mlx_fft_ifft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let s {
            // axes are the rightmost dimensions matching the number of dimensions of n
            let axes = Array(-s.count ..< 0)

            mlx_fft_ifft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else {
            let axes = Array(0 ..< array.ndim)
            let n = axes.map { array.dim($0) }

            mlx_fft_ifft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
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
    public static func rfft(
        _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fft_rfftn(
            &result, array.ctx, [(n ?? array.dim(axis)).int32], 1, [axis.int32], 1, stream.ctx)
        return MLXArray(result)
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
    public static func irfft(
        _ array: MLXArray, n: Int? = nil, axis: Int = -1, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let n = n ?? (array.dim(axis) - 1) * 2
        var result = mlx_array_new()
        mlx_fft_irfft(&result, array.ctx, n.int32, axis.int32, stream.ctx)
        return MLXArray(result)
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
    public static func rfft2(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
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
    ///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
    ///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
    ///   - axes: axes along which to perform the FFT
    ///   - stream: stream or device to evaluate on
    /// - Returns: inverse of ``rfft2(_:s:axes:stream:)``
    ///
    /// ### See Also
    /// - <doc:MLXFFT>
    public static func irfft2(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
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
    public static func rfftn(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let s, let axes {
            // both supplied

            mlx_fft_rfft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let axes {
            // no n, compute from dim()
            let n = axes.map { array.dim($0) }

            mlx_fft_rfft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let s {
            // axes are the rightmost dimensions matching the number of dimensions of n
            let axes = Array(-s.count ..< 0)

            mlx_fft_rfft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else {
            let axes = Array(0 ..< array.ndim)
            let n = axes.map { array.dim($0) }

            mlx_fft_rfft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
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
    ///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
    ///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
    ///   - axes: axes along which to perform the FFT
    ///   - stream: stream or device to evaluate on
    /// - Returns: inverse of ``rfftn(_:s:axes:stream:)``
    ///
    /// ### See Also
    /// - <doc:MLXFFT>
    public static func irfftn(
        _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
        axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let s, let axes {
            // both supplied

            mlx_fft_irfft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let axes {
            // no n, compute from dim()
            var n = axes.map { array.dim($0) }
            n[n.count - 1] = (n[n.count - 1] - 1) * 2

            mlx_fft_irfft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else if let s {
            // axes are the rightmost dimensions matching the number of dimensions of n
            let axes = Array(-s.count ..< 0)

            mlx_fft_irfft2(
                &result, array.ctx, s.asInt32, s.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        } else {
            let axes = Array(0 ..< array.ndim)
            var n = axes.map { array.dim($0) }
            n[n.count - 1] = (n[n.count - 1] - 1) * 2

            mlx_fft_irfft2(
                &result, array.ctx, n.asInt32, n.count, axes.asInt32, axes.count, stream.ctx)
            return MLXArray(result)
        }
    }

}  // MLXFFT

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
    MLXFFT.fft(array, n: n, axis: axis, stream: stream)
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
    MLXFFT.ifft(array, n: n, axis: axis, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.fft2(array, s: s, axes: axes, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.ifft2(array, s: s, axes: axes, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.fftn(array, s: s, axes: axes, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.ifftn(array, s: s, axes: axes, stream: stream)
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
    MLXFFT.rfft(array, n: n, axis: axis, stream: stream)
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
    MLXFFT.irfft(array, n: n, axis: axis, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [-2, -1], stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.rfft2(array, s: s, axes: axes, stream: stream)
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
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axes)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfft2(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
public func irfft2(
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [-2, -1],
    stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.irfft2(array, s: s, axes: axes, stream: stream)
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
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.rfftn(array, s: s, axes: axes, stream: stream)
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
///   - s: sizes of the transformed axis.  The corresponding axes in the input are truncated or padded with zeros to
///   match `s`.  If not specified `array.dim(axes)` will be used.
///   - axes: axes along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfftn(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
public func irfftn(
    _ array: MLXArray, s: (some Collection<Int>)? = [Int]?.none,
    axes: (some Collection<Int>)? = [Int]?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.irfftn(array, s: s, axes: axes, stream: stream)
}
