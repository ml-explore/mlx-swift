import Foundation
import MLX

@available(
    *, deprecated,
    message: "`import MLXFFT` is deprecated. All methods are now available through `import MLX"
)
public let deprecationWarning: Void = ()

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
@available(*, deprecated, message: "fft is now available in the main MLX module.")
@_disfavoredOverload
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
@available(*, deprecated, message: "ifft is now available in the main MLX module.")
@_disfavoredOverload
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
@available(*, deprecated, message: "fft2 is now available in the main MLX module.")
@_disfavoredOverload
public func fft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "ifft2 is now available in the main MLX module.")
@_disfavoredOverload
public func ifft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "fftn is now available in the main MLX module.")
@_disfavoredOverload
public func fftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "ifftn is now available in the main MLX module.")
@_disfavoredOverload
public func ifftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "rfft is now available in the main MLX module.")
@_disfavoredOverload
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
@available(*, deprecated, message: "rfft2 is now available in the main MLX module.")
@_disfavoredOverload
public func rfft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
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
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfft2(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
@available(*, deprecated, message: "irfft2 is now available in the main MLX module.")
@_disfavoredOverload
public func irfft2(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = [-2, -1], stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "rfftn is now available in the main MLX module.")
@_disfavoredOverload
public func rfftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
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
///   - n: size of the transformed axis.  The corresponding axis in the input is truncated or padded with zeros to
///   match `n / 2 + 1`.  If not specified `array.dim(axis) / 2 + 1` will be used.
///   - axis: axis along which to perform the FFT
///   - stream: stream or device to evaluate on
/// - Returns: inverse of ``rfftn(_:s:axes:stream:)``
///
/// ### See Also
/// - <doc:MLXFFT>
@available(*, deprecated, message: "irfftn is now available in the main MLX module.")
@_disfavoredOverload
public func irfftn(
    _ array: MLXArray, s: [Int]? = nil, axes: [Int]? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXFFT.irfftn(array, s: s, axes: axes, stream: stream)
}
