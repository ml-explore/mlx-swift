// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX

@available(
    *, deprecated,
    message: "`import MLXLinalg` is deprecated. All methods are now available through `import MLX"
)
public let deprecationWarning: Void = ()

/// Matrix or vector norm.
///
/// For values of `ord < 1`, the result is, strictly speaking, not a
/// mathematical norm, but it may still be useful for various numerical
/// purposes.
///
/// The following norms can be calculated:
///
/// ord   | norm for matrices            | norm for vectors
/// ----- | ---------------------------- | --------------------------
/// None  | Frobenius norm               | 2-norm
/// 'fro' | Frobenius norm               | --
/// inf   | max(sum(abs(x), axis-1))     | max(abs(x))
/// -inf  | min(sum(abs(x), axis-1))     | min(abs(x))
/// 0     | --                           | sum(x !- 0)
/// 1     | max(sum(abs(x), axis-0))     | as below
/// -1    | min(sum(abs(x), axis-0))     | as below
/// 2     | 2-norm (largest sing. value) | as below
/// -2    | smallest singular value      | as below
/// other | --                           | sum(abs(x)**ord)**(1./ord)
///
/// > Nuclear norm and norms based on singular values are not yet implemented.
///
/// The Frobenius norm is given by G. H. Golub and C. F. Van Loan, *Matrix Computations*,
///        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
///
/// The nuclear norm is the sum of the singular values.
///
/// Both the Frobenius and nuclear norm orders are only defined for
/// matrices and produce a fatal error when `array.ndim != 2`
///
/// - Parameters:
///   - array: input array
///   - ord: order of the norm, see table
///   - axes: axes that hold 2d matrices
///   - keepDims: if `true` the axes which are normed over are left in the result as dimensions with size one
///   - stream: stream to evaluate on
/// - Returns: output containing the norm(s)
///
/// ### See Also
/// - ``norm(_:ord:axes:keepDims:stream:)``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axes: [Int], keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axes: axes, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// For values of `ord < 1`, the result is, strictly speaking, not a
/// mathematical norm, but it may still be useful for various numerical
/// purposes.
///
/// The following norms can be calculated:
///
/// ord   | norm for matrices            | norm for vectors
/// ----- | ---------------------------- | --------------------------
/// None  | Frobenius norm               | 2-norm
/// 'fro' | Frobenius norm               | --
/// inf   | max(sum(abs(x), axis-1))     | max(abs(x))
/// -inf  | min(sum(abs(x), axis-1))     | min(abs(x))
/// 0     | --                           | sum(x !- 0)
/// 1     | max(sum(abs(x), axis-0))     | as below
/// -1    | min(sum(abs(x), axis-0))     | as below
/// 2     | 2-norm (largest sing. value) | as below
/// -2    | smallest singular value      | as below
/// other | --                           | sum(abs(x)**ord)**(1./ord)
///
/// > Nuclear norm and norms based on singular values are not yet implemented.
///
/// The Frobenius norm is given by G. H. Golub and C. F. Van Loan, *Matrix Computations*,
///        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
///
/// The nuclear norm is the sum of the singular values.
///
/// Both the Frobenius and nuclear norm orders are only defined for
/// matrices and produce a fatal error when `array.ndim != 2`
///
/// - Parameters:
///   - array: input array
///   - ord: order of the norm, see table
///   - axes: axes that hold 2d matrices
///   - keepDims: if `true` the axes which are normed over are left in the result as dimensions with size one
///   - stream: stream to evaluate on
/// - Returns: output containing the norm(s)
///
/// ### See Also
/// - ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: Double, axes: [Int], keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axes: axes, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: Double, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axis: IntOrArray? = nil,
    keepDims: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)``
@available(*, deprecated, message: "norm is now available in the main MLX module")
@_disfavoredOverload
public func norm(
    _ array: MLXArray, ord: Double, axis: IntOrArray? = nil,
    keepDims: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// The QR factorization of the input matrix.
///
/// This function supports arrays with at least 2 dimensions. The matrices
/// which are factorized are assumed to be in the last two dimensions of
/// the input.
///
/// - Returns: the `Q` and `R` matrices
@available(*, deprecated, message: "qr is now available in the main MLX module")
@_disfavoredOverload
public func qr(_ array: MLXArray, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray) {
    return MLXLinalg.qr(array, stream: stream)
}

/// The Singular Value Decomposition (SVD) of the input matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input
/// has more than two dimensions, the function iterates over all indices of the first
/// `array.ndim - 2` dimensions and for each combination SVD is applied to the last two indices.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: The `U`, `S`, and `Vt` matrices, such that `A = matmul(U, matmul(diag(S), Vt))`
@available(*, deprecated, message: "svd is now available in the main MLX module")
@_disfavoredOverload
public func svd(_ array: MLXArray, stream: StreamOrDevice = .default) -> (
    MLXArray, MLXArray, MLXArray
) {
    return MLXLinalg.svd(array, stream: stream)
}

/// Compute the inverse of a square matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input
/// has more than two dimensions, the inverse is computed for each matrix
/// in the last two dimensions of `array`.
///
/// - Parameters:
///   - array: input array
///   - stream: stream or device to evaluate on
/// - Returns: `ainv` such that `dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])`
@available(*, deprecated, message: "inv is now available in the main MLX module")
@_disfavoredOverload
public func inv(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    return MLXLinalg.inv(array, stream: stream)
}

/// Compute the inverse of a triangular square matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input
/// has more than two dimensions, the inverse is computed for each matrix
/// in the last two dimensions of `array`.
///
/// - Parameters:
///   - array: input array
///   - upper: true if the array is an upper triangular matrix
///   - stream: stream or device to evaluate on
/// - Returns: `ainv` such that `dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])`
@available(*, deprecated, message: "triInv is now available in the main MLX module")
@_disfavoredOverload
public func triInv(
    _ array: MLXArray, upper: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.triInv(array, stream: stream)
}

/// Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input
/// has more than two dimensions, the Cholesky decomposition is computed for each matrix
/// in the last two dimensions of `a`.
///
/// If the input matrix is not symmetric positive semi-definite, behavior is undefined.
///
/// - Parameters:
///   - array: input array
///   - upper: if true return the upper triangular Cholesky factor, otherwise the lower triangular
///         Cholesky factor.
///   - stream: stream or device to evaluate on
@available(*, deprecated, message: "cholesky is now available in the main MLX module")
@_disfavoredOverload
public func cholesky(_ array: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXLinalg.cholesky(array, upper: upper, stream: stream)
}

/// Compute the inverse of a real symmetric positive semi-definite matrix using it's Cholesky decomposition.
///
/// This function supports arrays with at least 2 dimensions. When the input
/// has more than two dimensions, the Cholesky decomposition is computed for each matrix
/// in the last two dimensions of `a`.
///
/// If the input matrix is not a triangular matrix behavior is undefined.
///
/// - Parameters:
///   - array: input array
///   - upper: if true return the upper triangular Cholesky factor, otherwise the lower triangular
///         Cholesky factor.
///   - stream: stream or device to evaluate on
@available(*, deprecated, message: "choleskyInv is now available in the main MLX module")
@_disfavoredOverload
public func choleskyInv(_ array: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXLinalg.choleskyInv(array, upper: upper, stream: stream)
}

/// Compute the cross product of two arrays along a specified axis.
///
/// The cross product is defined for arrays with size 2 or 3 in the
/// specified axis. If the size is 2 then the third value is assumed
/// to be zero.
///
/// - Parameters:
///   - a: input array
///   - b: input array
///   - axis: axis along which to compute the cross product
///   - stream: stream or device to evaluate on
@available(*, deprecated, message: "cross is now available in the main MLX module")
@_disfavoredOverload
public func cross(_ a: MLXArray, _ b: MLXArray, axis: Int = -1, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXLinalg.cross(a, b, axis: axis, stream: stream)
}

/// Compute the LU factorization of the given matrix ``A``.
///
/// Note, unlike the default behavior of ``scipy.linalg.lu``, the pivots
/// are indices. To reconstruct the input use ``L[P] @ U`` for 2
/// dimensions or ``takeAlong(L, P[.ellipsis, .newAxis], axis: -2) @ U``
/// for more than 2 dimensions.
///
/// To construct the full permuation matrix do:
///
///   P = putAlong(zeros(L), p[.ellipsis, .newAxis], MLXArray(1.0), axis: -1)
///
/// -Parameters:
///   - a: input array.
///   - stream: stream or device
@available(*, deprecated, message: "lu is now available in the main MLX module")
@_disfavoredOverload
public func lu(_ a: MLXArray, stream: StreamOrDevice = .default)
    -> (MLXArray, MLXArray, MLXArray)
{
    return MLXLinalg.lu(a, stream: stream)
}

/// Computes a compact representation of the LU factorization.
///
/// -Parameters:
///   - a: input array.
///   - stream: stream or device
@available(*, deprecated, message: "lu_factor is now available in the main MLX module")
@_disfavoredOverload
public func lu_factor(_ a: MLXArray, stream: StreamOrDevice = .default)
    -> (MLXArray, MLXArray)
{
    return MLXLinalg.lu_factor(a, stream: stream)
}

/// Compute the solution to a system of linear equations `AX = B`.
///
/// -Parameters:
///   - a: input array.
///   - b: input array.
///   - stream: stream or device
@available(*, deprecated, message: "solve is now available in the main MLX module")
@_disfavoredOverload
public func solve(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXLinalg.solve(a, b, stream: stream)
}

///Computes the solution of a triangular system of linear equations `AX = B`.
///
/// -Parameters:
///   - a: input array.
///   - b: input array.
///   - upper: Whether the array is upper or lower triangular
///   - stream: stream or device
@available(*, deprecated, message: "solveTriangular is now available in the main MLX module")
@_disfavoredOverload
public func solveTriangular(
    _ a: MLXArray, _ b: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default
)
    -> MLXArray
{
    return MLXLinalg.solveTriangular(a, b, upper: upper, stream: stream)
}
