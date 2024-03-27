// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX

/// Types of norms available.
///
/// ### See Also
/// - ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
/// - ``norm(_:ord:axes:keepDims:stream:)-3t3ay``
/// - ``MLXLinalg``
public enum NormKind: String {
    /// Frobenius norm
    case fro
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
/// - ``norm(_:ord:axes:keepDims:stream:)-3t3ay``
public func norm(
    _ array: MLXArray, ord: NormKind? = nil, axes: [Int], keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    if let ord {
        let ord_str = mlx_string_new(ord.rawValue.cString(using: .utf8))!
        defer { mlx_free(ord_str) }
        return MLXArray(
            mlx_linalg_norm_ord(array.ctx, ord_str, axes.asInt32, axes.count, keepDims, stream.ctx))
    } else {
        return MLXArray(mlx_linalg_norm(array.ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
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
public func norm(
    _ array: MLXArray, ord: Double, axes: [Int], keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_linalg_norm_p(array.ctx, ord, axes.asInt32, axes.count, keepDims, stream.ctx))
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
public func norm(
    _ array: MLXArray, ord: NormKind? = nil, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    if let ord {
        let ord_str = mlx_string_new(ord.rawValue.cString(using: .utf8))!
        defer { mlx_free(ord_str) }
        return MLXArray(
            mlx_linalg_norm_ord(array.ctx, ord_str, [axis].asInt32, 1, keepDims, stream.ctx))
    } else {
        return MLXArray(mlx_linalg_norm(array.ctx, [axis].asInt32, 1, keepDims, stream.ctx))
    }
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-3t3ay``
public func norm(
    _ array: MLXArray, ord: Double, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_linalg_norm_p(array.ctx, ord, [axis].asInt32, 1, keepDims, stream.ctx))
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-4dwwp``
public func norm(
    _ array: MLXArray, ord: NormKind? = nil, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    if let ord {
        let ord_str = mlx_string_new(ord.rawValue.cString(using: .utf8))!
        defer { mlx_free(ord_str) }
        return MLXArray(mlx_linalg_norm_ord(array.ctx, ord_str, nil, 0, keepDims, stream.ctx))
    } else {
        return MLXArray(mlx_linalg_norm(array.ctx, nil, 0, keepDims, stream.ctx))
    }
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)-3t3ay``
public func norm(
    _ array: MLXArray, ord: Double, keepDims: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(mlx_linalg_norm_p(array.ctx, ord, nil, 0, keepDims, stream.ctx))
}

/// The QR factorization of the input matrix.
///
/// This function supports arrays with at least 2 dimensions. The matrices
/// which are factorized are assumed to be in the last two dimensions of
/// the input.
///
/// - Returns: the `Q` and `R` matrices
public func qr(_ array: MLXArray, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray) {
    let result_vector = mlx_linalg_qr(array.ctx, stream.ctx)!
    defer { mlx_free(result_vector) }

    let arrays = mlx_vector_array_values(result_vector)
    return (arrays[0], arrays[1])
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
public func svd(_ array: MLXArray, stream: StreamOrDevice = .default) -> (
    MLXArray, MLXArray, MLXArray
) {
    let mlx_arrays = mlx_linalg_svd(array.ctx, stream.ctx)!
    defer { mlx_free(mlx_arrays) }

    let arrays = mlx_vector_array_values(mlx_arrays)
    return (arrays[0], arrays[1], arrays[2])
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
public func inv(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_linalg_inv(array.ctx, stream.ctx))
}
