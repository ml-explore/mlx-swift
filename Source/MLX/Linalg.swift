// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

public enum MLXLinalg {

    /// Types of norms available.
    ///
    /// ### See Also
    /// - ``norm(_:ord:axes:keepDims:stream:)``
    /// - ``norm(_:ord:axes:keepDims:stream:)-8zljj``
    /// - ``MLXLinalg``
    public enum NormKind: String, Sendable {
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
    /// - ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: NormKind? = nil, axes: some Collection<Int>, keepDims: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let ord {
            mlx_linalg_norm_matrix(
                &result, array.ctx, ord.rawValue, axes.asInt32, axes.count, keepDims, stream.ctx)
        } else {
            mlx_linalg_norm_l2(&result, array.ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        }
        return MLXArray(result)
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
    /// - ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: Double, axes: some Collection<Int>, keepDims: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_linalg_norm(&result, array.ctx, ord, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Matrix or vector norm.
    ///
    /// - ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: NormKind? = nil, axis: Int, keepDims: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let ord {
            mlx_linalg_norm_matrix(
                &result, array.ctx, ord.rawValue, [axis].asInt32, 1, keepDims, stream.ctx)
            return MLXArray(result)
        } else {
            mlx_linalg_norm_l2(&result, array.ctx, [axis].asInt32, 1, keepDims, stream.ctx)
            return MLXArray(result)
        }
    }

    /// Matrix or vector norm.
    ///
    /// See ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: Double, axis: Int, keepDims: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_linalg_norm(&result, array.ctx, ord, [axis].asInt32, 1, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Matrix or vector norm.
    ///
    /// - ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: NormKind? = nil, axis: IntOrArray? = nil,
        keepDims: Bool = false, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        if let ord {
            mlx_linalg_norm_matrix(
                &result, array.ctx, ord.rawValue, axis?.asInt32Array, axis?.count ?? 0, keepDims,
                stream.ctx)
            return MLXArray(result)
        } else {
            mlx_linalg_norm_l2(
                &result, array.ctx, axis?.asInt32Array, axis?.count ?? 0, keepDims, stream.ctx)
            return MLXArray(result)
        }
    }

    /// Matrix or vector norm.
    ///
    /// See ``norm(_:ord:axes:keepDims:stream:)``
    public static func norm(
        _ array: MLXArray, ord: Double, axis: IntOrArray? = nil,
        keepDims: Bool = false, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_linalg_norm(
            &result, array.ctx, ord, axis?.asInt32Array, axis?.count ?? 0, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// The QR factorization of the input matrix.
    ///
    /// This function supports arrays with at least 2 dimensions. The matrices
    /// which are factorized are assumed to be in the last two dimensions of
    /// the input.
    ///
    /// - Returns: the `Q` and `R` matrices
    public static func qr(_ array: MLXArray, stream: StreamOrDevice = .default) -> (
        MLXArray, MLXArray
    ) {
        var r0 = mlx_array_new()
        var r1 = mlx_array_new()

        mlx_linalg_qr(&r0, &r1, array.ctx, stream.ctx)

        return (MLXArray(r0), MLXArray(r1))
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
    @_disfavoredOverload
    public static func svd(_ array: MLXArray, stream: StreamOrDevice = .default) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        var vec = mlx_vector_array_new()
        mlx_linalg_svd(&vec, array.ctx, true, stream.ctx)
        defer { mlx_vector_array_free(vec) }

        let arrays = mlx_vector_array_values(vec)
        return (arrays[0], arrays[1], arrays[2])
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
    /// - Returns: The `S` matrix
    public static func svd(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var vec = mlx_vector_array_new()
        mlx_linalg_svd(&vec, array.ctx, false, stream.ctx)
        defer { mlx_vector_array_free(vec) }

        let arrays = mlx_vector_array_values(vec)
        return arrays[0]
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
    public static func inv(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_linalg_inv(&result, array.ctx, stream.ctx)
        return MLXArray(result)
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
    public static func triInv(
        _ array: MLXArray, upper: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_linalg_tri_inv(&result, array.ctx, upper, stream.ctx)
        return MLXArray(result)
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
    public static func cholesky(
        _ array: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_linalg_cholesky(&result, array.ctx, upper, stream.ctx)
        return MLXArray(result)
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
    public static func choleskyInv(
        _ array: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_linalg_cholesky_inv(&result, array.ctx, upper, stream.ctx)
        return MLXArray(result)
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
    public static func cross(
        _ a: MLXArray, _ b: MLXArray, axis: Int = -1, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_linalg_cross(&result, a.ctx, b.ctx, axis.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Compute the LU factorization of the given matrix `A`.
    ///
    /// Note, unlike the default behavior of `scipy.linalg.lu`, the pivots
    /// are indices. To reconstruct the input use `L[P] @ U` for 2
    /// dimensions or `takeAlong(L, P[.ellipsis, .newAxis], axis: -2) @ U`
    /// for more than 2 dimensions.
    ///
    /// To construct the full permuation matrix do:
    ///
    ///   P = putAlong(zeros(L), p[.ellipsis, .newAxis], MLXArray(1.0), axis: -1)
    ///
    /// -Parameters:
    ///   - a: input array.
    ///   - stream: stream or device
    public static func lu(_ a: MLXArray, stream: StreamOrDevice = .default)
        -> (MLXArray, MLXArray, MLXArray)
    {
        var vec = mlx_vector_array_new()
        mlx_linalg_lu(&vec, a.ctx, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        let arrays = mlx_vector_array_values(vec)
        return (arrays[0], arrays[1], arrays[2])
    }

    /// Computes a compact representation of the LU factorization.
    ///
    /// -Parameters:
    ///   - a: input array.
    ///   - stream: stream or device
    public static func lu_factor(_ a: MLXArray, stream: StreamOrDevice = .default)
        -> (MLXArray, MLXArray)
    {
        var res_0 = mlx_array_new()
        var res_1 = mlx_array_new()
        mlx_linalg_lu_factor(&res_0, &res_1, a.ctx, stream.ctx)
        return (MLXArray(res_0), MLXArray(res_1))
    }

    /// Compute the solution to a system of linear equations `AX = B`.
    ///
    /// -Parameters:
    ///   - a: input array.
    ///   - b: input array.
    ///   - stream: stream or device
    public static func solve(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_linalg_solve(&result, a.ctx, b.ctx, stream.ctx)
        return MLXArray(result)
    }

    ///Computes the solution of a triangular system of linear equations `AX = B`.
    ///
    /// -Parameters:
    ///   - a: input array.
    ///   - b: input array.
    ///   - upper: Whether the array is upper or lower triangular
    ///   - stream: stream or device
    public static func solveTriangular(
        _ a: MLXArray, _ b: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_linalg_solve_triangular(&result, a.ctx, b.ctx, upper, stream.ctx)
        return MLXArray(result)
    }

}  // MLXLinalg

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
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axes: some Collection<Int>,
    keepDims: Bool = false,
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
///   - axis: axes that hold 2d matrices
///   - keepDims: if `true` the axes which are normed over are left in the result as dimensions with size one
///   - stream: stream to evaluate on
/// - Returns: output containing the norm(s)
///
/// ### See Also
/// - ``norm(_:ord:axes:keepDims:stream:)``
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)``
public func norm(
    _ array: MLXArray, ord: Double, axis: Int, keepDims: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// - ``norm(_:ord:axes:keepDims:stream:)``
public func norm(
    _ array: MLXArray, ord: MLXLinalg.NormKind? = nil, axis: IntOrArray? = nil,
    keepDims: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXLinalg.norm(array, ord: ord, axis: axis, keepDims: keepDims, stream: stream)
}

/// Matrix or vector norm.
///
/// See ``norm(_:ord:axes:keepDims:stream:)``
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
public func cross(_ a: MLXArray, _ b: MLXArray, axis: Int = -1, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXLinalg.cross(a, b, axis: axis, stream: stream)
}

/// Compute the LU factorization of the given matrix `A`.
///
/// Note, unlike the default behavior of `scipy.linalg.lu`, the pivots
/// are indices. To reconstruct the input use `L[P] @ U` for 2
/// dimensions or `takeAlong(L, P[.ellipsis, .newAxis], axis: -2) @ U`
/// for more than 2 dimensions.
///
/// To construct the full permuation matrix do:
///
///   P = putAlong(zeros(L), p[.ellipsis, .newAxis], MLXArray(1.0), axis: -1)
///
/// -Parameters:
///   - a: input array.
///   - stream: stream or device
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
public func solveTriangular(
    _ a: MLXArray, _ b: MLXArray, upper: Bool = false, stream: StreamOrDevice = .default
)
    -> MLXArray
{
    return MLXLinalg.solveTriangular(a, b, upper: upper, stream: stream)
}
