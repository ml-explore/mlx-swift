// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Compute the Jacobian-vector product.
///
/// This computes the product of the Jacobian of a function `f` evaluated
/// at `primals` with the `tangents`.
///
/// - Parameters:
///   - f: function which takes an array of ``MLXArray`` and returns an array of ``MLXArray``
///   - primals: array of ``MLXArray`` at which to evaluate the Jacobian
///   - tangents: array of ``MLXArray`` which are the "vector" in the Jacobian-vector product.  The `tangents`
///     should be the same in number, shape and type as the inputs of `f`, e.g. the `primals`
/// - Returns: array of the Jacobian-vector products which is the same in number, shape and type of the outputs of `f`
public func jvp(
    _ f: @escaping ([MLXArray]) -> [MLXArray], primals: some Collection<MLXArray>,
    tangents: some Collection<MLXArray>
) -> ([MLXArray], [MLXArray]) {
    let primals_mlx = new_mlx_vector_array(primals)
    defer { mlx_vector_array_free(primals_mlx) }
    let tangents_mlx = new_mlx_vector_array(tangents)
    defer { mlx_vector_array_free(tangents_mlx) }

    var r0 = mlx_vector_array_new()
    var r1 = mlx_vector_array_new()

    let closure = new_mlx_closure(f)

    _ = evalLock.withLock {
        mlx_jvp(&r0, &r1, closure, primals_mlx, tangents_mlx)
    }

    mlx_closure_free(closure)

    defer { mlx_vector_array_free(r0) }
    defer { mlx_vector_array_free(r1) }

    return (mlx_vector_array_values(r0), mlx_vector_array_values(r1))
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a
/// function `f` evaluated at `primals`.
///
/// - Parameters:
///   - f: function which takes an array of ``MLXArray`` and returns an array of ``MLXArray``
///   - primals: array of ``MLXArray`` at which to evaluate the Jacobian
///   - cotangents: array of ``MLXArray`` which are the "vector" in the vector-Jacobian product.  The `cotangents`
///     should be the same in number, shape and type as the outputs of `f`
/// - Returns: array of the vector-Jacobian products which is the same in number, shape and type of the outputs of `f`
public func vjp(
    _ f: @escaping ([MLXArray]) -> [MLXArray], primals: some Collection<MLXArray>,
    cotangents: some Collection<MLXArray>
) -> ([MLXArray], [MLXArray]) {
    let primals_mlx = new_mlx_vector_array(primals)
    defer { mlx_vector_array_free(primals_mlx) }
    let cotangents_mlx = new_mlx_vector_array(cotangents)
    defer { mlx_vector_array_free(cotangents_mlx) }

    var r0 = mlx_vector_array_new()
    var r1 = mlx_vector_array_new()

    let closure = new_mlx_closure(f)

    _ = evalLock.withLock {
        mlx_vjp(&r0, &r1, closure, primals_mlx, cotangents_mlx)
    }

    mlx_closure_free(closure)

    defer { mlx_vector_array_free(r0) }
    defer { mlx_vector_array_free(r1) }

    return (mlx_vector_array_values(r0), mlx_vector_array_values(r1))
}

/// Returns a function that computes the gradient and result of `f`, computing the gradient with respect to the ``NestedDictionary``.
///
/// Note that this allows any parameters `<T>` s they will not be part of the gradient.
public func valueAndGrad<T>(
    _ f: @escaping (NestedDictionary<String, MLXArray>, T) -> [MLXArray]
) -> (NestedDictionary<String, MLXArray>, T) -> (
    [MLXArray], NestedDictionary<String, MLXArray>
) {
    buildValueAndGradient(f)
}
