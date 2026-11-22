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

    _ = withEvalLock {
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

    _ = withEvalLock {
        mlx_vjp(&r0, &r1, closure, primals_mlx, cotangents_mlx)
    }

    mlx_closure_free(closure)

    defer { mlx_vector_array_free(r0) }
    defer { mlx_vector_array_free(r1) }

    // outputs match the cotangents in number, vjps match the primals
    return (
        mlx_vector_array_values(r0, count: cotangents.count),
        mlx_vector_array_values(r1, count: primals.count)
    )
}

/// Holds the `mlx_closure` produced by `mlx_checkpoint` for the lifetime of the
/// Swift function returned by ``checkpoint(_:)-(([MLXArray])->[MLXArray])``.
final class _CheckpointState: @unchecked Sendable {

    private let checkpointed: mlx_closure

    init(_ f: @escaping ([MLXArray]) -> [MLXArray]) {
        let inner = new_mlx_closure(f)
        defer { mlx_closure_free(inner) }

        // mlx_checkpoint copies the underlying std::function, so `inner` can be
        // released once the checkpointed closure has been built.
        //
        // On failure mlx_error has already been dispatched (fatalError, or the
        // ErrorBox inside withError) and `result` stays an empty closure, which
        // deinit can still free.
        var result = mlx_closure_new()
        _ = withEvalLock {
            mlx_checkpoint(&result, inner)
        }
        self.checkpointed = result
    }

    deinit {
        mlx_closure_free(checkpointed)
    }

    func call(_ inputs: [MLXArray], expected: Int) -> [MLXArray] {
        let inputVector = new_mlx_vector_array(inputs)
        defer { mlx_vector_array_free(inputVector) }

        var resultVector = mlx_vector_array_new()
        defer { mlx_vector_array_free(resultVector) }

        // mlx_error was already dispatched on failure:
        //   • outside withError — fatalError was called; we won't reach here
        //   • inside withError  — error is stored in the ErrorBox; the result is
        //                         padded with `mlxNone` so callers can index
        //                         safely and withError can throw
        _ = withEvalLock {
            mlx_closure_apply(&resultVector, checkpointed, inputVector)
        }

        return mlx_vector_array_values(resultVector, count: expected)
    }
}

/// Transform `f` into a function that performs gradient checkpointing with
/// respect to its inputs.
///
/// The returned function produces the same output as `f`.  When a gradient is
/// taken through it (``grad(_:)-r8dv``, ``valueAndGrad(_:)``,
/// ``vjp(_:primals:cotangents:)``, ...) the intermediate arrays computed inside
/// `f` are not kept alive for the backward pass; instead `f` is evaluated again
/// during the backward pass to recompute them.
///
/// Use this to reduce memory use for gradient computations at the expense of
/// increased computation, e.g. by checkpointing each transformer block of a
/// model being fine tuned.
///
/// ```swift
/// func block(_ inputs: [MLXArray]) -> [MLXArray] {
///     [square(inputs[0] * inputs[1] + inputs[0] + inputs[1])]
/// }
///
/// let checkpointed = checkpoint(block)
///
/// // same result as block(), but the backward pass recomputes block()
/// let (value, gradients) = vjp(checkpointed, primals: [x, y], cotangents: [MLXArray(1.0)])
/// ```
///
/// - Parameters:
///   - f: function which takes an array of ``MLXArray`` and returns an array of ``MLXArray``
/// - Returns: a function that produces the same output as `f` and recomputes
///   intermediate values during gradient computation
///
/// ### See Also
/// - ``checkpoint(_:)-((MLXArray)->MLXArray)``
/// - ``grad(_:)-r8dv``
/// - ``vjp(_:primals:cotangents:)``
public func checkpoint(
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    let state = _CheckpointState(f)

    return { arrays in
        state.call(arrays, expected: 0)
    }
}

/// Overload of ``checkpoint(_:)-(([MLXArray])->[MLXArray])`` that takes a
/// single ``MLXArray`` and returns a single ``MLXArray``.
public func checkpoint(
    _ f: @escaping (MLXArray) -> MLXArray
) -> (MLXArray) -> MLXArray {
    let state = _CheckpointState { arrays in
        [f(arrays[0])]
    }

    return { array in
        state.call([array], expected: 1)[0]
    }
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
