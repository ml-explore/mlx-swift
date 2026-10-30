// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// see Transforms+Variants for generated grad() functions

/// Inner build of the value and gradient function
/// - Parameters:
///   - valueAndGrad: closure to transform
///   - arrays: inputs
///   - expected: minimum number of values and gradients to return.  If an error
///     occurs, the results are padded with `mlxNone` so callers can index safely.
private func valueAndGradient(
    apply valueAndGrad: mlx_closure_value_and_grad, arrays: some Collection<MLXArray>,
    expected: (values: Int, gradients: Int)
) -> ([MLXArray], [MLXArray]) {
    let input_vector = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(input_vector) }

    var r0 = mlx_vector_array_new()
    var r1 = mlx_vector_array_new()

    _ = withEvalLock {
        mlx_closure_value_and_grad_apply(&r0, &r1, valueAndGrad, input_vector)
    }

    defer { mlx_vector_array_free(r0) }
    defer { mlx_vector_array_free(r1) }

    return (
        mlx_vector_array_values(r0, count: expected.values),
        mlx_vector_array_values(r1, count: expected.gradients)
    )
}

func buildGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: some Collection<Int>)
    -> (
        [MLXArray]
    ) -> [MLXArray]
{
    { (arrays: [MLXArray]) in
        var vag = mlx_closure_value_and_grad_new()

        let closure = new_mlx_closure(f)
        mlx_value_and_grad(&vag, closure, argumentNumbers.asInt32, argumentNumbers.count)
        mlx_closure_free(closure)

        defer { mlx_closure_value_and_grad_free(vag) }

        // unwrapResult() reads gradients[0], so ensure at least one
        let (_, gradients) = valueAndGradient(
            apply: vag, arrays: arrays,
            expected: (values: 0, gradients: max(1, argumentNumbers.count)))
        return gradients
    }
}

func buildValueAndGradient(
    _ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: some Collection<Int>
) -> (
    [MLXArray]
) -> ([MLXArray], [MLXArray]) {
    { (arrays: [MLXArray]) in
        var vag = mlx_closure_value_and_grad_new()

        let closure = new_mlx_closure(f)
        mlx_value_and_grad(&vag, closure, argumentNumbers.asInt32, argumentNumbers.count)
        mlx_closure_free(closure)

        defer { mlx_closure_value_and_grad_free(vag) }

        return valueAndGradient(
            apply: vag, arrays: arrays,
            expected: (values: 1, gradients: argumentNumbers.count))
    }
}

func buildValueAndGradient<T>(
    _ f: @escaping (NestedDictionary<String, MLXArray>, T) -> [MLXArray]
) -> (NestedDictionary<String, MLXArray>, T) -> (
    [MLXArray], NestedDictionary<String, MLXArray>
) {
    {
        (parameters: NestedDictionary<String, MLXArray>, arrays: T) -> (
            [MLXArray], NestedDictionary<String, MLXArray>
        ) in

        // capture the state so that we can unflatten
        let flattenedParameters = parameters.flattened()
        let flattenedKeys = flattenedParameters.map { $0.0 }
        let flattenedArrays = flattenedParameters.map { $0.1 }

        // function to unflatten back into the NestedDictionary
        func unflattened(_ arrays: [MLXArray]) -> NestedDictionary<String, MLXArray> {
            let tuples = zip(flattenedKeys, arrays).map { ($0.0, $0.1) }
            return NestedDictionary.unflattened(tuples)
        }

        // this goes in the closure and is wrapped by mlx_value_and_grad
        //
        // Note: we pass the flattened array through the grad but
        // we capture the extra arrays used as arguments (matching
        // the python implementation).
        //
        // Potentially this could pass all the values and use the
        // arg indexes to indicate which ones to grad (it should work
        // as is)
        func inner(flattenedArrays: [MLXArray]) -> [MLXArray] {
            let parameters = unflattened(flattenedArrays)
            return f(parameters, arrays)
        }

        var vag = mlx_closure_value_and_grad_new()

        let closure = new_mlx_closure(inner)
        mlx_value_and_grad(
            &vag, closure, Array(Int32(0) ..< Int32(flattenedArrays.count)), flattenedArrays.count)
        mlx_closure_free(closure)

        defer { mlx_closure_value_and_grad_free(vag) }

        let (values, flatGradients) = valueAndGradient(
            apply: vag, arrays: flattenedArrays,
            expected: (values: 1, gradients: flattenedArrays.count))
        let gradients = unflattened(flatGradients)

        return (values, gradients)
    }
}
