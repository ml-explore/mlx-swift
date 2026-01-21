// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// see Transforms+Variants for generated grad() functions

private func valueAndGradient(
    apply valueAndGrad: mlx_closure_value_and_grad, arrays: some Collection<MLXArray>
)
    -> ([MLXArray], [MLXArray])
{
    let input_vector = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(input_vector) }

    var r0 = mlx_vector_array_new()
    var r1 = mlx_vector_array_new()

    _ = evalLock.withLock {
        mlx_closure_value_and_grad_apply(&r0, &r1, valueAndGrad, input_vector)
    }

    defer { mlx_vector_array_free(r0) }
    defer { mlx_vector_array_free(r1) }

    return (mlx_vector_array_values(r0), mlx_vector_array_values(r1))
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

        return valueAndGradient(apply: vag, arrays: arrays).1
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

        return valueAndGradient(apply: vag, arrays: arrays)
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

        let (values, flatGradients) = valueAndGradient(apply: vag, arrays: flattenedArrays)
        let gradients = unflattened(flatGradients)

        return (values, gradients)
    }
}
