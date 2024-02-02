// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// see Transforms+Variants for generated grad() functions

private func valueAndGradient(apply valueAndGrad: mlx_closure_value_and_grad, arrays: [MLXArray])
    -> ([MLXArray], [MLXArray])
{
    let input_vector = new_mlx_vector_array(arrays)
    defer { mlx_free(input_vector) }

    let vector_pair = mlx_closure_value_and_grad_apply(valueAndGrad, input_vector)!
    defer { mlx_free(vector_pair) }

    let values = mlx_vector_vector_array_get(vector_pair, 0)!
    defer { mlx_free((values)) }

    let gradient = mlx_vector_vector_array_get(vector_pair, 1)!
    defer { mlx_free((gradient)) }

    return (mlx_vector_array_values(values), mlx_vector_array_values(gradient))
}

func buildGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int]) -> (
    [MLXArray]
) -> [MLXArray] {
    { (arrays: [MLXArray]) in
        let closure = new_mlx_closure(f)
        let valueAndGrad = mlx_value_and_grad(
            closure, argumentNumbers.asInt32, argumentNumbers.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)

        return valueAndGradient(apply: valueAndGrad, arrays: arrays).1
    }
}

func buildValueAndGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int]) -> (
    [MLXArray]
) -> ([MLXArray], [MLXArray]) {
    { (arrays: [MLXArray]) in
        let closure = new_mlx_closure(f)
        let valueAndGrad = mlx_value_and_grad(
            closure, argumentNumbers.asInt32, argumentNumbers.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)

        return valueAndGradient(apply: valueAndGrad, arrays: arrays)
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
        // Potentially this could pass all the the values and use the
        // arg indexes to indicate which ones to grad (it should work
        // as is)
        func inner(flattenedArrays: [MLXArray]) -> [MLXArray] {
            let parameters = unflattened(flattenedArrays)
            return f(parameters, arrays)
        }

        let closure = new_mlx_closure(inner)
        let valueAndGrad = mlx_value_and_grad(
            closure, Array(Int32(0) ..< Int32(flattenedArrays.count)),
            flattenedArrays.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)

        let (values, flatGradients) = valueAndGradient(apply: valueAndGrad, arrays: flattenedArrays)
        let gradients = unflattened(flatGradients)

        return (values, gradients)
    }
}
