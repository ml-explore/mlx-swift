// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// see Transforms+Variants for generated grad() functions

private func valueAndGradient(valueAndGrad: mlx_closure_value_and_grad, arrays: [MLXArray]) -> ([MLXArray], [MLXArray]) {
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
        let valueAndGrad = mlx_value_and_grad(closure, argumentNumbers.asInt32, argumentNumbers.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)
        
        return valueAndGradient(valueAndGrad: valueAndGrad, arrays: arrays).1
    }
}

func buildValueAndGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int]) -> (
    [MLXArray]
) -> ([MLXArray], [MLXArray]) {
    { (arrays: [MLXArray]) in
        let closure = new_mlx_closure(f)
        let valueAndGrad = mlx_value_and_grad(closure, argumentNumbers.asInt32, argumentNumbers.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)
        
        return valueAndGradient(valueAndGrad: valueAndGrad, arrays: arrays)
    }
}

func buildValueAndGradient(_ f: @escaping (NestedDictionary<String, MLXArray>, [MLXArray]) -> [MLXArray], argumentNumbers: [Int]) -> (NestedDictionary<String, MLXArray>, [MLXArray]) -> ([MLXArray], [MLXArray]) {
    { (parameters: NestedDictionary<String, MLXArray>, arrays: [MLXArray]) -> ([MLXArray], [MLXArray]) in
        
        struct ParametersState {
            let keys: [String]
            
            func unflatten(_ arrays: ArraySlice<MLXArray>) -> NestedDictionary<String, MLXArray> {
                precondition(keys.count == arrays.count)
                let tuples = zip(keys, arrays).map { ($0.0, $0.1) }
                return NestedDictionary.unflattened(tuples)
            }
        }
        
        // capture the state so that we can unflatten
        let flattenedParameters = parameters.flattened()
        let parametersState = ParametersState(keys: flattenedParameters.map { $0.0 })
        
        // combine all the arrays
        var arrays = flattenedParameters.map { $0.1 } + arrays
                
        // a funcation that will get ParametersState back reconstitute the parameters
        func inner(arrays: [MLXArray]) -> [MLXArray] {
            let parameters = parametersState.unflatten(arrays[0 ..< parametersState.keys.count])
            let arrays = Array(arrays.dropFirst(parametersState.keys.count))
            
            return f(parameters, arrays)
        }
        
        let closure = new_mlx_closure(inner)
        let valueAndGrad = mlx_value_and_grad(closure, argumentNumbers.asInt32, argumentNumbers.count)!
        defer { mlx_free(valueAndGrad) }
        mlx_free(closure)
        
        return valueAndGradient(valueAndGrad: valueAndGrad, arrays: arrays)
    }
}
