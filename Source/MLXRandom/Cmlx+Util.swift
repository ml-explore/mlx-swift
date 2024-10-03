// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX

@inline(__always)
func mlx_free(_ ptr: OpaquePointer) {
    mlx_free(UnsafeMutableRawPointer(ptr))
}

// return a +1 mlx_vector_array containing the given arrays
func new_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    let result = mlx_vector_array_new()!
    mlx_vector_array_add_data(result, arrays.map { $0.ctx }, arrays.count)
    return result
}

func mlx_vector_array_values(_ vector_array: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector_array))
        .map { index in
            // ctx is a +1 object, the array takes ownership
            let ctx = mlx_vector_array_get(vector_array, index)!
            return MLXArray(ctx)
        }
}

func mlx_tuple_values(_ tuple: mlx_tuple_array_array) -> (MLXArray, MLXArray) {
    let a = mlx_tuple_array_array_get_0(tuple)!
    let b = mlx_tuple_array_array_get_1(tuple)!
    return (MLXArray(a), MLXArray(b))
}

func mlx_tuple_vectors(_ tuple: mlx_tuple_vector_array_vector_array) -> ([MLXArray], [MLXArray]) {
    let a = mlx_tuple_vector_array_vector_array_get_0(tuple)!
    defer { mlx_free(a) }
    let b = mlx_tuple_vector_array_vector_array_get_1(tuple)!
    defer { mlx_free(b) }
    return (mlx_vector_array_values(a), mlx_vector_array_values(b))
}

func mlx_tuple_values(_ tuple: mlx_tuple_array_array_array) -> (MLXArray, MLXArray, MLXArray) {
    let a = mlx_tuple_array_array_array_get_0(tuple)!
    let b = mlx_tuple_array_array_array_get_1(tuple)!
    let c = mlx_tuple_array_array_array_get_2(tuple)!
    return (MLXArray(a), MLXArray(b), MLXArray(c))
}
