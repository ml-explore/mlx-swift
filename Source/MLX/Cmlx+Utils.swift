// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

@inline(__always)
func mlx_free(_ ptr: OpaquePointer) {
    mlx_free(UnsafeMutableRawPointer(ptr))
}

@inline(__always)
func mlx_retain(_ ptr: OpaquePointer) {
    mlx_retain(UnsafeMutableRawPointer(ptr))
}

func mlx_describe(_ ptr: OpaquePointer) -> String? {
    let description = mlx_tostring(UnsafeMutableRawPointer(ptr))!
    defer { mlx_free(description) }
    return String(cString: mlx_string_data(description))
}

// return a +1 mlx_vector_array containing the given arrays
func new_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    let result = mlx_vector_array_new()!
    mlx_vector_array_add_arrays(result, arrays.map { $0.ctx }, arrays.count)
    return result
}

func mlx_vector_array_values(_ vector_array: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector_array))
        .map { index in
            // take a +1 on each array to transfer ownership
            let ctx = mlx_vector_array_get(vector_array, index)!
            mlx_retain(ctx)
            return MLXArray(ctx)
        }
}

func mlx_map_values(_ mlx_map: mlx_map_string_to_array) -> [String: MLXArray] {
    var result = [String: MLXArray]()

    let iterator = mlx_map_string_to_array_iterate(mlx_map)!
    defer { mlx_free(iterator) }

    while !mlx_map_string_to_array_iterator_end(iterator) {
        let mlx_key = mlx_map_string_to_array_iterator_key(iterator)!
        defer { mlx_free(mlx_key) }
        let key = String(cString: mlx_string_data(mlx_key))

        // note: transfer ownership
        let mlx_array_ctx = mlx_map_string_to_array_iterator_value(iterator)!
        let array = MLXArray(mlx_array_ctx)

        result[key] = array

        mlx_map_string_to_array_iterator_next(iterator)
    }

    return result
}

func new_mlx_map(_ dictionary: [String: MLXArray]) -> mlx_map_string_to_array {
    let mlx_map = mlx_map_string_to_array_new()!

    for (key, array) in dictionary {
        let mlx_key = mlx_string_new(key.cString(using: .utf8))!
        defer { mlx_free(mlx_key) }

        mlx_map_string_to_array_insert(mlx_map, mlx_key, array.ctx)
    }

    return mlx_map
}
