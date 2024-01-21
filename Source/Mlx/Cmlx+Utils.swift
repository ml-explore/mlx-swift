import Foundation
import Cmlx

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
