// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// return a +1 mlx_vector_array containing the given arrays
func new_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    mlx_vector_array_new_data(arrays.map { $0.ctx }, arrays.count)
}

func mlx_vector_array_values(_ vector_array: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector_array))
        .map { index in
            // ctx is a +1 object, the array takes ownership
            var ctx = mlx_array_new()
            mlx_vector_array_get(&ctx, vector_array, index)
            return MLXArray(ctx)
        }
}

func mlx_map_array_values(_ mlx_map: mlx_map_string_to_array) -> [String: MLXArray] {
    var result = [String: MLXArray]()

    let iterator = mlx_map_string_to_array_iterator_new(mlx_map)
    defer { mlx_map_string_to_array_iterator_free(iterator) }

    var mlx_key: UnsafePointer<CChar>?
    var mlx_value = mlx_array_new()
    defer { mlx_array_free(mlx_value) }

    while mlx_map_string_to_array_iterator_next(&mlx_key, &mlx_value, iterator) == 0 {
        guard let mlx_key else { continue }

        let key = String(cString: mlx_key)
        let array = MLXArray(mlx_value)

        result[key] = array
    }

    return result
}

func mlx_map_string_values(_ mlx_map: mlx_map_string_to_string) -> [String: String] {
    var result = [String: String]()

    let iterator = mlx_map_string_to_string_iterator_new(mlx_map)
    defer { mlx_map_string_to_string_iterator_free(iterator) }

    var mlx_key: UnsafePointer<CChar>?
    var mlx_value: UnsafePointer<CChar>?

    while mlx_map_string_to_string_iterator_next(&mlx_key, &mlx_value, iterator) == 0 {
        guard let mlx_key, let mlx_value else { continue }

        let key = String(cString: mlx_key)
        let value = String(cString: mlx_value)

        result[key] = value
    }

    return result
}

func new_mlx_array_map(_ dictionary: [String: MLXArray]) -> mlx_map_string_to_array {
    let mlx_map = mlx_map_string_to_array_new()

    for (key, array) in dictionary {
        mlx_map_string_to_array_insert(mlx_map, key.cString(using: .utf8), array.ctx)
    }

    return mlx_map
}

func new_mlx_string_map(_ dictionary: [String: String]) -> mlx_map_string_to_string {
    let mlx_map = mlx_map_string_to_string_new()

    for (key, value) in dictionary {
        mlx_map_string_to_string_insert(
            mlx_map, key.cString(using: .utf8), value.cString(using: .utf8))
    }

    return mlx_map
}

// TODO
/*

func new_mlx_closure(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {

    // holds reference to `f()` as capture state for the mlx_closure
    class ClosureCaptureState {
        let f: ([MLXArray]) -> [MLXArray]

        init(_ f: @escaping ([MLXArray]) -> [MLXArray]) {
            self.f = f
        }
    }

    func free(ptr: UnsafeMutableRawPointer?) {
        Unmanaged<ClosureCaptureState>.fromOpaque(ptr!).release()
    }

    let payload = Unmanaged.passRetained(ClosureCaptureState(f)).toOpaque()

    // the C function that the mlx_closure will call -- this will convert
    // arguments & results and call the captured `f()`
    func trampoline(vector_array: mlx_vector_array?, payload: UnsafeMutableRawPointer?)
        -> mlx_vector_array?
    {
        let state = Unmanaged<ClosureCaptureState>.fromOpaque(payload!).takeUnretainedValue()

        let arrays = mlx_vector_array_values(vector_array!)
        let result = state.f(arrays)
        return new_mlx_vector_array(result)
    }

    return mlx_closure_new_with_payload(trampoline, payload, free)!
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
*/
