// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

// see Transforms+Variants for generated grad() functions

func buildValueAndGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: [Int]) -> (
    [MLXArray]
) -> [MLXArray] {

    let closure = new_mlx_closure(f)

    // a container that will hold the mlx_closure_value_and_grad and mlx_closure
    // and will clean itself up when freed
    class ValueAndGradContainer {

        let valueAndGrad: mlx_closure_value_and_grad

        init(_ valueAndGrad: mlx_closure_value_and_grad) {
            self.valueAndGrad = valueAndGrad
        }

        deinit {
            mlx_free(valueAndGrad)
        }

        func callAsFunction(_ arrays: [MLXArray]) -> [MLXArray] {
            let input_vector = new_mlx_vector_array(arrays)
            defer { mlx_free(input_vector) }

            let vector_pair = mlx_closure_value_and_grad_apply(valueAndGrad, input_vector)!
            defer { mlx_free(vector_pair) }

            // on the c++ side the result is std::pair<array, std::vector<array>> where
            // the second is the gradients with respect to the first
            let output_vector = mlx_vector_vector_array_get(vector_pair, 1)!
            defer { mlx_free((output_vector)) }

            return mlx_vector_array_values(output_vector)
        }
    }

    let valueAndGrad = mlx_value_and_grad(closure, argumentNumbers.asInt32, argumentNumbers.count)!
    mlx_free(closure)

    let container = ValueAndGradContainer(valueAndGrad)

    return { [container] (arrays: [MLXArray]) in
        container(arrays)
    }
}

private func new_mlx_closure(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {

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
