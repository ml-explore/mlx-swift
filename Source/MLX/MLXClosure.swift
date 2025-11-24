import Cmlx
import Foundation

// MARK: - MLXClosure wrapper
public struct MLXClosure {
    public var c: mlx_closure
    public init(_ c: mlx_closure) { self.c = c }

    /// Apply closure to a Swift array of MLXArray and return output arrays
    public static func apply(_ closure: MLXClosure, _ inputs: [MLXArray]) throws -> [MLXArray] {
        // Create a C mlx_vector_array and append input arrays
        let inputVec = mlx_vector_array_new()
        for arr in inputs {
            mlx_vector_array_append_value(inputVec, arr.ctx)
        }

        // Prepare output vector
        var outputVec = mlx_vector_array_new()

        // Call C API
        let status = mlx_closure_apply(&outputVec, closure.c, inputVec)
        guard status == 0 else {
            throw MLXError.caught("Failed to apply MLXClosure: \(status)")
        }

        // Convert output mlx_vector_array to [MLXArray]
        let count = mlx_vector_array_size(outputVec)
        var result: [MLXArray] = []
        for i in 0 ..< count {
            var arr = mlx_array()
            mlx_vector_array_get(&arr, outputVec, i)
            result.append(MLXArray(arr))
        }

        // Free the C vectors if needed
        mlx_vector_array_free(inputVec)
        mlx_vector_array_free(outputVec)

        return result
    }
}
