import Foundation
import Cmlx

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:Lazy-Evaluation>
public func eval(_ arrays: MLXArray...) {
    let vector_array = new_mlx_vector_array(arrays)
    mlx_eval(vector_array)
    mlx_free(vector_array)
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:Lazy-Evaluation>
public func eval(_ arrays: [MLXArray]) {
    let vector_array = new_mlx_vector_array(arrays)
    mlx_eval(vector_array)
    mlx_free(vector_array)
}
