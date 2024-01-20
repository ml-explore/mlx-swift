import Foundation
import Cmlx

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:Lazy-Evaluation>
public func eval(_ arrays: MLXArray...) {
    mlx_eval(arrays.map { $0.ctx }, arrays.count)
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:Lazy-Evaluation>
public func eval(_ arrays: [MLXArray]) {
    mlx_eval(arrays.map { $0.ctx }, arrays.count)
}

