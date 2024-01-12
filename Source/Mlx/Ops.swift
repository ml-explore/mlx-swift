import Foundation
import Cmlx


/// Broadcast a vector of arrays against one another.
public func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let result = mlx_broadcast_arrays(arrays.map { $0.ctx }, arrays.count, stream.ctx)
    defer { free(result.arrays) }
    
    return (0 ..< result.size).map { MLXArray(result.arrays[$0]!) }
}

/// Element-wise square.
public func square(_ a: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_square(a.ctx, stream.ctx))
}
