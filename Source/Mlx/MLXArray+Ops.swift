import Foundation
import Cmlx

// TODO: this probably need to move to an Ops.swift along with other free functions

public func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let result = mlx_broadcast_arrays(arrays.map { $0.ctx }, arrays.count, stream.ctx)
    defer { free(result.arrays) }
    
    return (0 ..< result.size).map { MLXArray(result.arrays[$0]!) }
}

extension MLXArray {
    
    public func take(_ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        return MLXArray(mlx_take(ctx, indices.ctx, axis.int32, stream.ctx))
    }
    
    public func take(_ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take_all(ctx, indices.ctx, stream.ctx))
    }
    
    public func reshape(_ newShape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reshape(ctx, newShape.asInt32, newShape.count, stream.ctx))
    }
    
    public func reshape(_ newShape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reshape(ctx, newShape, newShape.count, stream.ctx))
    }
    
    public func broadcast(to shape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_broadcast_to(ctx, shape.asInt32, shape.count, stream.ctx))
    }
    
    public func broadcast(to shape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_broadcast_to(ctx, shape, shape.count, stream.ctx))
    }
    
    public func scatter(indices: [MLXArray], updates: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_scatter(ctx, indices.map { $0.ctx }, indices.count, updates.ctx, axes.asInt32, axes.count, stream.ctx))
    }
    
    public func scatter(indices: [MLXArray], updates: MLXArray, axes: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_scatter(ctx, indices.map { $0.ctx }, indices.count, updates.ctx, axes, axes.count, stream.ctx))
    }
    
    public func allClose(_ other: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, stream: StreamOrDevice = .default) -> MLXArray {
        return MLXArray(mlx_allclose(self.ctx, other.ctx, rtol, atol, stream.ctx))
    }
}
