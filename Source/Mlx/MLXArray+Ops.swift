import Foundation
import Cmlx

// TODO: this probably need to move to an Ops.swift along with other free functions

public func broadcast(arrays: [MLXArray], stream: StreamOrDevice = .default) -> [MLXArray] {
    let result = mlx_broadcast_arrays(arrays.map { $0.ctx }, arrays.count, stream.ctx)
    defer { free(result.arrays) }
    
    return (0 ..< result.size).map { MLXArray(result.arrays[$0]!) }
}

// MARK: - Operations

infix operator ** : BitwiseShiftPrecedence
infix operator *** : MultiplicationPrecedence
infix operator /% : MultiplicationPrecedence

extension MLXArray {
    
    public static func +(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_add(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func -(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_subtract(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static prefix func -(lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_negative(lhs.ctx, s.ctx))
    }

    public static func *(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_multiply(lhs.ctx, rhs.ctx, s.ctx))
    }
    
    public static func **(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_power(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func ***(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_matmul(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func /(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_divide(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func /%(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_floor_divide(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func %(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_remainder(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func ==(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func <=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_less_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func >=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_greater_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func !=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_not_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func <(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_less(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func >(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_less(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func &&(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_logical_and(lhs.ctx, rhs.ctx, s.ctx))
    }

    public static func ||(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_logical_or(lhs.ctx, rhs.ctx, s.ctx))
    }

}

// MARK: - Functions

extension MLXArray {
    
    public func all(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        return MLXArray(mlx_all_axes(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
    
    public func all(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_axis(ctx, axis.int32, keepDims, stream.ctx))
    }

    public func all(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_all(ctx, keepDims, stream.ctx))
    }

    /// Return `true` if all contents are `true` (in the mlx-sense where true is != 0).
    ///
    /// Equivalent to:
    ///
    /// ```
    /// let allTrue = array.all().item(Bool.self)
    /// ```
    public func allTrue(stream: StreamOrDevice = .default) -> Bool {
        let all = mlx_all_all(ctx, false, stream.ctx)!
        let bool = mlx_array_item_bool(all)
        mlx_free(all)
        return bool
    }

    public func take(_ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take(ctx, indices.ctx, axis.int32, stream.ctx))
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
