import Foundation
import Cmlx

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

// MARK: - Internal Functions

extension MLXArray {
    
    func broadcast(to shape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_broadcast_to(ctx, shape, shape.count, stream.ctx))
    }
        
    func scatter(indices: [MLXArray], updates: MLXArray, axes: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_scatter(ctx, indices.map { $0.ctx }, indices.count, updates.ctx, axes, axes.count, stream.ctx))
    }

    // varaiant with [Int32] argument
    func reshape(_ newShape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reshape(ctx, newShape, newShape.count, stream.ctx))
    }

}


// MARK: - Public Functions

extension MLXArray {
    
    /// An `and` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// // will produce a scalar MLXArray with false -- not all of the values are non-zero
    /// let all = array.all()
    ///
    /// // produces an MLXArray([false, true, true, true]) -- the first row has a zero
    /// let allRows = array.all(axes: [0])
    ///
    /// // equivalent
    /// let allRows2 = array.all(axis: 0)
    /// ```
    public func all(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        return MLXArray(mlx_all_axes(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
    
    /// An `and` reduction over the given axes.
    ///
    /// See ``all(axes:keepDims:stream:)``
    public func all(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_axis(ctx, axis.int32, keepDims, stream.ctx))
    }

    /// An `and` reduction over the given axes.
    ///
    /// See ``all(axes:keepDims:stream:)``
    public func all(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_all(ctx, keepDims, stream.ctx))
    }
    
    /// Approximate comparison of two arrays.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    /// Note unlike ``arrayEqual(_:equalNAN:stream:)``, this function supports numpy-style broadcasting.
    public func allClose(_ other: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_allclose(self.ctx, other.ctx, rtol, atol, stream.ctx))
    }
    
    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `True` if and only if the arrays
    /// have the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// ```
    /// let a1 = MLXArray([0, 1, 2, 3])
    /// let a2 = MLXArray([0.0, 1.0, 2.0, 3.0])
    ///
    /// print(a1.arrayEqual(a2))
    /// // prints the scalar array `true`
    /// ```
    ///
    /// See also ``allClose(_:rtol:atol:stream:)``
    public func arrayEqual(_ other: MLXArray, equalNAN: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_array_equal(ctx, other.ctx, equalNAN, stream.ctx))
    }

    /// Return `true` if all contents are `true` (in the mlx-sense where true is != 0).
    ///
    /// Equivalent to:
    ///
    /// ```
    /// let allTrue = array.all().item(Bool.self)
    /// ```
    ///
    /// Use this as:
    ///
    /// ```
    /// if (a < b || a.allClose(c)).allTrue() {
    ///     ...
    /// }
    /// ```
    public func allTrue(stream: StreamOrDevice = .default) -> Bool {
        let all = mlx_all_all(ctx, false, stream.ctx)!
        let bool = mlx_array_item_bool(all)
        mlx_free(all)
        return bool
    }
    
    /// Reshape an array while preserving the size.
    public func reshape(_ newShape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reshape(ctx, newShape.asInt32, newShape.count, stream.ctx))
    }
    
    /// Element-wise square.
    public func square(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_square(ctx, stream.ctx))
    }

    /// Element-wise square root
    public func sqrt(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sqrt(ctx, stream.ctx))
    }

    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// /// produces a [3, 2] result with the index 0 and 2 columns from the array
    /// let col = array.take([0, 2], axis: 1)
    /// ```
    ///
    /// See also ``take(_:stream:)``
    public func take(_ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take(ctx, indices.ctx, axis.int32, stream.ctx))
    }
    
    /// Take elements from flattened 1-D array.
    ///
    /// See also ``take(_:axis:stream:)``
    public func take(_ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take_all(ctx, indices.ctx, stream.ctx))
    }
    
    /// Transpose the dimensions of the array.
    ///
    /// - Parameters:
    ///     - axes: Specifies the source axis for each axis in the new array
    ///
    /// See also ``transpose(stream:)``
    public func transpose(axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose(ctx, axes.asInt32, axes.count, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// This swaps the position of the first dimension with the given axis.
    ///
    /// See also ``transpose(axes:stream:)``
    public func transpose(axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose(ctx, [axis.int32], 1, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// With no axes specified this will reverse the axes in the array.
    ///
    /// See also ``transpose(axes:stream:)``
    public func transpose(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose_all(ctx, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// Cover for ``transpose(stream:)``
    public func T(stream: StreamOrDevice = .default) -> MLXArray {
        transpose(stream: stream)
    }

}
