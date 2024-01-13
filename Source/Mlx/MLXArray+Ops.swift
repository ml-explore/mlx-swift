import Foundation
import Cmlx

// MARK: - Operations

infix operator ** : BitwiseShiftPrecedence
infix operator *** : MultiplicationPrecedence
infix operator /% : MultiplicationPrecedence

extension MLXArray {
    
    /// Element-wise addition.
    ///
    /// Add two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a + b + 7
    /// ```
    public static func +(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_add(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise subtraction.
    ///
    /// Subtract two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a - b - 7
    /// ```
    public static func -(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_subtract(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Unary element-wise negation.
    ///
    /// Negate the values in the array.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let r = -a
    /// ```
    public static prefix func -(lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_negative(lhs.ctx, s.ctx))
    }
    
    /// Unary element-wise logical not.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    /// let r = !(a == b)
    /// ```
    public static prefix func !(lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_logical_not(lhs.ctx, s.ctx))
    }

    /// Element-wise multiplication.
    ///
    /// Multiply two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a * b * 7
    /// ```
    public static func *(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_multiply(lhs.ctx, rhs.ctx, s.ctx))
    }
    
    /// Element-wise power operation.
    ///
    /// Raise the elements of `lhs` to the powers in elements of `rhs` with numpy-style
    /// broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// // same as a.pow(b)
    /// let r = a ** b
    /// ```
    ///
    /// ### See Also
    /// - ``pow(_:stream:)``
    public static func **(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_power(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Matrix multiplication.
    ///
    /// Perform the (possibly batched) matrix multiplication of two arrays. This function supports
    /// broadcasting for arrays with more than two dimensions.
    ///
    /// - If the first array is 1-D then a 1 is prepended to its shape to make it
    ///   a matrix. Similarly if the second array is 1-D then a 1 is appended to its
    ///   shape to make it a matrix. In either case the singleton dimension is removed
    ///   from the result.
    /// - A batched matrix multiplication is performed if the arrays have more than
    ///   2 dimensions.  The matrix dimensions for the matrix product are the last
    ///   two dimensions of each input.
    /// - All but the last two dimensions of each input are broadcast with one another using
    ///   standard numpy-style broadcasting semantics.
    ///
    /// Note: this is the same as the `@` operator in python.  `@` is not available as an operator in
    /// swift so we are using `***`.  You can also call it as a function: ``matmul(_:stream:)``.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray([1, 2, 3, 4], [2, 2])
    /// let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])
    ///
    /// // produces a [2, 3] result
    /// let r = a *** b
    /// ```
    ///
    /// ### See Also
    /// - ``matmul(_:stream:)``
    public static func ***(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_matmul(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise division.
    ///
    /// Divide two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a / b / 7
    /// ```
    public static func /(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_divide(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise integer division..
    ///
    /// Divide two arrays with numpy-style broadcasting semantics.
    ///
    /// If either array is a floating point type then it is equivalent to calling ``floor(stream:)`` after `/`.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a /% b
    /// ```
    ///
    /// Note: this is the same as the `//` operator in python.  We can't use `//` as an operator in
    /// swift (it means comment to end of line!) so we use `/%`.  ``floorDivide(_:stream:)``
    /// is also available as a method.
    ///
    /// ### See Also
    /// - ``floorDivide(_:stream:)``
    /// - ``floor(stream:)``
    public static func /%(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_floor_divide(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise remainder of division.
    ///
    /// Computes the remainder of dividing `lhs` with `rhs` with numpy-style
    /// broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    ///
    /// let r = a % 2
    /// ```
    public static func %(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_remainder(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise equality.
    ///
    /// Equality comparison on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a == b).allTrue() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - ``allClose(_:rtol:atol:stream:)``
    /// - ``arrayEqual(_:equalNAN:stream:)``
    /// - ``allTrue(stream:)``
    public static func ==(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise less than or equal.
    ///
    /// Less than or equal on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a <= b).allTrue() {
    ///     ...
    /// }
    /// ```
    public static func <=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_less_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise less greater than or equal.
    ///
    /// Greater than or equal on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a >= b).allTrue() {
    ///     ...
    /// }
    /// ```
    public static func >=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_greater_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise not equal.
    ///
    /// Not equal on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a != b).allTrue() {
    ///     ...
    /// }
    /// ```
    public static func !=(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_not_equal(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise less than.
    ///
    /// Less than on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a < b).allTrue() {
    ///     ...
    /// }
    /// ```
    public static func <(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_less(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise greater than.
    ///
    /// greater than on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a > b).allTrue() {
    ///     ...
    /// }
    /// ```
    public static func >(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_greater(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise logical and.
    ///
    /// Logical and on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a < b) && ((a + 1) > b)
    /// ```
    public static func &&(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        return MLXArray(mlx_logical_and(lhs.ctx, rhs.ctx, s.ctx))
    }

    /// Element-wise logical or.
    ///
    /// Logical or on two arrays with numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if r = (a < b) || ((a + 1) > b)
    /// ```
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
    ///
    /// ### See Also
    /// - ``all(axis:keepDims:stream:)``
    /// - ``all(keepDims:stream:)``
    /// - ``allTrue(stream:)``
    /// - ``any(axes:keepDims:stream:)``
    public func all(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_axes(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
    
    /// An `and` reduction over the given axes.
    ///
    /// ### See Also
    /// - ``all(axes:keepDims:stream:)``
    /// - ``all(keepDims:stream:)``
    /// - ``allTrue(stream:)``
    /// - ``any(axes:keepDims:stream:)``
    public func all(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_all_axis(ctx, axis.int32, keepDims, stream.ctx))
    }

    /// An `and` reduction over the given axes.
    ///
    /// ### See Also
    /// - ``all(axes:keepDims:stream:)``
    /// - ``all(axis:keepDims:stream:)``
    /// - ``allTrue(stream:)``
    /// - ``any(axes:keepDims:stream:)``
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
    ///
    /// Note unlike ``arrayEqual(_:equalNAN:stream:)``, this function supports numpy-style broadcasting.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 4).sqrt()
    /// let b: MLXArray(0 ..< 4) ** 0.5
    ///
    /// if a.allClose(b).allTrue() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - ``arrayEqual(_:equalNAN:stream:)``
    public func allClose(_ other: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_allclose(self.ctx, other.ctx, rtol, atol, stream.ctx))
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
    ///
    /// Note: this is equivalent to using an array in a boolean context in python.
    ///
    /// ### See Also
    /// - ``all(axes:keepDims:stream:)``
    /// - ``any(axes:keepDims:stream:)``
    public func allTrue(stream: StreamOrDevice = .default) -> Bool {
        if self.ndim > 1 || self.dtype != .bool {
            let all = mlx_all_all(ctx, false, stream.ctx)!
            let bool = mlx_array_item_bool(all)
            mlx_free(all)
            return bool
        } else {
            let bool = mlx_array_item_bool(ctx)
            return bool
        }
    }
    
    /// An `or` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// // will produce a scalar MLXArray with true -- some of the values are non-zero
    /// let all = array.any()
    ///
    /// // produces an MLXArray([true, true, true, true]) -- all rows have non-zeros
    /// let allRows = array.any(axes: [0])
    ///
    /// // equivalent
    /// let allRows2 = array.any(axis: 0)
    /// ```
    ///
    /// ### See Also
    /// - ``any(axis:keepDims:stream:)``
    /// - ``any(keepDims:stream:)``
    /// - ``all(axes:keepDims:stream:)``
    /// - ``allTrue(stream:)``
    public func any(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_any(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
    
    /// An `or` reduction over the given axes.
    ///
    /// ### See Also
    /// - ``any(axes:keepDims:stream:)``
    /// - ``any(keepDims:stream:)``
    /// - ``all(axes:keepDims:stream:)``
    /// - ``allTrue(stream:)``
    public func any(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_any(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// An `or` reduction over the given axes.
    ///
    /// ### See Also
    /// - ``any(axes:keepDims:stream:)``
    /// - ``any(axis:keepDims:stream:)``
    /// - ``all(axes:keepDims:stream:)``
    /// - ``allTrue(stream:)``
    public func any(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_any_all(ctx, keepDims, stream.ctx))
    }
    
    /// Indices of the maximum values along the axis.
    ///
    /// ```
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [3, 3, 3] -- the index in each column for the maximum value
    /// let i = array.argMax(axis=0)
    /// ```
    ///
    /// ### See Also
    /// - ``argMax(keepDims:stream:)``
    /// - ``argMin(axis:keepDims:stream:)``
    public func argMax(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_argmax(ctx, axis.int32, keepDims, stream.ctx))
    }

    /// Indices of the maximum value over the entire array.
    ///
    /// ```
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [11] -- the index in the flattened array of the largest value
    /// let i = array.argMax()
    /// ```
    ///
    /// ### See Also
    /// - ``argMax(axis:keepDims:stream:)``
    /// - ``argMin(axis:keepDims:stream:)``
    public func argMax(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_argmax_all(ctx, keepDims, stream.ctx))
    }

    /// Indices of the minimum values along the axis.
    ///
    /// ```
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [0, 0, 0] -- the index in each column for the minimum value
    /// let i = array.argMin(axis=0)
    /// ```
    ///
    /// ### See Also
    /// - ``argMin(keepDims:stream:)``
    /// - ``argMax(axis:keepDims:stream:)``
    public func argMin(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_argmin(ctx, axis.int32, keepDims, stream.ctx))
    }

    /// Indices of the minimum value over the entire array.
    ///
    /// ```
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [0] -- the index in the flattened array of the smallest value
    /// let i = array.argMin()
    /// ```
    ///
    /// ### See Also
    /// - ``argMin(axis:keepDims:stream:)``
    /// - ``argMax(axis:keepDims:stream:)``
    public func argMin(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_argmin_all(ctx, keepDims, stream.ctx))
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
    /// ### See Also
    /// - ``allClose(_:rtol:atol:stream:)``
    /// - ``item(_:)``
    /// - ``allTrue(stream:)``
    /// - ``==(lhs:rhs:)``
    public func arrayEqual(_ other: MLXArray, equalNAN: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_array_equal(ctx, other.ctx, equalNAN, stream.ctx))
    }

    /// Element-wise cosine.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func cos(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_cos(ctx, stream.ctx))
    }
    
    /// Return the cumulative maximum of the elements along the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(axis: 0)
    /// ```
    ///
    /// ### See Also
    /// - ``cummax(reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cummax(axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_cummax(ctx, axis.int32, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative maximum of the elements over the flattened array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 8, 8, 9]
    /// let result = array.cummax()
    /// ```
    ///
    /// ### See Also
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cummax(reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cummax(reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let flat = mlx_reshape(ctx, [-1], 1, stream.ctx)!
        defer { mlx_free(flat) }
        return MLXArray(mlx_cummax(ctx, 0, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative minimum of the elements along the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
    /// let result = array.cummin(axis: 0)
    /// ```
    ///
    /// ### See Also
    /// - ``cummin(reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cummin(axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_cummin(ctx, axis.int32, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative minimum of the elements over the flattened array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 5, 4, 4]
    /// let result = array.cummin()
    /// ```
    ///
    /// ### See Also
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cummin(reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let flat = mlx_reshape(ctx, [-1], 1, stream.ctx)!
        defer { mlx_free(flat) }
        return MLXArray(mlx_cummin(ctx, 0, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative product of the elements along the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [20, 72]] -- cumulative product along the columns
    /// let result = array.cumprod(axis: 0)
    /// ```
    ///
    /// ### See Also
    /// - ``cumprod(reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cumprod(axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_cumprod(ctx, axis.int32, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative product of the elements over the flattened array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 40, 160, 1440]
    /// let result = array.cumprod()
    /// ```
    ///
    /// ### See Also
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    public func cumprod(reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let flat = mlx_reshape(ctx, [-1], 1, stream.ctx)!
        defer { mlx_free(flat) }
        return MLXArray(mlx_cumprod(ctx, 0, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative sum of the elements along the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [9, 17]] -- cumulative sum along the columns
    /// let result = array.cumsum(axis: 0)
    /// ```
    ///
    /// ### See Also
    /// - ``cumsum(reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    public func cumsum(axis: Int, reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_cumsum(ctx, axis.int32, reverse, inclusive, stream.ctx))
    }

    /// Return the cumulative sum of the elements over the flattened array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 13, 17, 26]
    /// let result = array.cumsum()
    /// ```
    ///
    /// ### See Also
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    public func cumsum(reverse: Bool = false, inclusive: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let flat = mlx_reshape(ctx, [-1], 1, stream.ctx)!
        defer { mlx_free(flat) }
        return MLXArray(mlx_cumsum(ctx, 0, reverse, inclusive, stream.ctx))
    }

    /// Element-wise exponential.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func exp(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_exp(ctx, stream.ctx))
    }

    /// Flatten an array.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< (8 * 4 * 3), [8, 4, 3])
    ///
    /// // f1 is shape [8 * 4 * 3] = [96]
    /// let f1 = a.flatten()
    ///
    /// // f2 is [8, 4 * 3] = [8, 12]
    /// let f2 = a.flatten(start: 1)
    /// ```
    ///
    /// - Parameters:
    ///     - start: first dimension to flatten
    ///     - end: last dimension to flatten
    ///
    /// ### See Also
    /// - ``reshape(_:stream:)``
    /// - ``squeeze(axes:stream:)``
    public func flatten(start: Int = 0, end: Int = -1, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_flatten(ctx, start.int32, end.int32, stream.ctx))
    }
    
    /// Element-wise floor.
    ///
    /// ### See Also
    /// - ``round(decimals:stream:)``
    /// - ``floorDivide(_:stream:)``
    public func floor(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_floor(ctx, stream.ctx))
    }

    /// Element-wise integer division..
    ///
    /// Divide two arrays with numpy-style broadcasting semantics.
    ///
    /// If either array is a floating point type then it is equivalent to calling ``floor(stream:)`` after `/`.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a.floorDivide(b)
    /// ```
    ///
    /// ### See Also
    /// - ``/%(lhs:rhs:)``
    /// - ``floor(stream:)``
    public func floorDivide(_ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_floor_divide(ctx, other.ctx, stream.ctx))
    }

    /// Element-wise natural logarithm.
    /// 
    /// ### See Also
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    ///
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func log(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_log(ctx, stream.ctx))
    }

    /// Element-wise base-2 logarithm.
    ///
    /// ### See Also
    /// - ``log(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    ///
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func log2(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_log2(ctx, stream.ctx))
    }

    /// Element-wise base-10 logarithm.
    ///
    /// ### See Also
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log1p(stream:)``
    ///
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func log10(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_log10(ctx, stream.ctx))
    }

    /// Element-wise natural log of one plus the array.
    ///
    /// ### See Also
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    ///
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func log1p(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_log1p(ctx, stream.ctx))
    }
    
    /// A `log-sum-exp` reduction over the given axes.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```
    /// log(sum(exp(a), [axes]]))
    /// ```
    ///
    /// ### See Also
    /// - ``logSumExp(axis:keepDims:stream:)``
    /// - ``logSumExp(keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func logSumExp(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_logsumexp(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }

    /// A `log-sum-exp` reduction over the given axis.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```
    /// log(sum(exp(a), axis))
    /// ```
    ///
    /// ### See Also
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``logSumExp(keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func logSumExp(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_logsumexp(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// A `log-sum-exp` reduction over the entire array.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```
    /// log(sum(exp(a)))
    /// ```
    ///
    /// ### See Also
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``logSumExp(axis:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func logSumExp(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_logsumexp_all(ctx, keepDims, stream.ctx))
    }
    
    /// Matrix multiplication.
    ///
    /// Perform the (possibly batched) matrix multiplication of two arrays. This function supports
    /// broadcasting for arrays with more than two dimensions.
    ///
    /// - If the first array is 1-D then a 1 is prepended to its shape to make it
    ///   a matrix. Similarly if the second array is 1-D then a 1 is appended to its
    ///   shape to make it a matrix. In either case the singleton dimension is removed
    ///   from the result.
    /// - A batched matrix multiplication is performed if the arrays have more than
    ///   2 dimensions.  The matrix dimensions for the matrix product are the last
    ///   two dimensions of each input.
    /// - All but the last two dimensions of each input are broadcast with one another using
    ///   standard numpy-style broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray([1, 2, 3, 4], [2, 2])
    /// let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])
    ///
    /// // produces a [2, 3] result
    /// let r = a.matmul(b)
    /// ```
    ///
    /// ### See Also
    /// - ``***(lhs:rhs:)``
    public func matmul(_ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_matmul(ctx, other.ctx, stream.ctx))
    }
    
    /// A `max` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 9]
    /// let result = array.max(axis=[0])
    /// ```
    ///
    /// ### See Also
    /// - ``max(axis:keepDims:stream:)``
    /// - ``max(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func max(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_max(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }

    /// A `max` reduction over the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [8, 9]
    /// let result = array.max(axis=1)
    /// ```
    ///
    /// ### See Also
    /// - ``max(axes:keepDims:stream:)``
    /// - ``max(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func max(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_max(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// A `max` reduction over the entire array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [9]
    /// let result = array.max()
    /// ```
    ///
    /// ### See Also
    /// - ``max(axes:keepDims:stream:)``
    /// - ``max(axis:keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func max(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_max_all(ctx, keepDims, stream.ctx))
    }

    /// A `mean` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean(axis=[0])
    /// ```
    ///
    /// ### See Also
    /// - ``mean(axis:keepDims:stream:)``
    /// - ``mean(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func mean(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_mean(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }

    /// A `mean` reduction over the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [6.5, 6.5]
    /// let result = array.mean(axis=1)
    /// ```
    ///
    /// ### See Also
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``mean(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func mean(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_mean(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// A `mean` reduction over the entire array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [6.5]
    /// let result = array.mean()
    /// ```
    ///
    /// ### See Also
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``mean(axis:keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func mean(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_mean_all(ctx, keepDims, stream.ctx))
    }

    /// A `min` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [4, 8]
    /// let result = array.min(axis=[0])
    /// ```
    ///
    /// ### See Also
    /// - ``min(axis:keepDims:stream:)``
    /// - ``min(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    public func min(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_min(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }

    /// A `min` reduction over the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 4]
    /// let result = array.min(axis=1)
    /// ```
    ///
    /// ### See Also
    /// - ``min(axes:keepDims:stream:)``
    /// - ``min(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    public func min(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_min(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// A `min` reduction over the entire array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5]
    /// let result = array.min()
    /// ```
    ///
    /// ### See Also
    /// - ``min(axes:keepDims:stream:)``
    /// - ``min(axis:keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    public func min(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_min_all(ctx, keepDims, stream.ctx))
    }
    
    /// Move an axis to a new position.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
    /// print(array)
    /// // array([[[[0, 1],
    /// //          [2, 3]],
    /// //         [[4, 5],
    /// //          [6, 7]]],
    /// //        [[[8, 9],
    /// //          [10, 11]],
    /// //         [[12, 13],
    /// //          [14, 15]]]], dtype=int64)
    ///
    /// let r = array.moveAxis(source: 0, destination: 3)
    /// print(r)
    /// // array([[[[0, 8],
    /// //          [1, 9]],
    /// //         [[2, 10],
    /// //          [3, 11]]],
    /// //        [[[4, 12],
    /// //          [5, 13]],
    /// //         [[6, 14],
    /// //          [7, 15]]]], dtype=int64)
    /// ```
    ///
    /// ### See Also
    /// - ``transpose(axes:stream:)``
    /// - ``swapAxes(_:_:stream:)``
    public func moveAxis(source: Int, destination: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_moveaxis(ctx, source.int32, destination.int32, stream.ctx))
    }
    
    /// Element-wise power operation.
    ///
    /// Raise the elements of `self` to the powers in elements of `other` with numpy-style
    /// broadcasting semantics.
    ///
    /// For example:
    ///
    /// ```
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// // same as a ** b
    /// let r = a.pow(b)
    /// ```
    ///
    /// ### See Also
    /// - ``**(lhs:rhs:)``
    public func pow(_ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_power(ctx, other.ctx, stream.ctx))
    }
    
    /// A `product` reduction over the given axes.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [20, 72]
    /// let result = array.product(axis=[0])
    /// ```
    ///
    /// ### See Also
    /// - ``product(axis:keepDims:stream:)``
    /// - ``product(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func product(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_prod(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }

    /// A `product` reduction over the given axis.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [40, 36]
    /// let result = array.product(axis=1)
    /// ```
    ///
    /// ### See Also
    /// - ``product(axes:keepDims:stream:)``
    /// - ``product(keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func product(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_prod(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }

    /// A `product` reduction over the entire array.
    ///
    /// ```
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [1440]
    /// let result = array.product()
    /// ```
    ///
    /// ### See Also
    /// - ``product(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``max(axes:keepDims:stream:)``
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``min(axes:keepDims:stream:)``
    public func product(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_prod_all(ctx, keepDims, stream.ctx))
    }
    
    /// Element-wise reciprocal.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func reciprocal(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reciprocal(ctx, stream.ctx))
    }

    /// Reshape an array while preserving the size.
    public func reshape(_ newShape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_reshape(ctx, newShape.asInt32, newShape.count, stream.ctx))
    }
    
    /// Round to the given number of decimals.
    ///
    /// Roughly equivalent to:
    ///
    /// ```
    /// let array: MLXArray
    ///
    /// let s = 10 ** decimals
    /// let result = round(array * s) / s
    /// ```
    ///
    /// ### See Also
    /// - ``floor(stream:)``
    public func round(decimals: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_round(ctx, decimals.int32, stream.ctx))
    }
    
    /// Element-wise reciprocal and square root.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func rsqrt(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_rsqrt(ctx, stream.ctx))
    }

    /// Element-wise sine.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    /// - ``square(stream:)``
    public func sin(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sin(ctx, stream.ctx))
    }
    
    /// Remove length one axes from an array.
    ///
    /// - Parameters:
    ///     - parts: array is split into that many sections of equal size. It is a fatal error if this is not possible
    ///     - axis: axis to split along
    ///
    /// See also ``split(indices:axis:stream:)``
    public func split(parts: Int, axis: Int = 0, stream: StreamOrDevice = .default) -> [MLXArray] {
        let vec = mlx_split_equal_parts(ctx, parts.int32, axis.int32, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        return MLXArray.fromVector(vec)
    }

    /// Split an array along a given axis.
    ///
    /// - Parameters:
    ///     - indices: the indices of the start of each subarray along the given axis
    ///     - axis: axis to split along
    ///
    /// See also ``split(parts:axis:stream:)``
    public func split(indices: [Int], axis: Int = 0, stream: StreamOrDevice = .default) -> [MLXArray] {
        let vec = mlx_split(ctx, indices.asInt32, indices.count, axis.int32, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        return MLXArray.fromVector(vec)
    }

    /// Element-wise square root
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``square(stream:)``
    public func sqrt(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sqrt(ctx, stream.ctx))
    }
    
    /// Element-wise square.
    ///
    /// ### See Also
    /// - ``cos(stream:)``
    /// - ``exp(stream:)``
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``reciprocal(stream:)``
    /// - ``rsqrt(stream:)``
    /// - ``sin(stream:)``
    /// - ``sqrt(stream:)``
    public func square(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_square(ctx, stream.ctx))
    }
    
    /// Remove length one axes from an array.
    ///
    /// - Parameters:
    ///     - axes: axes to remove
    ///
    /// ### See Also
    /// - ``flatten(start:end:stream:)``
    /// - ``reshape(_:stream:)``
    public func squeeze(axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_squeeze(ctx, axes.asInt32, axes.count, stream.ctx))
    }

    /// Remove length one axes from an array.
    ///
    /// - Parameters:
    ///     - axis: axis to remove
    ///
    /// ### See Also
    /// - ``flatten(start:end:stream:)``
    /// - ``reshape(_:stream:)``
    /// - ``squeeze(axes:stream:)``
    public func squeeze(axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_squeeze(ctx, [axis.int32], 1, stream.ctx))
    }
    
    /// Remove all length one axes from an array.
    ///
    /// ### See Also
    /// - ``flatten(start:end:stream:)``
    /// - ``reshape(_:stream:)``
    /// - ``squeeze(axes:stream:)``
    public func squeeze(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_squeeze_all(ctx, stream.ctx))
    }

    /// Sum reduce the array over the given axes.
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///
    /// ### See Also
    /// - ``sum(axis:keepDims:stream:)``
    /// - ``sum(keepDims:stream:)``
    public func sum(axes: [Int], keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sum(ctx, axes.asInt32, axes.count, keepDims, stream.ctx))
    }
    
    /// Sum reduce the array over the given axis.
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep the reduces axis as singleton dimensions
    ///
    /// ### See Also
    /// - ``sum(axes:keepDims:stream:)``
    /// - ``sum(keepDims:stream:)``
    public func sum(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sum(ctx, [axis.int32], 1, keepDims, stream.ctx))
    }
    
    /// Sum reduce the array over all axes.
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///
    /// ### See Also
    /// - ``sum(axes:keepDims:stream:)``
    /// - ``sum(axis:keepDims:stream:)``
    public func sum(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_sum_all(ctx, keepDims, stream.ctx))
    }
    
    /// Swap two axes of an array.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 16, [2, 2, 2, 2])
    /// print(array)
    /// // array([[[[0, 1],
    /// //          [2, 3]],
    /// //         [[4, 5],
    /// //          [6, 7]]],
    /// //        [[[8, 9],
    /// //          [10, 11]],
    /// //         [[12, 13],
    /// //          [14, 15]]]], dtype=int64)
    ///
    /// let r = array.swapAxes(2, 1)
    /// print(r)
    /// // array([[[[0, 8],
    /// //          [2, 10]],
    /// //         [[4, 12],
    /// //          [6, 14]]],
    /// //        [[[1, 9],
    /// //          [3, 11]],
    /// //         [[5, 13],
    /// //          [7, 15]]]], dtype=int64)
    /// ```
    ///
    /// ### See Also
    /// - ``transpose(axes:stream:)``
    /// - ``moveAxis(source:destination:stream:)``
    public func swapAxes(_ axis1: Int, _ axis2: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_swapaxes(ctx, axis1.int32, axis2.int32, stream.ctx))
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
    /// ### See Also
    /// - ``take(_:stream:)``
    public func take(_ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take(ctx, indices.ctx, axis.int32, stream.ctx))
    }
    
    /// Take elements from flattened 1-D array.
    ///
    /// ### See Also
    /// - ``take(_:axis:stream:)``
    public func take(_ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_take_all(ctx, indices.ctx, stream.ctx))
    }
    
    /// Transpose the dimensions of the array.
    ///
    /// - Parameters:
    ///     - axes: Specifies the source axis for each axis in the new array
    ///
    /// ### See Also
    /// - ``transpose(axis:stream:)``
    /// - ``transpose(stream:)``
    /// - ``moveAxis(source:destination:stream:)``
    /// - ``transpose(axes:stream:)``
    /// - ``swapAxes(_:_:stream:)``
    public func transpose(axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose(ctx, axes.asInt32, axes.count, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// This swaps the position of the first dimension with the given axis.
    ///
    /// ### See Also
    /// - ``transpose(axes:stream:)``
    /// - ``transpose(stream:)``
    /// - ``moveAxis(source:destination:stream:)``
    /// - ``transpose(axes:stream:)``
    /// - ``swapAxes(_:_:stream:)``
    public func transpose(axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose(ctx, [axis.int32], 1, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// With no axes specified this will reverse the axes in the array.
    ///
    /// ### See Also
    /// - ``transpose(axes:stream:)``
    /// - ``transpose(axis:stream:)``
    /// - ``moveAxis(source:destination:stream:)``
    /// - ``transpose(axes:stream:)``
    /// - ``swapAxes(_:_:stream:)``
    public func transpose(stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_transpose_all(ctx, stream.ctx))
    }

    /// Transpose the dimensions of the array.
    ///
    /// Cover for ``transpose(stream:)``
    public func T(stream: StreamOrDevice = .default) -> MLXArray {
        transpose(stream: stream)
    }
    
    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///
    /// ### See Also
    /// - ``variance(axis:keepDims:ddof:stream:)``
    /// - ``variance(keepDims:ddof:stream:)``
    public func variance(axes: [Int], keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_var(ctx, axes.asInt32, axes.count, keepDims, ddof.int32, stream.ctx))
    }

    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - axis: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axis as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///
    ///
    /// ### See Also
    /// - ``variance(axes:keepDims:ddof:stream:)``
    /// - ``variance(keepDims:ddof:stream:)``
    public func variance(axis: Int, keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_var(ctx, [axis.int32], 1, keepDims, ddof.int32, stream.ctx))
    }

    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///
    ///
    /// ### See Also
    /// - ``variance(axes:keepDims:ddof:stream:)``
    /// - ``variance(axis:keepDims:ddof:stream:)``
    public func variance(keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_var_all(ctx, keepDims, ddof.int32, stream.ctx))
    }

}
