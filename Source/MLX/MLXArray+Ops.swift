// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

infix operator ** : BitwiseShiftPrecedence
infix operator .&& : LogicalConjunctionPrecedence
infix operator .|| : LogicalDisjunctionPrecedence

extension MLXArray {

    // MARK: - Arithmetic Operators

    // Note: each of the operators needs three overloads.  Ideally we could write:
    //
    // func +(ScalarOrArray, ScalarOrArray) -> MLXArray
    //
    // but there are a couple problems:
    // - this would be ambiguous with e.g. +(Int, Int)
    // - the operator on MLXArray has to have one parameter that is MLXArray
    //
    // so we need +(MLXArray, ScalarOrArray) and +(ScalarOrArray, MLXArray).
    // If we have two MLXArrays, it is ambiguous which one to call so we also
    // need +(MLXArray, MLXArray).

    /// Element-wise addition.
    ///
    /// Add two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a + b + 7
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``add(_:_:stream:)``
    public static func + (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_add(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise addition.
    ///
    /// Add two arrays with <doc:broadcasting> and store the result in the left-hand side array.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// a += b
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``add(_:_:stream:)``
    public static func += (lhs: inout MLXArray, rhs: MLXArray) {
        lhs._updateInternal(lhs + rhs)
    }

    /// Element-wise addition with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/+(_:_:)-(MLXArray,MLXArray)``
    public static func + (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs + rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise addition with a ``ScalarOrArray`` (scalar) argument and store the result in the left-hand side array.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/+(_:_:)-(MLXArray,MLXArray)``
    public static func += (lhs: inout MLXArray, rhs: some ScalarOrArray) {
        lhs += rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise addition with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/+(_:_:)-(MLXArray,MLXArray)``
    public static func + (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) + rhs
    }

    /// Element-wise subtraction.
    ///
    /// Subtract two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a - b - 7
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``subtract(_:_:stream:)``
    public static func - (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_subtract(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise subtraction.
    ///
    /// Subtract two arrays with <doc:broadcasting> and store the result in the left-hand side array.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// a -= b
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``subtract(_:_:stream:)``
    public static func -= (lhs: inout MLXArray, rhs: MLXArray) {
        lhs._updateInternal(lhs - rhs)
    }

    /// Element-wise subtraction with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func - (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs - rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise subtraction with a ``ScalarOrArray`` (scalar) argument and store the result in the left-hand side array.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func -= (lhs: inout MLXArray, rhs: some ScalarOrArray) {
        lhs -= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise subtraction with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func - (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) - rhs
    }

    /// Unary element-wise negation.
    ///
    /// Negate the values in the array.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let r = -a
    /// ```
    ///
    /// ### See Also
    /// - ``negative(_:stream:)``
    public static prefix func - (lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_negative(&result, lhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise multiplication.
    ///
    /// Multiply two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a * b * 7
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``multiply(_:_:stream:)``
    /// - ``matmul(_:stream:)``
    /// - ``matmul(_:_:stream:)``
    public static func * (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_multiply(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise multiplication.
    ///
    /// Multiply two arrays with <doc:broadcasting> and store the result in the left-hand side array.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// a *= b
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``multiply(_:_:stream:)``
    /// - ``matmul(_:stream:)``
    /// - ``matmul(_:_:stream:)``
    public static func *= (lhs: inout MLXArray, rhs: MLXArray) {
        lhs._updateInternal(lhs * rhs)
    }

    /// Element-wise multiplication with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/*(_:_:)-(MLXArray,MLXArray)``
    public static func * (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs * rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise multiplication with a ``ScalarOrArray`` (scalar) argument and store the result in the left-hand side array.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/*(_:_:)-(MLXArray,MLXArray)``
    public static func *= (lhs: inout MLXArray, rhs: some ScalarOrArray) {
        lhs *= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise multiplication with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/*(_:_:)-(MLXArray,MLXArray)``
    public static func * (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) * rhs
    }

    /// Element-wise power operation.
    ///
    /// Raise the elements of `lhs` to the powers in elements of `rhs` with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// // same as a.pow(b)
    /// let r = a ** b
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``pow(_:stream:)``
    /// - ``pow(_:_:stream:)-(MLXArray,ScalarOrArray,_)``
    /// - ``pow(_:_:stream:)-(ScalarOrArray,MLXArray,_)``
    public static func ** (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_power(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise power with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/**(_:_:)-(MLXArray,MLXArray)``
    public static func ** (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs ** rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise power with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/**(_:_:)-(MLXArray,MLXArray)``
    public static func ** (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) ** rhs
    }

    /// Element-wise division.
    ///
    /// Divide two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a / b / 7
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``divide(_:_:stream:)``
    /// - ``floorDivide(_:_:stream:)``
    public static func / (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_divide(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise division.
    ///
    /// Divide two arrays with <doc:broadcasting> and store the result in the left-hand side array.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// a /= b
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``divide(_:_:stream:)``
    /// - ``floorDivide(_:_:stream:)``
    public static func /= (lhs: inout MLXArray, rhs: MLXArray) {
        lhs._updateInternal(lhs / rhs)
    }

    /// Element-wise division with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func / (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs / rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise division with a ``ScalarOrArray`` (scalar) argument and store the result in the left-hand side array.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func /= (lhs: inout MLXArray, rhs: some ScalarOrArray) {
        lhs /= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise division with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public static func / (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) / rhs
    }

    /// Element-wise remainder of division.
    ///
    /// Computes the remainder of dividing `lhs` with `rhs` with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    ///
    /// let r = a % 2
    /// ```
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``remainder(_:_:stream:)``
    public static func % (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_remainder(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise remainder with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/%(_:_:)-(MLXArray,MLXArray)``
    public static func % (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) % rhs
    }

    /// Element-wise remainder with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/%(_:_:)-(MLXArray,MLXArray)``
    public static func % (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs % rhs.asMLXArray(dtype: lhs.dtype)
    }

    // MARK: - Logical Operators

    /// Unary element-wise logical not.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    /// let r = .!(a == b)
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``logicalNot(_:stream:)``
    public static prefix func .! (lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_logical_not(&result, lhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise equality.
    ///
    /// Equality comparison on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .== b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``allClose(_:rtol:atol:equalNaN:stream:)``
    /// - ``arrayEqual(_:equalNAN:stream:)``
    /// - ``allClose(_:_:rtol:atol:equalNaN:stream:)``
    public static func .== (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_equal(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise equality with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.==(_:_:)-(MLXArray,MLXArray)``
    public static func .== (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .== rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise less than or equal.
    ///
    /// Less than or equal on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .<= b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``lessEqual(_:_:stream:)``
    public static func .<= (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_less_equal(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise less than or equal with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.<=(_:_:)-(MLXArray,MLXArray)``
    public static func .<= (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .<= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise less greater than or equal.
    ///
    /// Greater than or equal on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .>= b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``greaterEqual(_:_:stream:)``
    public static func .>= (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_greater_equal(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise greater than or equal with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.>=(_:_:)-(MLXArray,MLXArray)``
    public static func .>= (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .>= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise not equal.
    ///
    /// Not equal on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .!= b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``notEqual(_:_:stream:)``
    public static func .!= (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_not_equal(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise not equal with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.!=(_:_:)-mbw0``
    public static func .!= (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .!= rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise less than.
    ///
    /// Less than on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .< b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``less(_:_:stream:)``
    public static func .< (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_less(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise less than with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.<(_:_:)-(MLXArray,MLXArray)``
    public static func .< (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .< rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise greater than.
    ///
    /// Greater than on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// if (a .> b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``greater(_:_:stream:)``
    public static func .> (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_greater(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise greater than with a ``ScalarOrArray`` (scalar) argument.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``MLXArray/.>(_:_:)-fwi1``
    public static func .> (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs .> rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise logical and.
    ///
    /// Logical and on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// let r = (a .< b) .&& ((a + 1) .> b)
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``logicalAnd(_:_:stream:)``
    public static func .&& (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_logical_and(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise logical or.
    ///
    /// Logical or on two arrays with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = a + 1
    ///
    /// let r = (a .< b) .|| ((a + 1) .> b)
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``logicalOr(_:_:stream:)``
    public static func .|| (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_logical_or(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Unary element-wise bitwise invert.
    ///
    /// Apply bitwise inversion to the values in the array.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseInvert(_:stream:)``
    public static prefix func ~ (lhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_bitwise_invert(&result, lhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise bitwise and.
    ///
    /// Take the bitwise and of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseAnd(_:_:stream:)``
    public static func & (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_bitwise_and(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise bitwise and.
    ///
    /// Take the bitwise and of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseAnd(_:_:stream:)``
    public static func & (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs & rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise bitwise and.
    ///
    /// Take the bitwise and of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseAnd(_:_:stream:)``
    public static func & (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) & rhs
    }

    /// Element-wise bitwise or.
    ///
    /// Take the bitwise or of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseOr(_:_:stream:)``
    public static func | (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_bitwise_or(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise bitwise or.
    ///
    /// Take the bitwise or of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseOr(_:_:stream:)``
    public static func | (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs | rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise bitwise or.
    ///
    /// Take the bitwise or of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseOr(_:_:stream:)``
    public static func | (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) | rhs
    }

    /// Element-wise bitwise xor.
    ///
    /// Take the bitwise xor of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseXOr(_:_:stream:)``
    public static func ^ (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_bitwise_xor(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise bitwise xor.
    ///
    /// Take the bitwise xor of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseXOr(_:_:stream:)``
    public static func ^ (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs ^ rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise bitwise xor.
    ///
    /// Take the bitwise xor of two arrays with numpy-style broadcasting
    /// semantics. Either or both input arrays can also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``bitwiseXOr(_:_:stream:)``
    public static func ^ (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) ^ rhs
    }

    /// Element-wise left shift.
    ///
    /// Shift the bits of the first input to the left by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``leftShift(_:_:stream:)``
    public static func << (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_left_shift(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise left shift.
    ///
    /// Shift the bits of the first input to the left by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``leftShift(_:_:stream:)``
    public static func << (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs << rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise left shift.
    ///
    /// Shift the bits of the first input to the left by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``leftShift(_:_:stream:)``
    public static func << (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) << rhs
    }

    /// Element-wise right shift.
    ///
    /// Shift the bits of the first input to the right by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``rightShift(_:_:stream:)``
    public static func >> (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        let s = StreamOrDevice.default
        var result = mlx_array_new()
        mlx_right_shift(&result, lhs.ctx, rhs.ctx, s.ctx)
        return MLXArray(result)
    }

    /// Element-wise right shift.
    ///
    /// Shift the bits of the first input to the right by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``rightShift(_:_:stream:)``
    public static func >> (lhs: MLXArray, rhs: some ScalarOrArray) -> MLXArray {
        lhs >> rhs.asMLXArray(dtype: lhs.dtype)
    }

    /// Element-wise right shift.
    ///
    /// Shift the bits of the first input to the right by the second using
    /// numpy-style broadcasting semantics. Either or both input arrays can
    /// also be scalars.
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``rightShift(_:_:stream:)``
    public static func >> (lhs: some ScalarOrArray, rhs: MLXArray) -> MLXArray {
        lhs.asMLXArray(dtype: rhs.dtype) >> rhs
    }

}

extension MLXArray {

    // MARK: - Logical Operator Deprecations

    // deprecations to help users find the right names.

    // variant that returns MLXArray

    @available(
        *, unavailable, renamed: ".!",
        message: "See the article on logical operations for more info."
    )
    public static prefix func ! (lhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".==",
        message: "See the article on logical operations for more info."
    )
    public static func == (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".==",
        message: "See the article on logical operations for more info."
    )
    public static func == <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<=",
        message: "See the article on logical operations for more info."
    )
    public static func <= (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<=",
        message: "See the article on logical operations for more info."
    )
    public static func <= <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>=",
        message: "See the article on logical operations for more info."
    )
    public static func >= (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>=",
        message: "See the article on logical operations for more info."
    )
    public static func >= <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".!=",
        message: "See the article on logical operations for more info."
    )
    public static func != (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".!=",
        message: "See the article on logical operations for more info."
    )
    public static func != <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<",
        message: "See the article on logical operations for more info."
    )
    public static func < (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<",
        message: "See the article on logical operations for more info."
    )
    public static func < <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>",
        message: "See the article on logical operations for more info."
    )
    public static func > (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>",
        message: "See the article on logical operations for more info."
    )
    public static func > <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".&&",
        message: "See the article on logical operations for more info."
    )
    public static func && (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".||",
        message: "See the article on logical operations for more info."
    )
    public static func || (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
        fatalError("unavailable")
    }

    // variant that returns bool

    @available(
        *, unavailable, renamed: ".!",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static prefix func ! (lhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".==",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func == (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".==",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func == <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func <= (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func <= <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func >= (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func >= <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".!=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func != (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".!=",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func != <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func < (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".<",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func < <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func > (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".>",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func > <T: ScalarOrArray>(lhs: MLXArray, rhs: T) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".&&",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func && (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }
    @available(
        *, unavailable, renamed: ".||",
        message:
            "See the article on logical operations for more info and the article on lazy evaluation for cautions of using this in a boolean context."
    )
    public static func || (lhs: MLXArray, rhs: MLXArray) -> Bool {
        fatalError("unavailable")
    }

}

// MARK: - Internal Functions

extension MLXArray {

    func broadcast(to shape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_broadcast_to(&result, ctx, shape, shape.count, stream.ctx)
        return MLXArray(result)
    }

    func scattered(
        indices: [MLXArray], updates: MLXArray, axes: [Int32], stream: StreamOrDevice = .default
    ) -> MLXArray {
        let vector_array = new_mlx_vector_array(indices)
        defer { mlx_vector_array_free(vector_array) }

        var result = mlx_array_new()
        mlx_scatter(&result, ctx, vector_array, updates.ctx, axes, axes.count, stream.ctx)
        return MLXArray(result)
    }

    // varaiant with [Int32] argument
    func reshaped(_ newShape: [Int32], stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_reshape(&result, ctx, newShape, newShape.count, stream.ctx)
        return MLXArray(result)
    }

}

// MARK: - Public Functions

extension MLXArray {

    /// Element-wise absolute value.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    public func abs(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_abs(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// An `and` reduction over the given axes.
    ///
    /// ```swift
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
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``all(axis:keepDims:stream:)``
    /// - ``all(keepDims:stream:)``
    /// - ``all(_:axes:keepDims:stream:)``
    public func all(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_all_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// An `and` reduction over the given axes.
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``all(axes:keepDims:stream:)``
    /// - ``all(keepDims:stream:)``
    /// - ``all(_:axes:keepDims:stream:)``
    public func all(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_all_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// An `and` reduction over the given axes.
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``all(axes:keepDims:stream:)``
    /// - ``all(axis:keepDims:stream:)``
    /// - ``all(_:axes:keepDims:stream:)``
    public func all(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_all(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Approximate comparison of two arrays.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```swift
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// Note: unlike ``arrayEqual(_:equalNAN:stream:)``, this function supports <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 4).sqrt()
    /// let b: MLXArray(0 ..< 4) ** 0.5
    ///
    /// if a.allClose(b).all().item() {
    ///     ...
    /// }
    /// ```
    ///
    /// - Parameters:
    ///     - other: array to compare to
    ///     - rtol: relative tolerance (see discussion)
    ///     - atol: absolute tolerance (see discussion)
    ///     - equalNaN: if `true` treat NaN values as equal to each other
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``arrayEqual(_:equalNAN:stream:)``
    /// - ``arrayEqual(_:_:equalNAN:stream:)``
    public func allClose(
        _ other: some ScalarOrArray, rtol: Double = 1e-5, atol: Double = 1e-8,
        equalNaN: Bool = false, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let other = other.asMLXArray(dtype: self.dtype)
        var result = mlx_array_new()
        mlx_allclose(&result, ctx, other.ctx, rtol, atol, equalNaN, stream.ctx)
        return MLXArray(result)
    }

    /// An `or` reduction over the given axes.
    ///
    /// ```swift
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
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``any(axis:keepDims:stream:)``
    /// - ``any(keepDims:stream:)``
    /// - ``any(_:axes:keepDims:stream:)``
    public func any(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_any_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// An `or` reduction over the given axes.
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``any(axes:keepDims:stream:)``
    /// - ``any(keepDims:stream:)``
    /// - ``any(_:axes:keepDims:stream:)``
    public func any(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_any_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// An `or` reduction over the given axes.
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``any(axes:keepDims:stream:)``
    /// - ``any(axis:keepDims:stream:)``
    /// - ``any(_:axes:keepDims:stream:)``
    public func any(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_any(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Indices of the maximum values along the axis.
    ///
    /// ```swift
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [3, 3, 3] -- the index in each column for the maximum value
    /// let i = array.argMax(axis=0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:indexes>
    /// - ``argMax(keepDims:stream:)``
    /// - ``argMin(axis:keepDims:stream:)``
    /// - ``argMax(_:axis:keepDims:stream:)``
    public func argMax(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_argmax_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Indices of the maximum value over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [11] -- the index in the flattened array of the largest value
    /// let i = array.argMax()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:indexes>
    /// - ``argMax(axis:keepDims:stream:)``
    /// - ``argMin(axis:keepDims:stream:)``
    /// - ``argMax(_:axis:keepDims:stream:)``
    public func argMax(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_argmax(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Indices of the minimum values along the axis.
    ///
    /// ```swift
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [0, 0, 0] -- the index in each column for the minimum value
    /// let i = array.argMin(axis=0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:indexes>
    /// - ``argMin(keepDims:stream:)``
    /// - ``argMax(axis:keepDims:stream:)``
    /// - ``argMin(_:axis:keepDims:stream:)``
    public func argMin(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_argmin_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Indices of the minimum value over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray(4 ..< 16, [4, 3])
    ///
    /// // this will produce [0] -- the index in the flattened array of the smallest value
    /// let i = array.argMin()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:indexes>
    /// - ``argMin(axis:keepDims:stream:)``
    /// - ``argMax(axis:keepDims:stream:)``
    /// - ``argMin(_:axis:keepDims:stream:)``
    public func argMin(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_argmin(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `True` if and only if the arrays
    /// have the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// ```swift
    /// let a1 = MLXArray([0, 1, 2, 3])
    /// let a2 = MLXArray([0.0, 1.0, 2.0, 3.0])
    ///
    /// print(a1.arrayEqual(a2))
    /// // prints the scalar array `true`
    /// ```
    ///
    /// ### See Also
    /// - <doc:logical>
    /// - ``allClose(_:rtol:atol:equalNaN:stream:)``
    /// - ``MLXArray/==(_:_:)`
    /// - ``arrayEqual(_:_:equalNAN:stream:)``
    public func arrayEqual(
        _ other: some ScalarOrArray, equalNAN: Bool = false, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let other = other.asMLXArray(dtype: self.dtype)
        var result = mlx_array_new()
        mlx_array_equal(&result, ctx, other.ctx, equalNAN, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise complex conjugate of the input.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``conjugate(_:stream:)``
    public func conjugate(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_conjugate(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise cosine.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``cos(_:stream:)``
    public func cos(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_cos(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative maximum of the elements along the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(axis: 0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cummax(reverse:inclusive:stream:)``
    /// - ``cummax(_:axis:reverse:inclusive:stream:)``
    public func cummax(
        axis: Int, reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_cummax(&result, ctx, axis.int32, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative maximum of the elements over the flattened array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 8, 8, 9]
    /// let result = array.cummax()
    /// ```
    ///
    /// - Parameters:
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cummax(axis:reverse:inclusive:stream:)``
    /// - ``cummax(_:axis:reverse:inclusive:stream:)``
    public func cummax(
        reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let flat = self.reshaped([-1], stream: stream)
        var result = mlx_array_new()
        mlx_cummax(&result, flat.ctx, 0, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative minimum of the elements along the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
    /// let result = array.cummin(axis: 0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cummin(reverse:inclusive:stream:)``
    /// - ``cummin(_:axis:reverse:inclusive:stream:)``
    public func cummin(
        axis: Int, reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_cummin(&result, ctx, axis.int32, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative minimum of the elements over the flattened array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 5, 4, 4]
    /// let result = array.cummin()
    /// ```
    ///
    /// - Parameters:
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cummin(axis:reverse:inclusive:stream:)``
    /// - ``cummin(_:axis:reverse:inclusive:stream:)``
    public func cummin(
        reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let flat = self.reshaped([-1], stream: stream)
        var result = mlx_array_new()
        mlx_cummin(&result, flat.ctx, 0, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative product of the elements along the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [20, 72]] -- cumulative product along the columns
    /// let result = array.cumprod(axis: 0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cumprod(reverse:inclusive:stream:)``
    /// - ``cumprod(_:axis:reverse:inclusive:stream:)``
    public func cumprod(
        axis: Int, reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_cumprod(&result, ctx, axis.int32, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative product of the elements over the flattened array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 40, 160, 1440]
    /// let result = array.cumprod()
    /// ```
    ///
    /// - Parameters:
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cumprod(axis:reverse:inclusive:stream:)``
    /// - ``cumprod(_:axis:reverse:inclusive:stream:)``
    public func cumprod(
        reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let flat = self.reshaped([-1], stream: stream)
        var result = mlx_array_new()
        mlx_cumprod(&result, flat.ctx, 0, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative sum of the elements along the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [[5, 8], [9, 17]] -- cumulative sum along the columns
    /// let result = array.cumsum(axis: 0)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cumsum(reverse:inclusive:stream:)``
    /// - ``cumsum(_:axis:reverse:inclusive:stream:)``
    public func cumsum(
        axis: Int, reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_cumsum(&result, ctx, axis.int32, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Return the cumulative sum of the elements over the flattened array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 13, 17, 26]
    /// let result = array.cumsum()
    /// ```
    ///
    /// - Parameters:
    ///     - reverse: reverse the reduction
    ///     - inclusive: include the initial value
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:cumulative>
    /// - ``cumsum(axis:reverse:inclusive:stream:)``
    /// - ``cumsum(_:axis:reverse:inclusive:stream:)``
    public func cumsum(
        reverse: Bool = false, inclusive: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let flat = self.reshaped([-1], stream: stream)
        var result = mlx_array_new()
        mlx_cumsum(&result, flat.ctx, 0, reverse, inclusive, stream.ctx)
        return MLXArray(result)
    }

    /// Extract a diagonal or construct a diagonal matrix.
    ///
    /// If self is 1-D then a diagonal matrix is constructed with self on the
    /// `k`-th diagonal. If self is 2-D then the `k`-th diagonal is
    /// returned.
    ///
    /// - Parameters:
    ///   - k: the diagonal to extract or construct
    ///   - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - ``diagonal(offset:axis1:axis2:stream:)``
    public func diag(k: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_diag(&result, ctx, k.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Return specified diagonals.
    ///
    /// If self is 2-D, then a 1-D array containing the diagonal at the given
    /// `offset` is returned.
    ///
    /// If self has more than two dimensions, then `axis1` and `axis2`
    /// determine the 2D subarrays from which diagonals are extracted. The new
    /// shape is the original shape with `axis1` and `axis2` removed and a
    /// new dimension inserted at the end corresponding to the diagonal.
    ///
    /// - Parameters:
    ///   - offset: offset of the diagonal.  Can be positive or negative
    ///   - axis1: first axis of the 2-D sub-array from which the diagonals should be taken
    ///   - axis2: second axis of the 2-D sub-array from which the diagonals should be taken
    ///   - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - ``diag(k:stream:)``
    public func diagonal(
        offset: Int = 0, axis1: Int = 0, axis2: Int = 1, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_diagonal(&result, ctx, offset.int32, axis1.int32, axis2.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise exponential.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``exp(_:stream:)``
    public func exp(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_exp(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Add a size one dimension at the given axis.
    ///
    /// - Parameters:
    ///     - axes: indexes of the inserted dimensions
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``expandedDimensions(axis:stream:)``
    public func expandedDimensions(axes: some Collection<Int>, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_expand_dims_axes(&result, self.ctx, axes.asInt32, axes.count, stream.ctx)
        return MLXArray(result)
    }

    /// Add a size one dimension at the given axis.
    ///
    /// - Parameters:
    ///     - axis: index of the inserted dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``expandedDimensions(axes:stream:)``
    public func expandedDimensions(axis: Int, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_expand_dims(&result, self.ctx, axis.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Flatten an array.
    ///
    /// The axes flattened will be between `start` and `end`,
    /// inclusive. Negative axes are supported. After converting negative axis to
    /// positive, axes outside the valid range will be clamped to a valid value,
    /// `start` to `0` and `end` to `ndim - 1`.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< (8 * 4 * 3), [8, 4, 3])
    ///
    /// // f1 is shape [8 * 4 * 3] = [96]
    /// let f1 = a.flattened()
    ///
    /// // f2 is [8, 4 * 3] = [8, 12]
    /// let f2 = a.flattened(start: 1)
    /// ```
    ///
    /// - Parameters:
    ///     - start: first dimension to flatten
    ///     - end: last dimension to flatten
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``flattened(_:start:end:stream:)``
    public func flattened(start: Int = 0, end: Int = -1, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_flatten(&result, ctx, start.int32, end.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise floor.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``round(decimals:stream:)``
    /// - ``floorDivide(_:stream:)``
    /// - ``floor(_:stream:)``
    public func floor(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_floor(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise integer division..
    ///
    /// Divide two arrays with <doc:broadcasting>.
    ///
    /// If either array is a floating point type then it is equivalent to calling ``floor(stream:)`` after `/`.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// let r = a.floorDivide(b)
    /// ```
    ///
    /// - Parameters:
    ///     - other: the divisor
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``floor(stream:)``
    /// - ``floorDivide(_:_:stream:)``
    public func floorDivide(_ other: some ScalarOrArray, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        let other = other.asMLXArray(dtype: self.dtype)
        var result = mlx_array_new()
        mlx_floor_divide(&result, ctx, other.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise natural logarithm.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``log(_:stream:)``
    public func log(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_log(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise base-2 logarithm.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``log(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(stream:)``
    /// - ``log2(_:stream:)``
    public func log2(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_log2(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise base-10 logarithm.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log1p(stream:)``
    /// - ``log10(_:stream:)``
    public func log10(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_log10(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise natural log of one plus the array.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``log(stream:)``
    /// - ``log2(stream:)``
    /// - ``log10(stream:)``
    /// - ``log1p(_:stream:)``
    public func log1p(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_log1p(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// A `log-sum-exp` reduction over the given axes.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```swift
    /// log(sum(exp(a), [axes]]))
    /// ```
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``logSumExp(axis:keepDims:stream:)``
    /// - ``logSumExp(keepDims:stream:)``
    /// - ``logSumExp(_:axes:keepDims:stream:)``
    public func logSumExp(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_logsumexp_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `log-sum-exp` reduction over the given axis.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```swift
    /// log(sum(exp(a), axis))
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``logSumExp(keepDims:stream:)``
    /// - ``logSumExp(_:axes:keepDims:stream:)``
    public func logSumExp(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_logsumexp_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `log-sum-exp` reduction over the entire array.
    ///
    /// The log-sum-exp reduction is a numerically stable version of:
    ///
    /// ```swift
    /// log(sum(exp(a)))
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``logSumExp(axes:keepDims:stream:)``
    /// - ``logSumExp(axis:keepDims:stream:)``
    /// - ``logSumExp(_:axes:keepDims:stream:)``
    public func logSumExp(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_logsumexp(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
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
    ///   standard <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray([1, 2, 3, 4], [2, 2])
    /// let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])
    ///
    /// // produces a [2, 3] result
    /// let r = a.matmul(b)
    /// ```
    ///
    /// - Parameters:
    ///     - other: the right hand side array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``matmul(_:_:stream:)``
    public func matmul(_ other: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_matmul(&result, ctx, other.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// A `max` reduction over the given axes.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 9]
    /// let result = array.max(axis=[0])
    /// ```
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``max(axis:keepDims:stream:)``
    /// - ``max(keepDims:stream:)``
    /// - ``max(_:axes:keepDims:stream:)``
    public func max(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_max_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `max` reduction over the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [8, 9]
    /// let result = array.max(axis=1)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``max(axes:keepDims:stream:)``
    /// - ``max(keepDims:stream:)``
    /// - ``max(_:axes:keepDims:stream:)``
    public func max(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_max_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `max` reduction over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [9]
    /// let result = array.max()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``max(axes:keepDims:stream:)``
    /// - ``max(axis:keepDims:stream:)``
    /// - ``max(_:axes:keepDims:stream:)``
    public func max(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_max(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `mean` reduction over the given axes.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean(axis=[0])
    /// ```
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``mean(axis:keepDims:stream:)``
    /// - ``mean(keepDims:stream:)``
    /// - ``mean(_:axes:keepDims:stream:)``
    public func mean(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_mean_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `mean` reduction over the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [6.5, 6.5]
    /// let result = array.mean(axis=1)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``mean(keepDims:stream:)``
    /// - ``mean(_:axes:keepDims:stream:)``
    public func mean(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_mean_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `mean` reduction over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [6.5]
    /// let result = array.mean()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``mean(axes:keepDims:stream:)``
    /// - ``mean(axis:keepDims:stream:)``
    /// - ``mean(_:axes:keepDims:stream:)``
    public func mean(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_mean(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `min` reduction over the given axes.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [4, 8]
    /// let result = array.min(axis=[0])
    /// ```
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``min(axis:keepDims:stream:)``
    /// - ``min(keepDims:stream:)``
    /// - ``min(_:axes:keepDims:stream:)``
    public func min(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_min_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `min` reduction over the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5, 4]
    /// let result = array.min(axis=1)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``min(axes:keepDims:stream:)``
    /// - ``min(keepDims:stream:)``
    /// - ``min(_:axes:keepDims:stream:)``
    public func min(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_min_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `min` reduction over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [5]
    /// let result = array.min()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``min(axes:keepDims:stream:)``
    /// - ``min(axis:keepDims:stream:)``
    /// - ``min(_:axes:keepDims:stream:)``
    public func min(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_min(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Move an axis to a new position.
    ///
    /// ```swift
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
    /// let r = array.movedAxis(source: 0, destination: 3)
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
    /// - Parameters:
    ///     - source: axis to move
    ///     - destination: new position for the axis
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``swappedAxes(_:_:stream:)``
    /// - ``movedAxis(_:source:destination:stream:)``
    public func movedAxis(source: Int, destination: Int, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_moveaxis(&result, ctx, source.int32, destination.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise power operation.
    ///
    /// Raise the elements of `self` to the powers in elements of `other` with <doc:broadcasting>.
    ///
    /// For example:
    ///
    /// ```swift
    /// let a = MLXArray(0 ..< 12, [4, 3])
    /// let b = MLXArray([4, 5, 6])
    ///
    /// // same as a ** b
    /// let r = a.pow(b)
    /// ```
    ///
    /// - Parameters:
    ///     - other: the exponent
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - <doc:arithmetic>
    /// - ``**(_:_:)-(MLXArray,MLXArray)``
    /// - ``pow(_:_:stream:)-(MLXArray,ScalarOrArray,_)``
    public func pow(_ other: some ScalarOrArray, stream: StreamOrDevice = .default) -> MLXArray {
        let other = other.asMLXArray(dtype: self.dtype)
        var result = mlx_array_new()
        mlx_power(&result, ctx, other.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// A `product` reduction over the given axes.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [20, 72]
    /// let result = array.product(axis=[0])
    /// ```
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``product(axis:keepDims:stream:)``
    /// - ``product(keepDims:stream:)``
    /// - ``product(_:axes:keepDims:stream:)``
    public func product(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_prod_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `product` reduction over the given axis.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [40, 36]
    /// let result = array.product(axis=1)
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``product(axes:keepDims:stream:)``
    /// - ``product(keepDims:stream:)``
    /// - ``product(_:axes:keepDims:stream:)``
    public func product(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_prod_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// A `product` reduction over the entire array.
    ///
    /// ```swift
    /// let array = MLXArray([5, 8, 4, 9], [2, 2])
    ///
    /// // result is [1440]
    /// let result = array.product()
    /// ```
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``product(axes:keepDims:stream:)``
    /// - ``product(axis:keepDims:stream:)``
    /// - ``product(_:axes:keepDims:stream:)``
    public func product(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_prod(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise reciprocal.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``reciprocal(_:stream:)``
    public func reciprocal(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_reciprocal(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Reshape an array while preserving the size.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12)
    ///
    /// let r = array.reshaped([4, 3])
    /// ```
    ///
    /// - Parameters:
    ///     - newShape: the new shape
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``reshaped(_:stream:)``
    /// - ``reshaped(_:_:stream:)-(MLXArray,Collection<Int>,StreamOrDevice)``
    public func reshaped(_ newShape: some Collection<Int>, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_reshape(&result, ctx, newShape.asInt32, newShape.count, stream.ctx)
        return MLXArray(result)
    }

    /// Reshape an array while preserving the size.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12)
    ///
    /// let r = array.reshaped(4, 3)
    /// ```
    ///
    /// - Parameters:
    ///     - newShape: the new shape
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``reshaped(_:_:stream:)-(MLXArray,Int...,StreamOrDevice)``
    /// - ``reshaped(_:stream:)``
    public func reshaped(_ newShape: Int..., stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_reshape(&result, ctx, newShape.asInt32, newShape.count, stream.ctx)
        return MLXArray(result)
    }

    /// Round to the given number of decimals.
    ///
    /// Roughly equivalent to:
    ///
    /// ```swift
    /// let array: MLXArray
    ///
    /// let s = 10 ** decimals
    /// let result = round(array * s) / s
    /// ```
    ///
    /// - Parameters:
    ///     - decimals: number of decimal places to round to
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``floor(stream:)``
    /// - ``round(_:decimals:stream:)``
    public func round(decimals: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_round(&result, ctx, decimals.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise reciprocal and square root.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``sqrt(_:stream:)``
    public func rsqrt(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_rsqrt(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise sine.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``sin(_:stream:)``
    public func sin(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_sin(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Split an array into equal size pieces along a given axis.
    ///
    /// Splits the array into equal size pieces along a given axis and returns an array of `MLXArray`:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, (4, 3))
    ///
    /// let halves = array.split(2)
    /// print(halves)
    ///
    /// [array([[0, 1, 2],
    ///         [3, 4, 5]], dtype=int64),
    ///  array([[6, 7, 8],
    ///         [9, 10, 11]], dtype=int64)]
    /// ```
    ///
    /// - Parameters:
    ///     - parts: array is split into that many sections of equal size. It is a fatal error if this is not possible
    ///     - axis: axis to split along
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``split(axis:stream:)``
    /// - ``split(indices:axis:stream:)``
    /// - ``split(_:parts:axis:stream:)``
    public func split(parts: Int, axis: Int = 0, stream: StreamOrDevice = .default) -> [MLXArray] {
        var vec = mlx_vector_array_new()
        mlx_split(&vec, ctx, parts.int32, axis.int32, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        return mlx_vector_array_values(vec)
    }

    /// Split an array into 2 pieces along an axis and returns a tuple -- convenient for destructuring.
    ///
    /// Splits the array into 2 pieces along a given axis and returns an tuple of `MLXArray`:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, (4, 3))
    ///
    /// let (a, b) = array.split()
    /// ```
    ///
    /// - Parameters:
    ///     - axis: axis to split along
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``split(parts:axis:stream:)``
    /// - ``split(indices:axis:stream:)``
    /// - ``split(_:parts:axis:stream:)``
    public func split(axis: Int = 0, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray) {
        var vec = mlx_vector_array_new()
        mlx_split(&vec, ctx, 2, axis.int32, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        let pieces = mlx_vector_array_values(vec)
        return (pieces[0], pieces[1])
    }

    /// Split an array along a given axis.
    ///
    /// - Parameters:
    ///     - indices: the indices of the start of each subarray along the given axis
    ///     - axis: axis to split along
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``split(parts:axis:stream:)``
    /// - ``split(_:indices:axis:stream:)``
    public func split(
        indices: some Collection<Int>, axis: Int = 0, stream: StreamOrDevice = .default
    )
        -> [MLXArray]
    {
        var vec = mlx_vector_array_new()
        mlx_split_sections(&vec, ctx, indices.asInt32, indices.count, axis.int32, stream.ctx)
        defer { mlx_vector_array_free(vec) }
        return mlx_vector_array_values(vec)
    }

    /// Element-wise square root
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``sqrt(_:stream:)``
    public func sqrt(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_sqrt(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Element-wise square.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:arithmetic>
    /// - ``square(_:stream:)``
    public func square(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_square(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Remove length one axes from an array.
    ///
    /// - Parameters:
    ///     - axes: axes to remove
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``squeezed(axis:stream:)``
    /// - ``squeezed(stream:)``
    /// - ``squeezed(_:axes:stream:)``
    public func squeezed(axes: some Collection<Int>, stream: StreamOrDevice = .default) -> MLXArray
    {
        var result = mlx_array_new()
        mlx_squeeze_axes(&result, ctx, axes.asInt32, axes.count, stream.ctx)
        return MLXArray(result)
    }

    /// Remove length one axes from an array.
    ///
    /// - Parameters:
    ///     - axis: axis to remove
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``squeezed(axes:stream:)``
    /// - ``squeezed(stream:)``
    /// - ``squeezed(_:axes:stream:)``
    public func squeezed(axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_squeeze_axis(&result, ctx, axis.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Remove all length one axes from an array.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``squeezed(axes:stream:)``
    /// - ``squeezed(axis:stream:)``
    /// - ``squeezed(_:axes:stream:)``
    public func squeezed(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_squeeze(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Sum reduce the array over the given axes.
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``sum(axis:keepDims:stream:)``
    /// - ``sum(keepDims:stream:)``
    /// - ``sum(_:axes:keepDims:stream:)``
    public func sum(
        axes: some Collection<Int>, keepDims: Bool = false, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_sum_axes(&result, ctx, axes.asInt32, axes.count, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Sum reduce the array over the given axis.
    ///
    /// - Parameters:
    ///     - axis: axis to reduce over
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``sum(axes:keepDims:stream:)``
    /// - ``sum(keepDims:stream:)``
    /// - ``sum(_:axes:keepDims:stream:)``
    public func sum(axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_sum_axis(&result, ctx, axis.int32, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Sum reduce the array over all axes.
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep reduced axis as singleton dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``sum(axes:keepDims:stream:)``
    /// - ``sum(axis:keepDims:stream:)``
    /// - ``sum(_:axes:keepDims:stream:)``
    public func sum(keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_sum(&result, ctx, keepDims, stream.ctx)
        return MLXArray(result)
    }

    /// Swap two axes of an array.
    ///
    /// ```swift
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
    /// let r = array.swappedAxes(2, 1)
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
    /// - Parameters:
    ///     - axis1: first axis
    ///     - axis2: second axis
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``swappedAxes(_:_:_:stream:)``
    public func swappedAxes(_ axis1: Int, _ axis2: Int, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_swapaxes(&result, ctx, axis1.int32, axis2.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Take elements along an axis.
    ///
    /// The elements are taken from `indices` along the specified axis.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// /// produces a [3, 2] result with the index 0 and 2 columns from the array
    /// let col = array.take([0, 2], axis: 1)
    /// ```
    ///
    /// - Parameters:
    ///     - indices: array of indices
    ///     - axis: axis to take from
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - ``take(_:stream:)``
    /// - ``take(_:_:axis:stream:)``
    public func take(_ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray
    {
        var result = mlx_array_new()
        mlx_take_axis(&result, ctx, indices.ctx, axis.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Take elements from flattened 1-D array.
    ///
    /// - Parameters:
    ///     - indices: array of indices
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - ``take(_:axis:stream:)``
    /// - ``take(_:_:axis:stream:)``
    public func take(_ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        let input = self.reshaped([-1], stream: stream)
        var result = mlx_array_new()
        mlx_take(&result, input.ctx, indices.ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Transpose the dimensions of the array.
    ///
    /// - Parameters:
    ///     - axes: Specifies the source axis for each axis in the new array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``transposed(axis:stream:)``
    /// - ``transposed(stream:)``
    /// - ``transposed(_:axes:stream:)``
    public func transposed(axes: some Collection<Int>, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_transpose_axes(&result, ctx, axes.asInt32, axes.count, stream.ctx)
        return MLXArray(result)
    }

    /// Transpose the dimensions of the array.
    ///
    /// - Parameters:
    ///     - axes: Specifies the source axis for each axis in the new array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``transposed(axes:stream:)``
    /// - ``transposed(axis:stream:)``
    /// - ``transposed(stream:)``
    /// - ``transposed(_:axes:stream:)``
    public func transposed(_ axes: Int..., stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_transpose_axes(&result, ctx, axes.asInt32, axes.count, stream.ctx)
        return MLXArray(result)
    }

    /// Transpose the dimensions of the array.
    ///
    /// This swaps the position of the first dimension with the given axis.
    ///
    /// - Parameters:
    ///     - axis: axis to swap with the first dimension
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``transposed(axes:stream:)``
    /// - ``transposed(stream:)``
    /// - ``transposed(_:axes:stream:)``
    public func transposed(axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_transpose_axes(&result, ctx, [axis.int32], 1, stream.ctx)
        return MLXArray(result)
    }

    /// Transpose the dimensions of the array.
    ///
    /// With no axes specified this will reverse the axes in the array.
    ///
    /// - Parameters:
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:shapes>
    /// - ``transposed(axes:stream:)``
    /// - ``transposed(axis:stream:)``
    /// - ``transposed(_:axes:stream:)``
    public func transposed(stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_transpose(&result, ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Transpose the dimensions of the array.
    ///
    /// Cover for ``transposed(stream:)``
    public var T: MLXArray {
        transposed()
    }

    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - axes: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``variance(axis:keepDims:ddof:stream:)``
    /// - ``variance(keepDims:ddof:stream:)``
    /// - ``variance(_:axes:keepDims:ddof:stream:)``
    public func variance(
        axes: some Collection<Int>, keepDims: Bool = false, ddof: Int = 0,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_var_axes(&result, ctx, axes.asInt32, axes.count, keepDims, ddof.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - axis: axes to reduce over
    ///     - keepDims: if `true` keep the reduces axis as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``variance(axes:keepDims:ddof:stream:)``
    /// - ``variance(keepDims:ddof:stream:)``
    /// - ``variance(_:axes:keepDims:ddof:stream:)``
    public func variance(
        axis: Int, keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_var_axis(&result, ctx, axis.int32, keepDims, ddof.int32, stream.ctx)
        return MLXArray(result)
    }

    /// Compute the variance(s) over the given axes
    ///
    /// - Parameters:
    ///     - keepDims: if `true` keep the reduces axes as singleton dimensions
    ///     - ddof: the divisor to compute the variance is `N - ddof`
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:reduction>
    /// - ``variance(axes:keepDims:ddof:stream:)``
    /// - ``variance(axis:keepDims:ddof:stream:)``
    /// - ``variance(_:axes:keepDims:ddof:stream:)``
    public func variance(keepDims: Bool = false, ddof: Int = 0, stream: StreamOrDevice = .default)
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_var(&result, ctx, keepDims, ddof.int32, stream.ctx)
        return MLXArray(result)
    }

    /// View the array as a different type.
    ///
    /// The output array will change along the last axis if the input array's
    /// type and the output array's type do not have the same size.
    ///
    /// Note: the view op does not imply that the input and output arrays share
    /// their underlying data. The view only gaurantees that the binary
    /// representation of each element (or group of elements) is the same.
    ///
    /// - Parameters:
    ///   - dtype: type to change to
    ///   - stream: stream or device to evaluate on
    /// - Returns: array with the new type
    public func view(dtype: DType, stream: StreamOrDevice = .default) -> MLXArray {
        var result = mlx_array_new()
        mlx_view(&result, ctx, dtype.cmlxDtype, stream.ctx)
        return MLXArray(result)
    }

    /// Force an array to be row contiguous. Copy if necessary.
    ///
    /// - Parameters:
    ///   - allowColMajor: consider column major as contiguous and don’t copy
    ///   - stream: stream or device to evaluate on
    /// - Returns: the row or col contiguous output.
    public func contiguous(
        allowColMajor: Bool = false, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_contiguous(&result, ctx, allowColMajor, stream.ctx)
        return MLXArray(result)
    }
}
