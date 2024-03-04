// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Numerics

public final class MLXArray {

    /// Internal pointer to the mlx-c wrapper on `mlx::core::array`, used with `Cmlx` interop.
    public package(set) var ctx: mlx_array

    /// Initialize with the given +1 context (transfer ownership).
    ///
    /// This initializer is for `Cmlx` interoperation.
    public init(_ ctx: mlx_array) {
        self.ctx = ctx
    }

    deinit {
        mlx_free(ctx)
    }

    /// Number of bytes per element
    public var itemSize: Int { mlx_array_itemsize(ctx) }

    /// Total number of elements in the array
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.size)
    /// // 12
    /// ```
    public var size: Int { mlx_array_size(ctx) }

    /// Number of elements in the 0th dimension.
    ///
    /// For example, these would be equivalent:
    ///
    /// ```swift
    /// for row in array {
    ///     ...
    /// }
    ///
    /// for i in 0..<array.count {
    ///     let row = array[i]
    ///     ...
    /// }
    /// ```
    public var count: Int { dim(0) }

    /// Number of bytes in the array.
    public var nbytes: Int { mlx_array_nbytes(ctx) }

    /// Number of dimensions in the array.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.ndim)
    /// // 2
    /// ```
    public var ndim: Int { mlx_array_ndim(ctx) }

    /// Data type of the elements in the array.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.dtype)
    /// // .int64 (aka Int.dtype)
    /// ```
    public var dtype: DType { DType(mlx_array_get_dtype(ctx)) }

    /// Dimensions of the array.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.shape)
    /// // [3, 4]
    /// ```
    public var shape: [Int] {
        let ndim = mlx_array_ndim(ctx)
        guard ndim > 0 else { return [] }
        let cShape = mlx_array_shape(ctx)!
        return (0 ..< ndim).map { Int(cShape[$0]) }
    }

    /// Strides of the array.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.strides)
    /// // [4, 1]
    /// ```
    public var strides: [Int] {
        let ndim = mlx_array_ndim(ctx)
        guard ndim > 0 else { return [] }
        let strides = mlx_array_strides(ctx)!
        return (0 ..< ndim).map { Int(strides[$0]) }
    }

    /// Return the scalar value of the array.
    ///
    /// It is a contract violation to call this on an array with more than one element
    /// or to read a type other than the `dtype`.
    ///
    /// ```swift
    /// let array = MLXArray([3.5, 4.5])
    ///
    /// // 4.5
    /// let value: Float = array[1].item()
    /// ```
    public func item<T: HasDType>() -> T {
        item(T.self)
    }

    /// Return the scalar value of the array.
    ///
    /// It is a contract violation to call this on an array with more than one element.
    /// If the `type` does not match the `dtype` this will convert the type first.
    ///
    /// ```swift
    /// let array = MLXArray([3.5, 4.5])
    ///
    /// // 4.5
    /// let value = array[1].item(Float.self)
    /// ```
    public func item<T: HasDType>(_ type: T.Type) -> T {
        self.eval()

        var array_ctx = self.ctx
        var free = false
        if type.dtype != self.dtype {
            array_ctx = mlx_astype(self.ctx, type.dtype.cmlxDtype, StreamOrDevice.default.ctx)
            mlx_array_eval(array_ctx)
            free = true
        }

        // can't do it inside the else as it will free at the end of the block
        defer { if free { mlx_free(array_ctx) } }

        switch type {
        case is Bool.Type: return mlx_array_item_bool(array_ctx) as! T
        case is UInt8.Type: return mlx_array_item_uint8(array_ctx) as! T
        case is UInt16.Type: return mlx_array_item_uint16(array_ctx) as! T
        case is UInt32.Type: return mlx_array_item_uint32(array_ctx) as! T
        case is UInt64.Type: return mlx_array_item_uint64(array_ctx) as! T
        case is Int8.Type: return mlx_array_item_int8(array_ctx) as! T
        case is Int16.Type: return mlx_array_item_int16(array_ctx) as! T
        case is Int32.Type: return mlx_array_item_int32(array_ctx) as! T
        case is Int64.Type: return mlx_array_item_int64(array_ctx) as! T
        case is Int.Type: return Int(mlx_array_item_int64(array_ctx)) as! T
        #if !arch(x86_64)
            case is Float16.Type: return mlx_array_item_float16(array_ctx) as! T
        #endif
        case is Float32.Type: return mlx_array_item_float32(array_ctx) as! T
        case is Float.Type: return mlx_array_item_float32(array_ctx) as! T
        case is Complex<Float32>.Type:
            // mlx_array_item_complex64() isn't visible in swift so read the array
            // contents
            let ptr = UnsafePointer<Complex<Float32>>(mlx_array_data_complex64(ctx))!
            return ptr.pointee as! T
        default:
            fatalError("Unable to get item() as \(type)")
        }
    }

    /// Read a dimension of the array.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.dim(1))
    /// // 4
    /// ```
    public func dim(_ dim: Int) -> Int {
        Int(mlx_array_dim(ctx, MLX.resolve(axis: dim, ndim: mlx_array_ndim(ctx)).int32))
    }

    /// Read a dimension of the array.
    ///
    /// Convenience override for `Int32`.
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// let index = Int32(1)
    /// print(array.dim(index))
    /// // 4
    /// ```
    func dim(_ dim: Int32) -> Int32 {
        mlx_array_dim(ctx, MLX.resolve(axis: Int(dim), ndim: mlx_array_ndim(ctx)).int32)
    }

    /// Create a new `MLXArray` with the contents converted to the given ``DType``.
    /// - Parameters:
    ///   - type: type to convert to
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func asType(_ type: DType, stream: StreamOrDevice = .default) -> MLXArray {
        guard type != self.dtype else { return self }
        return MLXArray(mlx_astype(ctx, type.cmlxDtype, stream.ctx))
    }

    /// Create a new `MLXArray` with the contents converted to the given type, e.g. `Float.self`.
    /// - Parameters:
    ///   - type: type to convert to
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func asType<T: HasDType>(_ type: T.Type, stream: StreamOrDevice = .default) -> MLXArray {
        asType(T.dtype, stream: stream)
    }

    /// Return the contents as a single contiguous 1d `Swift.Array`.
    ///
    /// Note: because the number of dimensions is dynamic, this cannot produce a multi-dimensional
    /// array.
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        self.eval()

        var array_ctx = self.ctx
        var free = false
        if type.dtype != self.dtype {
            array_ctx = mlx_astype(self.ctx, type.dtype.cmlxDtype, StreamOrDevice.default.ctx)
            mlx_array_eval(array_ctx)
            free = true
        }

        // can't do it inside the else as it will free at the end of the block
        defer { if free { mlx_free(array_ctx) } }

        func convert(_ ptr: UnsafePointer<T>) -> [T] {
            Array(UnsafeBufferPointer(start: ptr, count: self.size))
        }

        switch type {
        case is Bool.Type: return convert(mlx_array_data_bool(array_ctx) as! UnsafePointer<T>)
        case is UInt8.Type: return convert(mlx_array_data_uint8(array_ctx) as! UnsafePointer<T>)
        case is UInt16.Type: return convert(mlx_array_data_uint16(array_ctx) as! UnsafePointer<T>)
        case is UInt32.Type: return convert(mlx_array_data_uint32(array_ctx) as! UnsafePointer<T>)
        case is UInt64.Type: return convert(mlx_array_data_uint64(array_ctx) as! UnsafePointer<T>)
        case is Int8.Type: return convert(mlx_array_data_int8(array_ctx) as! UnsafePointer<T>)
        case is Int16.Type: return convert(mlx_array_data_int16(array_ctx) as! UnsafePointer<T>)
        case is Int32.Type: return convert(mlx_array_data_int32(array_ctx) as! UnsafePointer<T>)
        case is Int64.Type: return convert(mlx_array_data_int64(array_ctx) as! UnsafePointer<T>)
        case is Int.Type:
            // Int and Int64 are the same bits but distinct types. coerce pointers as needed
            let pointer = mlx_array_data_int64(array_ctx)
            let bufferPointer = UnsafeBufferPointer(start: pointer, count: self.size)
            return bufferPointer.withMemoryRebound(to: Int.self) { buffer in
                Array(buffer) as! [T]
            }
        #if !arch(x86_64)
            case is Float16.Type:
                return convert(mlx_array_data_float16(array_ctx) as! UnsafePointer<T>)
        #endif
        case is Float32.Type: return convert(mlx_array_data_float32(array_ctx) as! UnsafePointer<T>)
        case is Float.Type: return convert(mlx_array_data_float32(array_ctx) as! UnsafePointer<T>)
        case is Complex<Float32>.Type:
            let ptr = UnsafeBufferPointer(
                start: UnsafePointer<Complex<Float32>>(mlx_array_data_complex64(ctx)),
                count: self.size)
            return Array(ptr) as! [T]
        default:
            fatalError("Unable to get item() as \(type)")
        }
    }

    /// Convert the real array into a ``DType/complex64`` imaginary part.
    ///
    /// This is equivalent to the following in python:
    ///
    /// ```python
    /// result = 1j * array
    /// ```
    ///
    /// ### See Also
    /// - ``init(real:imaginary:)``
    /// - ``realPart(stream:)``
    /// - ``imaginaryPart(stream:)``
    /// - <doc:conversion>
    public func asImaginary(stream: StreamOrDevice = .default) -> MLXArray {
        precondition(!dtype.isComplex)
        let i = MLXArray(real: 0, imaginary: 1)
        return self * i
    }

    /// Extract the real part of a ``DType/complex64`` array.
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func realPart(stream: StreamOrDevice = .default) -> MLXArray {
        precondition(dtype.isComplex)
        return asType(Float.self)
    }

    /// Extract the imaginary part of a ``DType/complex64`` array.
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func imaginaryPart(stream: StreamOrDevice = .default) -> MLXArray {
        precondition(dtype.isComplex)
        let i = MLXArray(real: 0, imaginary: 1)
        return (self / i).asType(.float32)
    }

    /// Evaluate the array.
    ///
    /// MLX is lazy and arrays are not fully realized until they are evaluated.  This method is typically
    /// not needed as all reads ensure the contents are evaluated.
    public func eval() {
        mlx_array_eval(ctx)
    }

    /// Replace the contents with a reference to a new array.
    public func update(_ array: MLXArray) {
        if array.ctx != self.ctx {
            mlx_retain(array.ctx)
            mlx_free(ctx)
            self.ctx = array.ctx
        }
    }

    public func copy() -> MLXArray {
        mlx_retain(ctx)
        return MLXArray(ctx)
    }
}

extension MLXArray: Updatable, Evaluatable {
    public func innerState() -> [MLXArray] {
        [self]
    }
}

extension MLXArray: CustomStringConvertible {
    public var description: String {
        mlx_describe(ctx) ?? String(describing: type(of: self))
    }
}
