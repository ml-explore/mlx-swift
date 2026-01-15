// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Numerics

public final class MLXArray {

    /// Internal pointer to the mlx-c wrapper on `mlx::core::array`, used with `Cmlx` interop.
    public internal(set) var ctx: mlx_array

    /// Initialize with the given +1 context (transfer ownership).
    ///
    /// This initializer is for `Cmlx` interoperation.
    public init(_ ctx: consuming mlx_array) {
        // We don't have lifecycle control over the MLX system but all interesting
        // paths will come through here -- make sure the error handler is installed.
        initError()
        self.ctx = ctx
    }

    /// return the equivalent of a `.none` MLXArray (for the C API).
    ///
    /// Not called `.none` to avoid ambiguity with `Optional`.  This can be used
    /// to pass an optional ``MLXArray`` as a non-optional (but possibly empty/null)
    /// `mlx_array`:
    ///
    /// ```swift
    /// mlx_func((freqs ?? .mlxNone).ctx)
    /// ```
    public static var mlxNone: MLXArray {
        .init(mlx_array_new())
    }

    deinit {
        mlx_array_free(ctx)
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
    public var dtype: DType { DType(mlx_array_dtype(ctx)) }

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

    /// Dimensions of the 2d array as a tuple.
    ///
    /// ```swift
    /// let (w, h) = array.shape2
    /// ```
    public var shape2: (Int, Int) {
        let ndim = mlx_array_ndim(ctx)
        precondition(ndim == 2)
        let cShape = mlx_array_shape(ctx)!
        return (Int(cShape[0]), Int(cShape[1]))
    }

    /// Dimensions of the 3d array as a tuple.
    ///
    /// ```swift
    /// let (w, h, c) = array.shape3
    /// ```
    public var shape3: (Int, Int, Int) {
        let ndim = mlx_array_ndim(ctx)
        precondition(ndim == 3)
        let cShape = mlx_array_shape(ctx)!
        return (Int(cShape[0]), Int(cShape[1]), Int(cShape[2]))
    }

    /// Dimensions of the 4d array as a tuple.
    ///
    /// ```swift
    /// let (b, w, h, c) = array.shape4
    /// ```
    public var shape4: (Int, Int, Int, Int) {
        let ndim = mlx_array_ndim(ctx)
        precondition(ndim == 4)
        let cShape = mlx_array_shape(ctx)!
        return (Int(cShape[0]), Int(cShape[1]), Int(cShape[2]), Int(cShape[3]))
    }

    /// Strides of the array.  Note: do not use this as it changes
    /// before and after evaluation.  See also ``asData(access:)``
    /// and ``MLXArray/MLXArrayData/strides``.
    @available(*, deprecated, message: "Do not use -- see asData(access:)")
    public var strides: [Int] {
        let ndim = mlx_array_ndim(ctx)
        guard ndim > 0 else { return [] }
        let strides = mlx_array_strides(ctx)!
        return (0 ..< ndim).map { Int(strides[$0]) }
    }

    /// Strides of the array backing.
    ///
    /// Note: this is only stable once the array is evaluated.
    var internalStrides: [Int] {
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

    /// specialized conversion between integer types -- see ``item(_:)``
    private func itemInt() -> Int {
        precondition(self.size == 1)
        eval()

        switch self.dtype {
        case .bool:
            var r = false
            mlx_array_item_bool(&r, self.ctx)
            return r ? 1 : 0
        case .uint8:
            var r: UInt8 = 0
            mlx_array_item_uint8(&r, self.ctx)
            return Int(r)
        case .uint16:
            var r: UInt16 = 0
            mlx_array_item_uint16(&r, self.ctx)
            return Int(r)
        case .uint32:
            var r: UInt32 = 0
            mlx_array_item_uint32(&r, self.ctx)
            return Int(r)
        case .uint64:
            var r: UInt64 = 0
            mlx_array_item_uint64(&r, self.ctx)
            return Int(r)
        case .int8:
            var r: Int8 = 0
            mlx_array_item_int8(&r, self.ctx)
            return Int(r)
        case .int16:
            var r: Int16 = 0
            mlx_array_item_int16(&r, self.ctx)
            return Int(r)
        case .int32:
            var r: Int32 = 0
            mlx_array_item_int32(&r, self.ctx)
            return Int(r)
        case .int64:
            var r: Int64 = 0
            mlx_array_item_int64(&r, self.ctx)
            return Int(r)

        default:
            fatalError("itemInt expected an integer dtype: \(self.dtype)")
        }
    }

    /// specialized conversion between integer types -- see ``item(_:)``
    private func itemUInt() -> UInt {
        precondition(self.size == 1)
        eval()

        switch self.dtype {
        case .bool:
            var r = false
            mlx_array_item_bool(&r, self.ctx)
            return r ? 1 : 0
        case .uint8:
            var r: UInt8 = 0
            mlx_array_item_uint8(&r, self.ctx)
            return UInt(r)
        case .uint16:
            var r: UInt16 = 0
            mlx_array_item_uint16(&r, self.ctx)
            return UInt(r)
        case .uint32:
            var r: UInt32 = 0
            mlx_array_item_uint32(&r, self.ctx)
            return UInt(r)
        case .uint64:
            var r: UInt64 = 0
            mlx_array_item_uint64(&r, self.ctx)
            return UInt(r)
        case .int8:
            var r: Int8 = 0
            mlx_array_item_int8(&r, self.ctx)
            return UInt(r)
        case .int16:
            var r: Int16 = 0
            mlx_array_item_int16(&r, self.ctx)
            return UInt(r)
        case .int32:
            var r: Int32 = 0
            mlx_array_item_int32(&r, self.ctx)
            return UInt(r)
        case .int64:
            var r: Int64 = 0
            mlx_array_item_int64(&r, self.ctx)
            return UInt(r)

        default: fatalError("itemUInt expected an integer dtype: \(self.dtype)")
        }
    }

    /// specialized conversion between float types -- see ``item(_:)``
    private func itemFloat() -> Float {
        precondition(self.size == 1)
        eval()

        switch self.dtype {
        #if !arch(x86_64)
            case .float16:
                var r: Float16 = 0
                mlx_array_item_float16(&r, self.ctx)
                return Float(r)
        #endif
        case .float32:
            var r: Float32 = 0
            mlx_array_item_float32(&r, self.ctx)
            return Float(r)

        case .float64:
            var r: Float64 = 0
            mlx_array_item_float64(&r, self.ctx)
            return Float(r)

        default: fatalError("itemFloat expected a floating point dtype: \(self.dtype)")
        }
    }

    private func itemDouble() -> Double {
        precondition(self.size == 1)
        eval()

        switch self.dtype {
        #if !arch(x86_64)
            case .float16:
                var r: Float16 = 0
                mlx_array_item_float16(&r, self.ctx)
                return Double(r)
        #endif
        case .float32:
            var r: Float32 = 0
            mlx_array_item_float32(&r, self.ctx)
            return Double(r)

        case .float64:
            var r: Float64 = 0
            mlx_array_item_float64(&r, self.ctx)
            return Double(r)

        default: fatalError("itemFloat expected a floating point dtype: \(self.dtype)")
        }
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
        precondition(self.size == 1)
        eval()

        // special cases for reading integers and floats from (roughly)
        // same typed arrays -- this avoids doing a conversion which
        // might end up as an unexpected operation that would mess up
        // async evaluation
        switch type {
        case is Int.Type, is Int8.Type, is Int16.Type, is Int32.Type, is Int64.Type:
            if self.dtype.isInteger {
                switch type {
                case is Int.Type: return Int(itemInt()) as! T
                case is Int8.Type: return Int8(itemInt()) as! T
                case is Int16.Type: return Int16(itemInt()) as! T
                case is Int32.Type: return Int32(itemInt()) as! T
                case is Int64.Type: return Int64(itemInt()) as! T
                default:
                    // fall through to default handling
                    break
                }
            }
        case is UInt8.Type, is UInt16.Type, is UInt32.Type, is UInt64.Type, is UInt.Type:
            if self.dtype.isInteger {
                switch type {
                case is UInt8.Type: return UInt8(itemUInt()) as! T
                case is UInt16.Type: return UInt16(itemUInt()) as! T
                case is UInt32.Type: return UInt32(itemUInt()) as! T
                case is UInt64.Type: return UInt64(itemUInt()) as! T
                case is UInt.Type: return UInt(itemUInt()) as! T
                default:
                    // fall through to default handling
                    break
                }
            }
        #if !arch(x86_64)
            case is Float.Type, is Float32.Type, is Float16.Type, is Double.Type:
                switch self.dtype {
                case .float16, .float32, .float64:
                    switch type {
                    case is Float.Type: return Float(itemFloat()) as! T
                    case is Float64.Type: return itemDouble() as! T
                    case is Float32.Type: return Float32(itemFloat()) as! T
                    case is Float16.Type: return Float16(itemFloat()) as! T
                    default:
                        // fall through to default handling
                        break
                    }
                default:
                    break
                }
        #endif
        default:
            break
        }

        // default handling -- convert the type if needed
        if type.dtype != self.dtype {
            return self.asType(type).item(type)
        }

        switch type {
        case is Bool.Type:
            var r: Bool = false
            mlx_array_item_bool(&r, self.ctx)
            return r as! T
        case is UInt8.Type:
            var r: UInt8 = 0
            mlx_array_item_uint8(&r, self.ctx)
            return r as! T
        case is UInt16.Type:
            var r: UInt16 = 0
            mlx_array_item_uint16(&r, self.ctx)
            return r as! T
        case is UInt32.Type:
            var r: UInt32 = 0
            mlx_array_item_uint32(&r, self.ctx)
            return r as! T
        case is UInt64.Type:
            var r: UInt64 = 0
            mlx_array_item_uint64(&r, self.ctx)
            return r as! T
        case is Int8.Type:
            var r: Int8 = 0
            mlx_array_item_int8(&r, self.ctx)
            return r as! T
        case is Int16.Type:
            var r: Int16 = 0
            mlx_array_item_int16(&r, self.ctx)
            return r as! T
        case is Int32.Type:
            var r: Int32 = 0
            mlx_array_item_int32(&r, self.ctx)
            return r as! T
        case is Int64.Type:
            var r: Int64 = 0
            mlx_array_item_int64(&r, self.ctx)
            return r as! T
        case is Int.Type:
            var r: Int64 = 0
            mlx_array_item_int64(&r, self.ctx)
            return Int(r) as! T
        #if !arch(x86_64)
            case is Float16.Type:
                var r: Float16 = 0
                mlx_array_item_float16(&r, self.ctx)
                return r as! T
        #endif
        case is Float32.Type:
            var r: Float32 = 0
            mlx_array_item_float32(&r, self.ctx)
            return r as! T
        case is Float.Type:
            var r: Float = 0
            mlx_array_item_float32(&r, self.ctx)
            return r as! T
        case is Double.Type:
            var r: Float64 = 0
            mlx_array_item_float64(&r, self.ctx)
            return r as! T
        case is Complex<Float32>.Type:
            // mlx_array_item_complex64() isn't visible in swift so read the array
            // contents.  call self.eval() as this doesn't end up in item()
            self.eval()
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
    ///   - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func asType(_ type: DType, stream: StreamOrDevice = .default) -> MLXArray {
        guard type != self.dtype else { return self }
        var result = mlx_array_new()
        mlx_astype(&result, ctx, type.cmlxDtype, stream.ctx)
        return MLXArray(result)
    }

    /// Create a new `MLXArray` with the contents converted to the given type, e.g. `Float.self`.
    /// - Parameters:
    ///   - type: type to convert to
    ///   - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:conversion>
    public func asType(_ type: (some HasDType).Type, stream: StreamOrDevice = .default) -> MLXArray
    {
        asType(type.dtype, stream: stream)
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
        _ = evalLock.withLock {
            mlx_array_eval(ctx)
        }
    }

    /// Replace the contents with a reference to a new array (INTERNAL).
    ///
    /// Note: this is an implementation detail and only visible because of the need to call it from
    /// other `mlx-swift` modules.
    public func _updateInternal(_ array: MLXArray) {
        mlx_array_set(&self.ctx, array.ctx)
    }

    /// Internal function for copying the backing `mlx::core::array` context.
    func copyContext() -> MLXArray {
        var new = mlx_array_new()
        mlx_array_set(&new, self.ctx)
        return MLXArray(new)
    }

    /// Used to apply update at given indices.
    ///
    /// An assignment through indices `array[indices]` will produce
    /// a result where each index will only be updated once.  For example:
    ///
    /// ```swift
    /// // this references each index twice
    /// let idx = MLXArray([0, 1, 0, 1])
    ///
    /// let a1 = MLXArray([0, 0])
    /// a1[idx] += 1
    /// assertEqual(a1, MLXArray([1, 1]))
    ///
    /// // this will update 0 and 1 twice
    /// var a2 = MLXArray([0, 0])
    /// a2 = a2.at[idx].add(1)
    /// assertEqual(a2, MLXArray([2, 2]))
    /// ```
    ///
    /// This is because the assignment through `array[indices]` writes
    /// a sub-array of `array` rather than performing the operation on each
    /// resolved index.
    ///
    /// The `at` property produces an intermediate value that can take a subscript
    /// `[]` and produce an ``ArrayAtIndices`` that has several methods to
    /// update values.
    ///
    /// ### See Also
    /// - ``subscript(indices:stream:)``
    /// - ``ArrayAtIndices``
    public var at: ArrayAt { ArrayAt(array: self) }
}

extension MLXArray: Updatable, Evaluatable {
    public func innerState() -> [MLXArray] {
        [self]
    }
}

extension MLXArray: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        mlx_array_tostring(&s, ctx)
        defer { mlx_string_free(s) }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
