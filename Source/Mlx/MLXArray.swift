import Foundation
import Cmlx

public final class MLXArray {
    
    var ctx: OpaquePointer!

    /// initialize with the given +1 context (transfer ownership)
    init(_ ctx: mlx_array) {
        self.ctx = ctx
    }
    
    deinit {
        mlx_free(ctx)
    }
        
    /// Number of bytes per element
    public var itemSize: Int { mlx_array_itemsize(ctx) }
    
    /// Total number of elements in the array
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.size)
    /// // 12
    /// ```
    public var size: Int { mlx_array_size(ctx) }
    
    /// Number of elements in the 0th dimension.
    ///
    /// For example, these would be equivalent:
    ///
    /// ```
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
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.ndim)
    /// // 2
    /// ```
    public var ndim: Int { mlx_array_ndim(ctx) }
    
    /// Data type of the elements in the array.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.dtype)
    /// // .int64 (aka Int.dtype)
    /// ```
    public var dtype: DType { DType(mlx_array_get_dtype(ctx)) }
    
    /// Dimensions of the array.
    ///
    /// ```
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
    
    /// Return the scalar value of the array.
    ///
    /// It is a contract violation to call this on an array with more than one element
    /// or to read a type other than the `dtype`.
    ///
    /// ```
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
    /// It is a contract violation to call this on an array with more than one element
    /// or to read a type other than the `dtype`.
    ///
    /// ```
    /// let array = MLXArray([3.5, 4.5])
    ///
    /// // 4.5
    /// let value = array[1].item(Float.self)
    /// ```
    public func item<T: HasDType>(_ type: T.Type) -> T {
        precondition(T.dtype == self.dtype, "\(T.dtype) != \(self.dtype)")
        
        switch type {
        case is Bool.Type: return mlx_array_item_bool(ctx) as! T
        case is UInt8.Type: return mlx_array_item_uint8(ctx) as! T
        case is UInt16.Type: return mlx_array_item_uint16(ctx) as! T
        case is UInt32.Type: return mlx_array_item_uint32(ctx) as! T
        case is UInt64.Type: return mlx_array_item_uint64(ctx) as! T
        case is Int8.Type: return mlx_array_item_int8(ctx) as! T
        case is Int16.Type: return mlx_array_item_int16(ctx) as! T
        case is Int32.Type: return mlx_array_item_int32(ctx) as! T
        case is Int64.Type: return mlx_array_item_int64(ctx) as! T
        case is Int.Type: return Int(mlx_array_item_int64(ctx)) as! T
        case is Float16.Type: return mlx_array_item_float16(ctx) as! T
        case is Float32.Type: return mlx_array_item_float32(ctx) as! T
        case is Float.Type: return mlx_array_item_float32(ctx) as! T
        default:
            fatalError("Unable to get item() as \(type)")
        }
    }
        
    /// Read a dimension of the array.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    /// print(array.dim(1))
    /// // 4
    /// ```
    public func dim(_ dim: Int) -> Int {
        Int(mlx_array_dim(ctx, dim.int32))
    }
    
    /// Read a dimension of the array.
    ///
    /// Convenience override for `Int32`.
    ///
    /// ```
    /// let array = MLXArray(0 ..< 12, [3, 4])
    ///
    /// let index = Int32(1)
    /// print(array.dim(index))
    /// // 4
    /// ```
    public func dim(_ dim: Int32) -> Int32 {
        mlx_array_dim(ctx, dim)
    }
    
    public func asType(_ type: DType, stream: StreamOrDevice = .default) -> MLXArray {
        guard type != self.dtype else { return self }
        return MLXArray(mlx_astype(ctx, type.cmlxDtype, stream.ctx))
    }
    
    public func asType<T: HasDType>(_ type: T.Type, stream: StreamOrDevice = .default) -> MLXArray {
        asType(T.dtype, stream: stream)
    }
    
    /// Return the contents as a single contiguous 1d `Swift.Array`.
    ///
    /// Note: because the number of dimensions is dynamic, this cannot produce a multi-dimensional
    /// array.
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        precondition(T.dtype == self.dtype, "\(T.dtype) != \(self.dtype)")
        
        // make sure the contents are realized
        mlx_array_eval(ctx)

        func convert(_ ptr: UnsafePointer<T>) -> [T] {
            Array(UnsafeBufferPointer(start: ptr, count: self.size))
        }
        
        switch type {
        case is Bool.Type: return convert(mlx_array_data_bool(ctx) as! UnsafePointer<T>)
        case is UInt8.Type: return convert(mlx_array_data_uint8(ctx) as! UnsafePointer<T>)
        case is UInt16.Type: return convert(mlx_array_data_uint16(ctx) as! UnsafePointer<T>)
        case is UInt32.Type: return convert(mlx_array_data_uint32(ctx) as! UnsafePointer<T>)
        case is UInt64.Type: return convert(mlx_array_data_uint64(ctx) as! UnsafePointer<T>)
        case is Int8.Type: return convert(mlx_array_data_int8(ctx) as! UnsafePointer<T>)
        case is Int16.Type: return convert(mlx_array_data_int16(ctx) as! UnsafePointer<T>)
        case is Int32.Type: return convert(mlx_array_data_int32(ctx) as! UnsafePointer<T>)
        case is Int64.Type: return convert(mlx_array_data_int64(ctx) as! UnsafePointer<T>)
        case is Int.Type:
            // Int and Int64 are the same bits but distinct types. coerce pointers as needed
            let pointer = mlx_array_data_int64(ctx)
            let bufferPointer = UnsafeBufferPointer(start: pointer, count: self.size)
            return bufferPointer.withMemoryRebound(to: Int.self) { buffer in
                Array(buffer) as! [T]
            }
        case is Float16.Type: return convert(mlx_array_data_float16(ctx) as! UnsafePointer<T>)
        case is Float32.Type: return convert(mlx_array_data_float32(ctx) as! UnsafePointer<T>)
        case is Float.Type: return convert(mlx_array_data_float32(ctx) as! UnsafePointer<T>)
        default:
            fatalError("Unable to get item() as \(type)")
        }
    }
    
    /// Evaluate the array.
    ///
    /// MLX is lazy and arrays are not fully realized until they are evaluated.  This method is typically
    /// not needed as all reads ensure the contents are evaluated.
    public func eval() {
        mlx_array_eval(ctx)
    }
}

extension MLXArray: CustomStringConvertible {
    public var description: String {
        if let cDsc = Cmlx.mlx_tostring(UnsafeMutableRawPointer(ctx)) {
            defer { free(cDsc) }
            return String(cString: cDsc)
        } else {
            return String(describing: type(of: self))
        }
    }
}

