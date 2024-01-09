import Foundation
import Cmlx

public final class MLXArray {
    
    var ctx: OpaquePointer!

    /// initialize with the given +1 context (transfer ownership)
    init(_ ctx: mlx_array) {
        self.ctx = ctx
    }
    
    deinit {
        Cmlx.mlx_free(UnsafeMutableRawPointer(ctx))
    }
        
    public var itemSize: Int { mlx_array_itemsize(ctx) }
    public var count: Int { mlx_array_size(ctx) }
    public var nbytes: Int { mlx_array_nbytes(ctx) }
    public var ndim: Int { mlx_array_ndim(ctx) }
    public var dtype: DType { DType(mlx_array_get_dtype(ctx)) }
    public var shape: [Int] { 
        let ndim = mlx_array_ndim(ctx)
        guard ndim > 0 else { return [] }
        let cShape = mlx_array_shape(ctx)!
        return (0 ..< ndim).map { Int(cShape[$0]) }
    }
    
    public func item<T: HasDType>() -> T {
        item(T.self)
    }
    
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
    
    public func dim(_ dim: Int) -> Int {
        Int(mlx_array_dim(ctx, dim.int32))
    }
    
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

