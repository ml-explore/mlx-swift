import Cmlx
import Darwin

extension String: Error {}

public enum DType {
    case bool
    case uint8
    case uint16
    case uint32
    case uint64
    case int8
    case int16
    case int32
    case int64
    case float16
    case float32
    case bfloat16
    case complex64
}

public final class Array {
    let ctx: OpaquePointer!

    public init(_ ctx_: mlx_array) {
        ctx = ctx_
    }
    public init(_ value: Bool) {
        ctx = Cmlx.mlx_array_from_bool(value)
    }
    public init(_ value: Int32) {
        ctx = Cmlx.mlx_array_from_int(value)
    }
    public init(_ value: Float) {
        ctx = Cmlx.mlx_array_from_float(value)
    }
    public init<T>(_ data: Swift.Array<T>, _ shape: Swift.Array<Int32>) {
        let mlxDtype = Array.toCmlxDtype(type: type(of: data).Element.self)
        ctx = data.withUnsafeBufferPointer({ dataPtr in
                shape.withUnsafeBufferPointer({ shapePtr in
                  Cmlx.mlx_array_from_data(dataPtr.baseAddress, shapePtr.baseAddress, Int32(shape.count), mlxDtype)
                })
              })
    }
    static func toCmlxDtype<T>(type: T.Type) -> mlx_array_dtype {
        switch type {
        case is Bool.Type:
            return Cmlx.MLX_BOOL;
        case is UInt8.Type:
            return Cmlx.MLX_UINT8;
        case is UInt16.Type:
            return Cmlx.MLX_UINT16;
        case is UInt32.Type:
            return Cmlx.MLX_UINT32;
        case is UInt64.Type:
            return Cmlx.MLX_UINT64;
        case is Int8.Type:
            return Cmlx.MLX_INT8;
        case is Int16.Type:
            return Cmlx.MLX_INT16;
        case is Int32.Type:
            return Cmlx.MLX_INT32;
        case is Int64.Type:
            return Cmlx.MLX_INT64;
        case is Float.Type:
            return Cmlx.MLX_FLOAT32;
        default:
            fatalError("MLX: internal error: could not convert type <" + String(describing: type) + "> to MLX type")
        }
    }
    public func itemsize() -> Int {
        return mlx_array_itemsize(ctx);
    }
    public func size() -> Int {
        return mlx_array_size(ctx);
    }
    public func nbytes() -> Int {
        return mlx_array_nbytes(ctx);
    }
    public func ndim() -> Int {
        return mlx_array_ndim(ctx);
    }
    public func dim(_ dim : Int32) -> Int32 {
        return mlx_array_dim(ctx, dim);
    }
    public func dtype() -> DType {
        let cDtype = mlx_array_get_dtype(ctx)
        switch(cDtype) {
        case MLX_BOOL:
            return DType.bool
        case MLX_UINT8:
            return DType.uint8
        case MLX_UINT16:
            return DType.uint16
        case MLX_UINT32:
            return DType.uint32
        case MLX_UINT64:
            return DType.uint64
        case MLX_INT8:
            return DType.int8
        case MLX_INT16:
            return DType.int16
        case MLX_INT32:
            return DType.int32
        case MLX_INT64:
            return DType.int64
        case MLX_FLOAT16:
            return DType.float16
        case MLX_FLOAT32:
            return DType.float32
        case MLX_BFLOAT16:
            return DType.bfloat16
        case MLX_COMPLEX64:
            return DType.complex64
        default:
            fatalError("MLX: internal: unsupported MLX dtype")
        }
    }
    public func shape() -> Swift.Array<Int32> {
        let cShape = mlx_array_shape(ctx)
        let ndim = mlx_array_ndim(ctx)
        let buffer = UnsafeBufferPointer<Int32>(start: cShape, count: ndim)
        return Swift.Array<Int32>(buffer)
    }
    public func asType(_ type: DType, stream: Stream = Stream()) -> Array {
        var cType : mlx_array_dtype
        switch(type) {
        case DType.bool:
            cType = MLX_BOOL 
        case DType.uint8:
            cType = MLX_UINT8 
        case DType.uint16:
            cType = MLX_UINT16 
        case DType.uint32:
            cType = MLX_UINT32 
        case DType.uint64:
            cType = MLX_UINT64 
        case DType.int8:
            cType = MLX_INT8 
        case DType.int16:
            cType = MLX_INT16 
        case DType.int32:
            cType = MLX_INT32 
        case DType.int64:
            cType = MLX_INT64 
        case DType.float16:
            cType = MLX_FLOAT16 
        case DType.float32:
            cType = MLX_FLOAT32 
        case DType.complex64:
            cType = MLX_COMPLEX64 
        default:
            fatalError("MLX: internal: unsupported MLX dtype")
        }
        return Array(mlx_astype(ctx, cType, stream.ctx))
    }
    deinit {
        Cmlx.mlx_free(UnsafeMutableRawPointer(ctx))
    }
}

extension Array: CustomStringConvertible {
    public var description: String {
        let cDsc = Cmlx.mlx_tostring(UnsafeMutableRawPointer(ctx))
        var dsc = String(describing: type(of: self))
        if cDsc != nil {
            dsc = String(cString: cDsc!)
        }
        free(cDsc)
        return dsc
    }
}
