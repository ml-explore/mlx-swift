import Cmlx

/// Enum wrapping `Cmlx.mlx_array_dtype`.
///
/// This is typically not used directly, rather it is inferred from parameters that are ``HasDType``.
public enum DType : Sendable, Hashable {
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
    
    init(_ cmlxDtype: mlx_array_dtype) {
        switch cmlxDtype {
        case MLX_BOOL: self = .bool
        case MLX_UINT8: self = .uint8
        case MLX_UINT16: self = .uint16
        case MLX_UINT32: self = .uint32
        case MLX_UINT64: self = .uint64
        case MLX_INT8: self = .int8
        case MLX_INT16: self = .int16
        case MLX_INT32: self = .int32
        case MLX_INT64: self = .int64
        case MLX_FLOAT16: self = .float16
        case MLX_FLOAT32: self = .float32
        case MLX_BFLOAT16: self = .bfloat16
        case MLX_COMPLEX64: self = .complex64
        default:
            fatalError("Unsupported dtype: \(cmlxDtype)")
        }
    }
    
    var cmlxDtype: mlx_array_dtype {
        switch self {
        case .bool: MLX_BOOL
        case .uint8: MLX_UINT8
        case .uint16: MLX_UINT16
        case .uint32: MLX_UINT32
        case .uint64: MLX_UINT64
        case .int8: MLX_INT8
        case .int16: MLX_INT16
        case .int32: MLX_INT32
        case .int64: MLX_INT64
        case .float16: MLX_FLOAT16
        case .float32: MLX_FLOAT32
        case .bfloat16: MLX_BFLOAT16
        case .complex64: MLX_COMPLEX64
        }
    }
}

/// Protocol for types that can provide a ``DType``
///
/// This is used to extract the ``DType`` for values pass in to ``MLXArray`` implicitly
/// where possible.
public protocol HasDType {
    
    /// Return the type's ``DType``
    static var dtype: DType { get }
}

extension Bool : HasDType {
    static public var dtype: DType { .bool }
}

extension Int : HasDType {
    static public var dtype: DType { .int64 }
}

extension Int8 : HasDType {
    static public var dtype: DType { .int8 }
}
extension Int16 : HasDType {
    static public var dtype: DType { .int16 }
}
extension Int32 : HasDType {
    static public var dtype: DType { .int32 }
}
extension Int64 : HasDType {
    static public var dtype: DType { .int64 }
}

extension UInt8 : HasDType {
    static public var dtype: DType { .uint8 }
}
extension UInt16 : HasDType {
    static public var dtype: DType { .uint16 }
}
extension UInt32 : HasDType {
    static public var dtype: DType { .uint32 }
}
extension UInt64 : HasDType {
    static public var dtype: DType { .uint64 }
}

extension Float16 : HasDType {
    static public var dtype: DType { .float16 }
}
extension Float32 : HasDType {
    static public var dtype: DType { .float32 }
}
