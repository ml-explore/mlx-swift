// Copyright © 2024 Apple Inc.

import Cmlx
import Numerics

/// Enum wrapping `Cmlx.mlx_array_dtype`.
///
/// This is typically not used directly, rather it is inferred from parameters that are ``HasDType``, for example:
///
/// ```swift
/// let ones = MLXArray.ones([2, 2], type: Int32.self)
/// ```
///
/// provides ``int32`` via ``HasDType/dtype``.
///
/// Some methods take a `dtype` directly for cases where it may not be possible to use a Swift type:
///
/// ```swift
/// let bf = MLXArray(10.5, dtype: .bfloat16)
/// ```
///
/// ### See Also
/// - ``HasDType``
/// - ``MLXArray/asType(_:stream:)-6d44y``
/// - ``MLXArray/asType(_:stream:)-4eqoc``
/// - ``MLXArray/init(_:dtype:)``
public enum DType: Hashable {
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

    public var cmlxDtype: mlx_array_dtype {
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

    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .float32, .bfloat16, .complex64: true
        default: false
        }
    }

    public var isComplex: Bool {
        switch self {
        case .complex64: true
        default: false
        }
    }
}

/// Protocol for types that can provide a ``DType``
///
/// This is used to extract the ``DType`` for values pass in to ``MLXArray`` implicitly
/// where possible.
///
/// See also ``ScalarOrArray``.
public protocol HasDType: ScalarOrArray {

    /// Return the type's ``DType``
    static var dtype: DType { get }
}

extension HasDType {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        MLXArray(self, dtype: dtype ?? Self.dtype)
    }
}

extension Bool: HasDType {
    static public var dtype: DType { .bool }
}

extension Int: HasDType {
    static public var dtype: DType { .int64 }
}

extension Int8: HasDType {
    static public var dtype: DType { .int8 }
}
extension Int16: HasDType {
    static public var dtype: DType { .int16 }
}
extension Int32: HasDType {
    static public var dtype: DType { .int32 }
}
extension Int64: HasDType {
    static public var dtype: DType { .int64 }
}

extension UInt8: HasDType {
    static public var dtype: DType { .uint8 }
}
extension UInt16: HasDType {
    static public var dtype: DType { .uint16 }
}
extension UInt32: HasDType {
    static public var dtype: DType { .uint32 }
}
extension UInt64: HasDType {
    static public var dtype: DType { .uint64 }
}

#if !arch(x86_64)
    extension Float16: HasDType {
        static public var dtype: DType { .float16 }

        public func asMLXArray(dtype: DType?) -> MLXArray {
            let dtype = dtype ?? Self.dtype
            return MLXArray(self, dtype: dtype.isFloatingPoint ? dtype : Self.dtype)
        }
    }
#endif
extension Float32: HasDType {
    static public var dtype: DType { .float32 }

    public func asMLXArray(dtype: DType?) -> MLXArray {
        let dtype = dtype ?? Self.dtype
        return MLXArray(self, dtype: dtype.isFloatingPoint ? dtype : Self.dtype)
    }
}

extension Complex: HasDType where RealType == Float {
    static public var dtype: DType { .complex64 }
}

/// Protocol for promoting a value (e.g. a scalar) to an MLXArray.
public protocol ScalarOrArray {
    /// Convert to ``MLXArray`` using the optional suggested ``DType``.
    ///
    /// If the receiver is a scalar this should consider the `dtype` when producing
    /// an `MLXArray`.  For example:
    ///
    /// ```swift
    /// let x = MLXArray(1.5, dtype: .float16)
    ///
    /// // the 2.5 is expected to conform to the .float16 of the other
    /// // argument and r will be a .float16 result.
    /// let r = x + 2.5
    /// ```
    ///
    /// See also `toArrays(_:_:)` (internal).
    func asMLXArray(dtype: DType?) -> MLXArray
}

extension Int: ScalarOrArray {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        // callers can use Int64() to get explicit .int64 behavior
        MLXArray(Int32(self), dtype: dtype ?? .int32)
    }
}

extension Double: ScalarOrArray {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        MLXArray(Float(self), dtype: dtype ?? .float32)
    }
}

extension Complex: ScalarOrArray where RealType == Float {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        MLXArray(self)
    }
}

extension MLXArray: ScalarOrArray {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        self
    }
}

extension Array: ScalarOrArray where Element: HasDType {
    public func asMLXArray(dtype: DType?) -> MLXArray {
        MLXArray(self).asType(dtype ?? Element.dtype)
    }
}

/// Convert two ``ScalarOrArray`` into ``MLXArray``.
///
/// This is used when taking two arguments that might be scalars:
///
/// ```swift
/// func maximum<A: ScalarOrArray, B: ScalarOrArray>(_ a: A, _ b: B) -> MLXArray {
///     let (a, b) = toArrays(a, b)
///     return MLXArray(mlx_maximum(a.ctx, b.ctx, stream.ctx))
/// }
/// ```
///
/// Four cases:
/// - If both a and b are arrays leave their types alone
/// - If a is an array but b is not, treat b as a weak python type
/// - If b is an array but a is not, treat a as a weak python type
/// - If neither is an array convert to arrays but leave their types alone
///
/// See also ``ScalarOrArray``.
@_documentation(visibility:internal)
public func toArrays<T1: ScalarOrArray, T2: ScalarOrArray>(_ a: T1, _ b: T2) -> (MLXArray, MLXArray)
{
    if let a = a as? MLXArray {
        if let b = b as? MLXArray {
            return (a, b)
        } else {
            return (a, b.asMLXArray(dtype: a.dtype))
        }
    } else if let b = b as? MLXArray {
        return (a.asMLXArray(dtype: b.dtype), b)
    } else {
        return (a.asMLXArray(dtype: nil), b.asMLXArray(dtype: nil))
    }
}
