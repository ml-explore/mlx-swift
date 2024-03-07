// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Numerics

private func shapePrecondition(shape: [Int]?, count: Int) {
    if let shape {
        let total = shape.reduce(1, *)
        precondition(total == count, "shape \(shape) total \(total) != \(count) (actual)")
    }
}

extension MLXArray {

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `Int32`.
    ///
    /// ```swift
    /// let a = MLXArray(Int32(7))
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Int32) {
        self.init(mlx_array_from_int(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `Int` as
    /// a `Dtype.int32`.
    ///
    /// ```swift
    /// let a = MLXArray(7)
    /// ```
    ///
    /// Note: if the value is out of bounds for an `Int32` the precondition will fail.  If you
    /// need an `Int` (`Int64`) scalar, please use ``init(int64:)``.
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``init(int64:)``
    public convenience init(_ value: Int) {
        precondition(
            (Int(Int32.min) ... Int(Int32.max)).contains(value),
            "\(value) is out of range for Int32 -- please use MLXArray(int64: Int) if you need 64 bits."
        )
        self.init(mlx_array_from_int(Int32(value)))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `Int` as
    /// a `Dtype.int64`.
    ///
    /// ```swift
    /// let a = MLXArray(int64: Int(Int32.max) + 10)
    /// ```
    ///
    /// Note ``init(_:)-6nnka`` (producing an `int32` scalar is preferred).
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(int64 value: Int) {
        self.init(
            withUnsafePointer(to: value) { ptr in
                mlx_array_from_data(ptr, [], 0, Int.dtype.cmlxDtype)
            })
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `Bool`.
    ///
    /// ```swift
    /// let a = MLXArray(true)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Bool) {
        self.init(mlx_array_from_bool(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `Float`.
    ///
    /// ```swift
    /// let a = MLXArray(35.1)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Float) {
        self.init(mlx_array_from_float(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `HasDType` value.
    ///
    /// ```swift
    /// let a = MLXArray(UInt64(7))
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ value: T) {
        self.init(
            withUnsafePointer(to: value) { ptr in
                mlx_array_from_data(ptr, [], 0, T.dtype.cmlxDtype)
            })
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` with a ``DType/bfloat16``
    /// from a `Float32`.
    ///
    /// ```swift
    /// let a = MLXArray(bfloat16: 35)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(bfloat16 value: Float32) {
        let stream = StreamOrDevice.default
        let v_mlx = mlx_array_from_float(Float32(value))!
        defer { mlx_free(v_mlx) }
        let v_bfloat = mlx_astype(v_mlx, DType.bfloat16.cmlxDtype, stream.ctx)!
        self.init(v_bfloat)
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `HasDType` value
    /// with a conversion to a given ``DType``.
    ///
    /// ```swift
    /// let a = MLXArray(7.5, dtype: .float16)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ScalarOrArray``
    public convenience init<T: HasDType>(_ value: T, dtype: DType) {
        if T.dtype == dtype {
            // matching dtypes, no coercion
            switch type(of: value) {
            case is Int32.Type:
                self.init(value as! Int32)
            case is Bool.Type:
                self.init(value as! Bool)
            case is Float32.Type:
                self.init(value as! Float32)
            default:
                self.init(
                    withUnsafePointer(to: value) { ptr in
                        mlx_array_from_data(ptr, [], 0, T.dtype.cmlxDtype)
                    })
            }
        } else {
            if let v = value as? (any BinaryFloatingPoint) {
                // Floatish-ish source
                switch dtype {
                case .bool:
                    self.init(!v.isZero)
                case .uint8:
                    self.init(UInt8(v))
                case .uint16:
                    self.init(UInt16(v))
                case .uint32:
                    self.init(UInt32(v))
                case .uint64:
                    self.init(UInt64(v))
                case .int8:
                    self.init(Int8(v))
                case .int16:
                    self.init(Int16(v))
                case .int32:
                    self.init(Int32(v))
                case .int64:
                    self.init(Int64(v))
                #if !arch(x86_64)
                    case .float16:
                        self.init(Float16(v))
                #else
                    case .float16:
                        fatalError("dtype \(dtype) not supported")
                #endif
                case .float32:
                    self.init(Float32(v))
                case .bfloat16:
                    self.init(bfloat16: Float32(v))
                case .complex64:
                    self.init(real: Float32(v), imaginary: 0)
                }

            } else if let v = value as? (any BinaryInteger) {
                // Int-ish source
                switch dtype {
                case .bool:
                    self.init(Int(v) != 0)
                case .uint8:
                    self.init(UInt8(v))
                case .uint16:
                    self.init(UInt16(v))
                case .uint32:
                    self.init(UInt32(v))
                case .uint64:
                    self.init(UInt64(v))
                case .int8:
                    self.init(Int8(v))
                case .int16:
                    self.init(Int16(v))
                case .int32:
                    self.init(Int32(v))
                case .int64:
                    self.init(Int64(v))
                #if !arch(x86_64)
                    case .float16:
                        self.init(Float16(v))
                #else
                    case .float16:
                        fatalError("dtype \(dtype) not supported")
                #endif
                case .float32:
                    self.init(Float32(v))
                case .bfloat16:
                    self.init(bfloat16: Float32(v))
                case .complex64:
                    self.init(real: Float32(v), imaginary: 0)
                }

            } else {
                // e.g. Bool -> Int
                fatalError("unable to coerce \(T.dtype) to \(dtype)")
            }
        }
    }

    /// Initalizer allowing creation of `MLXArray` from an array of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let linear = MLXArray([0, 1, 2, 3])
    /// let twoByTwo = MLXArray([0, 1, 2, 3], [2, 2])
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ value: [T], _ shape: [Int]? = nil) {
        shapePrecondition(shape: shape, count: value.count)
        self.init(
            value.withUnsafeBufferPointer { ptr in
                let shape = shape ?? [value.count]
                return mlx_array_from_data(
                    ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype)
            })
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
    /// a `Dtype.int32`.
    ///
    /// ```swift
    /// let a = MLXArray([1, 2, 3])
    /// ```
    ///
    /// Note: if the value is out of bounds for an `Int32` the precondition will fail.  If you
    /// need an `Int` (`Int64`) scalar, please use ``init(int64:_:)-7bgj2``.
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``init(int64:_:)-7bgj2``
    public convenience init(_ value: [Int], _ shape: [Int]? = nil) {
        shapePrecondition(shape: shape, count: value.count)
        precondition(
            value.allSatisfy { (Int(Int32.min) ... Int(Int32.max)).contains($0) },
            "\(value) is out of range for Int32 -- please use MLXArray(int64: [Int]]) if you need 64 bits."
        )

        self.init(
            value
                .map { Int32($0) }
                .withUnsafeBufferPointer { ptr in
                    let shape = shape ?? [value.count]
                    return mlx_array_from_data(
                        ptr.baseAddress!, shape.asInt32, shape.count.int32, Int32.dtype.cmlxDtype)
                })
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
    /// a `Dtype.int64`.
    ///
    /// ```swift
    /// let a = MLXArray(int64: [1, 2, 3])
    /// ```
    ///
    /// Note ``init(int64:_:)`` (producing an `int32` scalar is preferred).
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(int64 value: [Int], _ shape: [Int]? = nil) {
        shapePrecondition(shape: shape, count: value.count)

        self.init(
            value
                .withUnsafeBufferPointer { ptr in
                    let shape = shape ?? [value.count]
                    return mlx_array_from_data(
                        ptr.baseAddress!, shape.asInt32, shape.count.int32, Int.dtype.cmlxDtype)
                })
    }

    /// Initalizer allowing creation of `MLXArray` from an array of `Double` values with
    /// an optional shape.
    ///
    /// Note: this converts the types to `Float`, which is a type representable in `MLXArray`
    ///
    /// ```swift
    /// let array = MLXArray(convert: [0.5, 0.9])
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(converting value: [Double], _ shape: [Int]? = nil) {
        shapePrecondition(shape: shape, count: value.count)
        let floats = value.map { Float($0) }
        self.init(
            floats.withUnsafeBufferPointer { ptr in
                let shape = shape ?? [floats.count]
                return mlx_array_from_data(
                    ptr.baseAddress!, shape.asInt32, shape.count.int32, Float.dtype.cmlxDtype)
            })
    }

    /// Unavailable init to redirect for initializing with a `[Double]`
    @available(
        *, unavailable, renamed: "MLXArray(converting:shape:)",
        message: "Use MLXArray(converting: [1.0, 2.0, ...]) instead"
    )
    public convenience init(_ value: [Double], _ shape: [Int]? = nil) {
        fatalError("unavailable")
    }

    /// Initalizer allowing creation of `MLXArray` from a sequence of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let ramp = MLXArray(0 ..< 64)
    /// let square = MLXArray(0 ..< 64, [8, 8])
    /// ```
    ///
    /// Note: if the element type is `Int` this will produce an ``DType/int32`` result.
    /// See ``init(int64:_:)-74tu0`` if an `.int64` is required.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<S: Sequence>(_ sequence: S, _ shape: [Int]? = nil)
    where S.Element: HasDType {
        let value = Array(sequence)
        if S.Element.self == Int.self {
            // having an override for Sequence<Int> is ambiguous so
            // do a runtime check and force it to the [Int] variant
            self.init(value as! [Int], shape)
        } else {
            self.init(value, shape)
        }
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
    /// a `Dtype.int64`.
    ///
    /// ```swift
    /// let a = MLXArray(int64: [1, 2, 3])
    /// ```
    ///
    /// Note ``init(int64:_:)`` (producing an `int32` scalar is preferred).
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(int64 sequence: any Sequence<Int>, _ shape: [Int]? = nil) {
        let value = Array(sequence)
        shapePrecondition(shape: shape, count: value.count)
        self.init(
            value.withUnsafeBufferPointer { ptr in
                let shape = shape ?? [value.count]
                return mlx_array_from_data(
                    ptr.baseAddress!, shape.asInt32, shape.count.int32, Int.dtype.cmlxDtype)
            })
    }

    /// Initalizer allowing creation of `MLXArray` from a buffer of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let image = vImage.PixelBuffer
    /// let array = image.withUnsafeBufferPointer { ptr in
    ///     MLXArray(ptr, [64, 64, 4])
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ ptr: UnsafeBufferPointer<T>, _ shape: [Int]? = nil) {
        shapePrecondition(shape: shape, count: ptr.count)
        let shape = shape ?? [ptr.count]
        self.init(
            mlx_array_from_data(
                ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype))
    }

    /// Initalizer allowing creation of `MLXArray` from a `UnsafeRawBufferPointer` filled
    /// with bytes of `HasDType` values with an optional shape.
    ///
    /// ```swift
    /// let ptr: UnsafeRawBufferPointer
    /// let array = MLXArray(ptr, [2, 3], type: Int32.self)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(
        _ ptr: UnsafeRawBufferPointer, _ shape: [Int]? = nil, type: T.Type
    ) {
        let buffer = ptr.assumingMemoryBound(to: type)
        self.init(buffer, shape)
    }

    /// Initalizer allowing creation of `MLXArray` from a `Data` filled with bytes of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let data: Data
    /// let array = MLXArray(data, [2, 3], type: Int32.self)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ data: Data, _ shape: [Int]? = nil, type: T.Type) {
        self.init(
            data.withUnsafeBytes { ptr in
                let buffer = ptr.assumingMemoryBound(to: type)
                shapePrecondition(shape: shape, count: buffer.count)
                let shape = shape ?? [buffer.count]
                return mlx_array_from_data(
                    ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype)
            })
    }

    /// Create a ``DType/complex64`` scalar.
    /// - Parameters:
    ///   - real: real part
    ///   - imaginary: imaginary part
    public convenience init(real: Float, imaginary: Float) {
        self.init(mlx_array_from_data([real, imaginary], [], 0, DType.complex64.cmlxDtype))
    }

    /// Create a ``DType/complex64`` scalar from `Complex<Float>`.
    public convenience init(_ value: Complex<Float>) {
        self.init(real: value.real, imaginary: value.imaginary)
    }
}

// MARK: - Expressible by literals

extension MLXArray: ExpressibleByArrayLiteral {

    // Note: MLXArray does not implement ExpressibleByFloatLiteral etc. because
    // we want to create arrays in the context of the other arrays.  For example:
    //
    // let x = MLXArray(1.5, dtype: .float16)
    // let r = x + 2.5
    //
    // We expect r to have a dtype of float16.  See ``ScalarOrArray``.

    /// Initalizer allowing creation of 1d `MLXArray` from an array literal.
    ///
    /// ```swift
    /// let a: MLXArray = [1, 2, 3]
    /// ```
    ///
    /// This is convenient for methods that have `MLXArray` parameters:
    ///
    /// ```swift
    /// print(array.take([1, 2, 3], axis: 0))
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(arrayLiteral elements: Int32...) {
        let ctx = elements.withUnsafeBufferPointer { ptr in
            let shape = [Int32(elements.count)]
            return mlx_array_from_data(
                ptr.baseAddress!, shape, Int32(shape.count), Int32.dtype.cmlxDtype)!
        }
        self.init(ctx)
    }
}
