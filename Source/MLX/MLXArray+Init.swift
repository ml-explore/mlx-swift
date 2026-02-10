// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Numerics

private func shapePrecondition(shape: (some Collection<Int>)?, count: Int) {
    if let shape {
        let total = shape.reduce(1, *)
        precondition(total == count, "shape \(shape) total \(total) != \(count) (actual)")
    }
}

private func shapePrecondition(shape: (some Collection<Int>)?, byteCount: Int, type: DType) {
    if let shape {
        let total = shape.reduce(1, *) * type.size
        precondition(total == byteCount, "shape \(shape) total \(total)B != \(byteCount)B (actual)")
    }
}

// holds reference to `finalizer` as capture state
private class FinalizerCaptureState {
    let f: () -> Void

    init(_ f: @escaping () -> Void) {
        self.f = f
    }
}

// the C function that the mlx_array_new_data_managed_payload will call
func finalizerTrampoline(
    payload: UnsafeMutableRawPointer?
) {
    let state = Unmanaged<FinalizerCaptureState>.fromOpaque(payload!).takeUnretainedValue()
    state.f()
}

extension MLXArray {

    /// Initialize an MLXArray by transferring ownership of a raw pointer.
    ///
    /// Note: the raw pointer must be compatible with the computational backing, e.g. a
    /// Metal stream requires something compatible with an `MTLBuffer`.
    ///
    /// For example:
    ///
    /// ```swift
    /// let height = 100
    /// let width = 128
    /// let pixelFormat = kCVPixelFormatType_32BGRA
    ///
    /// let properties: [IOSurfacePropertyKey: any Sendable] = [
    ///     .width: width,
    ///     .height: height,
    ///     .pixelFormat: pixelFormat,
    ///     .bytesPerElement: 4
    /// ]
    ///
    /// guard let ioSurface = IOSurface(properties: properties) else {
    ///     XCTFail("unable to allocate IOSurface")
    ///     return
    /// }
    ///
    /// let array = MLXArray(rawPointer: ioSurface.baseAddress, [height, width, 4] ,dtype: .uint8) {
    ///     [ioSurface] in
    ///     // this holds reference to the ioSurface and implicitly releases it when it returns
    ///     _ = ioSurface
    ///     print("release IOSurface")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - rawPointer: raw pointer to input data -- ownership is transferred to MLXArray.  The data
    ///     must be contiguous in the shape and dtype given (or at least it will be treated as such)
    ///   - shape: shape of the data in the rawPointer
    ///   - dtype: data type
    ///   - finalizer: closure that will release the associated resource
    // TODO: disabled per issue on mlx side -- buffer enters residency set but
    // not removed -- enable in next release.  Also testIOSurface
    //    public convenience init(
    //        rawPointer: UnsafeMutableRawPointer,
    //        _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType,
    //        finalizer: @escaping () -> Void
    //    ) {
    //        func free(ptr: UnsafeMutableRawPointer?) {
    //            Unmanaged<FinalizerCaptureState>.fromOpaque(ptr!).release()
    //        }
    //
    //        let payload = Unmanaged.passRetained(FinalizerCaptureState(finalizer)).toOpaque()
    //
    //        self.init(
    //            mlx_array_new_data_managed_payload(
    //                rawPointer,
    //                shape?.asInt32, (shape?.count ?? 0).int32,
    //                dtype.cmlxDtype,
    //                payload, finalizerTrampoline))
    //    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `Int32`.
    ///
    /// ```swift
    /// let a = MLXArray(Int32(7))
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Int32) {
        self.init(mlx_array_new_int(value))
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `Int` as
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
        self.init(mlx_array_new_int(Int32(value)))
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `Int` as
    /// a `Dtype.int64`.
    ///
    /// ```swift
    /// let a = MLXArray(int64: Int(Int32.max) + 10)
    /// ```
    ///
    /// Note ``init(_:)-(Int)`` (producing an `int32` scalar is preferred).
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(int64 value: Int) {
        self.init(
            withUnsafePointer(to: value) { ptr in
                mlx_array_new_data(ptr, [], 0, Int.dtype.cmlxDtype)
            })
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from a `Bool`.
    ///
    /// ```swift
    /// let a = MLXArray(true)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Bool) {
        self.init(mlx_array_new_bool(value))
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from a `Float`.
    ///
    /// ```swift
    /// let a = MLXArray(35.1)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Float) {
        self.init(mlx_array_new_float(value))
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from a `Double` as
    /// a `Dtype.float64`.
    ///
    /// ```swift
    /// let a = MLXArray(float64: 1.1e11)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(float64 value: Double) {
        self.init(
            withUnsafePointer(to: value) { ptr in
                mlx_array_new_data(ptr, [], 0, Double.dtype.cmlxDtype)
            })
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from a `HasDType` value.
    ///
    /// ```swift
    /// let a = MLXArray(UInt64(7))
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(_ value: T) {
        let floatMax = Double(Float.greatestFiniteMagnitude)
        if let doubleValue = value as? Double, doubleValue < floatMax && doubleValue > -floatMax {
            self.init(
                withUnsafePointer(to: Float(doubleValue)) { ptr in
                    mlx_array_new_data(ptr, [], 0, DType.float32.cmlxDtype)
                })
        } else {
            self.init(
                withUnsafePointer(to: value) { ptr in
                    mlx_array_new_data(ptr, [], 0, T.dtype.cmlxDtype)
                })
        }
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` with a ``DType/bfloat16``
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
        let v_mlx = mlx_array_new_float(Float32(value))
        defer { mlx_array_free(v_mlx) }
        var v_bfloat = mlx_array_new()
        mlx_astype(&v_bfloat, v_mlx, DType.bfloat16.cmlxDtype, stream.ctx)
        self.init(v_bfloat)
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from a `HasDType` value
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
                        mlx_array_new_data(ptr, [], 0, T.dtype.cmlxDtype)
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
                case .float64:
                    self.init(Float64(v))
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
                case .float64:
                    self.init(Float64(v))
                }

            } else {
                // e.g. Bool -> Int
                fatalError("unable to coerce \(T.dtype) to \(dtype)")
            }
        }
    }

    /// Initializer allowing creation of `MLXArray` from an array of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let linear = MLXArray([0, 1, 2, 3])
    /// let twoByTwo = MLXArray([0, 1, 2, 3], [2, 2])
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<T: HasDType>(
        _ value: [T], _ shape: (some Collection<Int>)? = [Int]?.none
    ) {
        shapePrecondition(shape: shape, count: value.count)
        self.init(
            value.withUnsafeBufferPointer { ptr in
                let shape = shape?.asInt32 ?? [value.count.int32]
                return mlx_array_new_data(
                    ptr.baseAddress!, shape, shape.count.int32, T.dtype.cmlxDtype)
            })
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
    /// a `Dtype.int32`.
    ///
    /// ```swift
    /// let a = MLXArray([1, 2, 3])
    /// ```
    ///
    /// Note: if the value is out of bounds for an `Int32` the precondition will fail.  If you
    /// need an `Int` (`Int64`) scalar, please use ``init(int64:_:)-(Sequence<Int>,_)``.
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``init(int64:_:)-(Sequence<Int>,_)``
    public convenience init(_ value: [Int], _ shape: (some Collection<Int>)? = [Int]?.none) {
        shapePrecondition(shape: shape, count: value.count)
        precondition(
            value.allSatisfy { (Int(Int32.min) ... Int(Int32.max)).contains($0) },
            "\(value) is out of range for Int32 -- please use MLXArray(int64: [Int]]) if you need 64 bits."
        )

        self.init(
            value
                .map { Int32($0) }
                .withUnsafeBufferPointer { ptr in
                    let shape = shape?.asInt32 ?? [value.count.int32]
                    return mlx_array_new_data(
                        ptr.baseAddress!, shape, shape.count.int32, Int32.dtype.cmlxDtype)
                })
    }

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
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
    public convenience init(int64 value: [Int], _ shape: (some Collection<Int>)? = [Int]?.none) {
        shapePrecondition(shape: shape, count: value.count)

        self.init(
            value
                .withUnsafeBufferPointer { ptr in
                    let shape = shape?.asInt32 ?? [value.count.int32]
                    return mlx_array_new_data(
                        ptr.baseAddress!, shape, shape.count.int32, Int.dtype.cmlxDtype)
                })
    }

    /// Initializer allowing creation of `MLXArray` from an array of `Double` values with
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
    public convenience init(
        converting value: [Double], _ shape: (some Collection<Int>)? = [Int]?.none
    ) {
        shapePrecondition(shape: shape, count: value.count)
        let floats = value.map { Float($0) }
        self.init(
            floats.withUnsafeBufferPointer { ptr in
                let shape = shape?.asInt32 ?? [floats.count.int32]
                return mlx_array_new_data(
                    ptr.baseAddress!, shape, shape.count.int32, Float.dtype.cmlxDtype)
            })
    }

    /// Unavailable init to redirect for initializing with a `[Double]`
    @available(
        *, unavailable, renamed: "MLXArray(converting:shape:)",
        message: "Use MLXArray(converting: [1.0, 2.0, ...]) instead"
    )
    public convenience init(_ value: [Double], _ shape: (some Collection<Int>)? = [Int]?.none) {
        fatalError("unavailable")
    }

    /// Initializer allowing creation of `MLXArray` from a sequence of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let ramp = MLXArray(0 ..< 64)
    /// let square = MLXArray(0 ..< 64, [8, 8])
    /// ```
    ///
    /// Note: if the element type is `Int` this will produce an ``DType/int32`` result.
    /// See ``init(int64:_:)-(Sequence<Int>,_)`` if an `.int64` is required.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<S: Sequence>(
        _ sequence: S, _ shape: (some Collection<Int>)? = [Int]?.none
    )
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

    /// Initializer allowing creation of scalar (0-dimension) `MLXArray` from an `[Int]` as
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
    public convenience init(
        int64 sequence: some Sequence<Int>, _ shape: (some Collection<Int>)? = [Int]?.none
    ) {
        let value = Array(sequence)
        shapePrecondition(shape: shape, count: value.count)
        self.init(
            value.withUnsafeBufferPointer { ptr in
                let shape = shape?.asInt32 ?? [value.count.int32]
                return mlx_array_new_data(
                    ptr.baseAddress!, shape, shape.count.int32, Int.dtype.cmlxDtype)
            })
    }

    /// Initializer allowing creation of `MLXArray` from a buffer of `HasDType` values with
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
    public convenience init<T: HasDType>(
        _ ptr: UnsafeBufferPointer<T>, _ shape: (some Collection<Int>)? = [Int]?.none
    ) {
        shapePrecondition(shape: shape, count: ptr.count)
        let shape = shape?.asInt32 ?? [ptr.count.int32]
        self.init(
            mlx_array_new_data(
                ptr.baseAddress!, shape, shape.count.int32, T.dtype.cmlxDtype))
    }

    /// Initializer allowing creation of `MLXArray` from a `UnsafeRawBufferPointer` filled
    /// with bytes of `HasDType` values with an optional shape.
    ///
    /// ```swift
    /// let ptr: UnsafeRawBufferPointer
    /// let array = MLXArray(ptr, [2, 3], type: Int32.self)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(
        _ ptr: UnsafeRawBufferPointer, _ shape: (some Collection<Int>)? = [Int]?.none,
        type: (some HasDType).Type
    ) {
        let buffer = ptr.assumingMemoryBound(to: type)
        self.init(buffer, shape)
    }

    /// Initializer allowing creation of `MLXArray` from a `Data` filled with bytes of `HasDType` values with
    /// an optional shape.
    ///
    /// ```swift
    /// let data: Data
    /// let array = MLXArray(data, [2, 3], type: Int32.self)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(
        _ data: Data, _ shape: (some Collection<Int>)? = [Int]?.none, type: (some HasDType).Type
    ) {
        self.init(
            data.withUnsafeBytes { ptr in
                let buffer = ptr.assumingMemoryBound(to: type)
                shapePrecondition(shape: shape, count: buffer.count)
                let shape = shape?.asInt32 ?? [buffer.count.int32]
                return mlx_array_new_data(
                    ptr.baseAddress!, shape, shape.count.int32, type.dtype.cmlxDtype)
            })
    }

    /// Initializer allowing creation of `MLXArray` from a `Data`  buffer values with
    /// an optional shape and an explicit DType.
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(
        _ data: Data, _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType
    ) {
        self.init(
            data.withUnsafeBytes { ptr in
                shapePrecondition(shape: shape, byteCount: data.count, type: dtype)
                precondition(data.count % dtype.size == 0)
                let shape = shape?.asInt32 ?? [Int32(data.count / dtype.size)]
                return mlx_array_new_data(
                    ptr.baseAddress!, shape, shape.count.int32, dtype.cmlxDtype)
            })
    }

    public convenience init(data: MLXArrayData) {
        self.init(data.data, data.shape, dtype: data.dType)
    }

    /// Create a ``DType/complex64`` scalar.
    /// - Parameters:
    ///   - real: real part
    ///   - imaginary: imaginary part
    public convenience init(real: Float, imaginary: Float) {
        self.init(mlx_array_new_data([real, imaginary], [], 0, DType.complex64.cmlxDtype))
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

    /// Initializer allowing creation of 1d `MLXArray` from an array literal.
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
            return mlx_array_new_data(
                ptr.baseAddress!, shape, Int32(shape.count), Int32.dtype.cmlxDtype)
        }
        self.init(ctx)
    }
}
