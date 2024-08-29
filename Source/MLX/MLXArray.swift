// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Metal
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
        switch self.dtype {
        case .bool: mlx_array_item_bool(self.ctx) ? 1 : 0
        case .uint8: Int(mlx_array_item_uint8(self.ctx))
        case .uint16: Int(mlx_array_item_uint16(self.ctx))
        case .uint32: Int(mlx_array_item_uint32(self.ctx))
        case .uint64: Int(mlx_array_item_uint64(self.ctx))
        case .int8: Int(mlx_array_item_int8(self.ctx))
        case .int16: Int(mlx_array_item_int16(self.ctx))
        case .int32: Int(mlx_array_item_int32(self.ctx))
        case .int64: Int(mlx_array_item_int64(self.ctx))
        default: fatalError("itemInt expected an integer dtype: \(self.dtype)")
        }
    }

    /// specialized conversion between integer types -- see ``item(_:)``
    private func itemUInt() -> UInt {
        switch self.dtype {
        case .bool: mlx_array_item_bool(self.ctx) ? 1 : 0
        case .uint8: UInt(mlx_array_item_uint8(self.ctx))
        case .uint16: UInt(mlx_array_item_uint16(self.ctx))
        case .uint32: UInt(mlx_array_item_uint32(self.ctx))
        case .uint64: UInt(mlx_array_item_uint64(self.ctx))
        case .int8: UInt(mlx_array_item_int8(self.ctx))
        case .int16: UInt(mlx_array_item_int16(self.ctx))
        case .int32: UInt(mlx_array_item_int32(self.ctx))
        case .int64: UInt(mlx_array_item_int64(self.ctx))
        default: fatalError("itemUInt expected an integer dtype: \(self.dtype)")
        }
    }

    /// specialized conversion between float types -- see ``item(_:)``
    private func itemFloat() -> Float {
        switch self.dtype {
        #if !arch(x86_64)
            case .float16: Float(mlx_array_item_float16(self.ctx))
        #endif
        case .float32: Float(mlx_array_item_float32(self.ctx))
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
            case is Float.Type, is Float32.Type, is Float16.Type:
                switch self.dtype {
                case .float16, .float32:
                    switch type {
                    case is Float.Type: return Float(itemFloat()) as! T
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
        case is Bool.Type: return mlx_array_item_bool(self.ctx) as! T
        case is UInt8.Type: return mlx_array_item_uint8(self.ctx) as! T
        case is UInt16.Type: return mlx_array_item_uint16(self.ctx) as! T
        case is UInt32.Type: return mlx_array_item_uint32(self.ctx) as! T
        case is UInt64.Type: return mlx_array_item_uint64(self.ctx) as! T
        case is Int8.Type: return mlx_array_item_int8(self.ctx) as! T
        case is Int16.Type: return mlx_array_item_int16(self.ctx) as! T
        case is Int32.Type: return mlx_array_item_int32(self.ctx) as! T
        case is Int64.Type: return mlx_array_item_int64(self.ctx) as! T
        case is Int.Type: return Int(mlx_array_item_int64(self.ctx)) as! T
        #if !arch(x86_64)
            case is Float16.Type: return mlx_array_item_float16(self.ctx) as! T
        #endif
        case is Float32.Type: return mlx_array_item_float32(self.ctx) as! T
        case is Float.Type: return mlx_array_item_float32(self.ctx) as! T
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

    /// Internal function for copying the backing `mlx::core::array` context.
    func copyContext() -> MLXArray {
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

// MARK: - Backing / Bytes

extension MLXArray {

    /// Return the dimension where the storage is contiguous.
    ///
    /// If this returns 0 then the whole storage is contiguous.  If it returns ndmin + 1 then none of it is contiguous.
    func contiguousToDimension() -> Int {
        let shape = self.shape
        let strides = self.internalStrides

        var expectedStride = 1

        for (dimension, (shape, stride)) in zip(shape, strides).enumerated().reversed() {
            // as long as the actual strides match the expected (contiguous) strides
            // the backing is contiguous in these dimensions
            if stride != expectedStride {
                return dimension + 1
            }
            expectedStride *= shape
        }

        return 0
    }

    /// Return the physical size of the backing (assuming it is evaluated) in elements.  This should
    /// only be used when accessing the backing directly, e.g. via `mlx_array_data_uint8()`
    var physicalSize: Int {
        // nbytes is the logical size of the input, not the physical size
        return zip(self.shape, self.internalStrides)
            .map { Swift.abs($0.0 * $0.1) }
            .max()
            ?? self.size
    }

    func copy(from: UnsafeRawBufferPointer, toContiguous output: UnsafeMutableRawBufferPointer) {
        let contiguousDimension = self.contiguousToDimension()
        let shape = self.shape
        let strides = self.internalStrides

        if contiguousDimension == 0 {
            // entire backing is contiguous
            from.copyBytes(to: output)

        } else {
            // only part of the backing is contiguous (possibly a single element)
            // iterate the non-contiguous parts and copy the contiguous chunks into
            // the output.

            // these are the parts to iterate
            let ndim = contiguousDimension
            let itemSize = self.itemSize

            // the size of each chunk that we copy.  this computes the stride of
            // (contiguousDimension - 1) if it were contiguous
            let destItemSize: Int
            if contiguousDimension == self.ndim {
                // nothing contiguous
                destItemSize = itemSize
            } else {
                destItemSize = strides[contiguousDimension] * shape[contiguousDimension] * itemSize
            }

            // the index of the current source item
            var index = Array.init(repeating: 0, count: ndim)

            // output pointer
            var dest = output.baseAddress!

            while true {
                // compute the source index by multiplying the index by the
                // stride for each dimension

                // note: in the case where the array has negative strides / offset
                // the base pointer we have will have the offset already applied,
                // e.g. asStrided(a, [3, 3], strides: [-3, -1], offset: 8)

                let sourceIndex = zip(index, strides).reduce(0) { $0 + ($1.0 * $1.1) }

                // convert to byte pointer
                let src = from.baseAddress! + sourceIndex * itemSize
                dest.copyMemory(from: src, byteCount: destItemSize)

                // next output address
                dest += destItemSize

                // increment the index
                for dimension in Swift.stride(from: ndim - 1, through: 0, by: -1) {
                    // do we need to "carry" into the next dimension?
                    if index[dimension] == (shape[dimension] - 1) {
                        if dimension == 0 {
                            // all done
                            return
                        }

                        index[dimension] = 0
                    } else {
                        // just increment the dimension and we are done
                        index[dimension] += 1
                        break
                    }
                }
            }

        }
    }

    /// Return the contents as a single contiguous 1d `Swift.Array`.
    ///
    /// Note: because the number of dimensions is dynamic, this cannot produce a multi-dimensional
    /// array.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asData(noCopy:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        if type.dtype != self.dtype {
            return self.asType(type).asArray(type)
        }

        self.eval()

        return [T](unsafeUninitializedCapacity: self.size) { destination, initializedCount in
            let source = UnsafeRawBufferPointer(
                start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)
            copy(from: source, toContiguous: UnsafeMutableRawBufferPointer(destination))
            initializedCount = self.size
        }
    }

    /// How to access backing data with ``asData(access:)`` -- this controls how
    /// ``MLXArrayData`` is produced.
    public enum AccessMethod {
        /// Create a contiguous copy of the data backing the ``MLXArray``.  The lifetime of this data
        /// independent of the lifetime of the ``MLXArray``.
        case copy

        /// Return contiguous data from the backing of the ``MLXArray``, avoiding a copy if possible.
        /// The lifetime of the result is valid only while keeping the ``MLXArray`` alive.  This is best
        /// for temporary copies, e.g. creating and writing an image to disk.
        case noCopyIfContiguous

        /// Return a wrapper around the backing of the ``MLXArray``.  This might not be
        /// contiguous -- the caller must examine ``MLXArrayData/strides``
        /// The lifetime of the result is valid only while keeping the ``MLXArray`` alive.  This is best
        /// for temporary copies, e.g. creating and writing an image to disk.
        case noCopy
    }

    /// Container for ``Data`` backing of ``MLXArray``.
    ///
    /// ### See Also
    /// - ``MLXArray/asData(access:)``
    public struct MLXArrayData {
        /// The bytes backing the ``MLXArray``.
        ///
        /// The ``MLXArray/AccessMethod`` passed to ``MLXArray/asData(access:)``
        /// controls the lifetime and potential layout of this data (e.g. contiguous vs non-contiguous).
        public let data: Data

        /// Dimensions of the data when viewed as ``dType``
        public let shape: [Int]

        /// Strides of the data in terms of the ``dType``
        public let strides: [Int]

        /// The layout of a single items in the ``data``
        public let dType: DType
    }

    /// return a copy of the backing in contiguous layout
    private func asDataCopy() -> MLXArrayData {
        // point into the possibly non-contiguous backing
        let source = UnsafeRawBufferPointer(
            start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)

        var data = Data(count: self.nbytes)
        data.withUnsafeMutableBytes { destination in
            copy(from: source, toContiguous: destination)
        }
        return MLXArrayData(
            data: data,
            shape: self.shape, strides: contiguousStrides(shape: self.shape),
            dType: self.dtype)
    }

    /// Return the contents as ``Data`` bytes in the native ``dtype``.
    ///
    /// > If you use ``AccessMethod/noCopy`` or ``AccessMethod/noCopyIfContiguous`` you
    /// must guarantee that the lifetime of the ``MLXArray`` exceeds the lifetime of the result.
    ///
    /// Callers can specify an ``AccessMethod`` to cause the data to be in _contiguous_ memory
    /// vs. whatever layout the backing actually has.  Callers can use ``MLXArrayData/strides``
    /// if the data is not contiguous.
    ///
    /// By default it will use ``AccessMethod/copy`` and will return a copy of the data in
    /// contiguous memory.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    public func asData(access: AccessMethod = .copy) -> MLXArrayData {
        self.eval()

        switch access {
        case .copy:
            return asDataCopy()

        case .noCopyIfContiguous:
            if self.contiguousToDimension() == 0 {
                // the backing is contiguous, we can provide a wrapper
                // for the contents without a copy
                let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
                let data = Data(
                    bytesNoCopy: source, count: size * itemSize,
                    deallocator: .none)

                return MLXArrayData(
                    data: data,
                    shape: self.shape, strides: contiguousStrides(shape: self.shape),
                    dType: self.dtype)

            } else {
                // not contiguous
                return asDataCopy()
            }

        case .noCopy:
            let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
            let data = Data(
                bytesNoCopy: source, count: nbytes,
                deallocator: .none)

            let strides: [Int]
            if ndim == 0 {
                strides = []
            } else {
                let internalStrides = mlx_array_strides(ctx)!
                strides = (0 ..< ndim).map { Int(internalStrides[$0]) }
            }

            return MLXArrayData(
                data: data,
                shape: self.shape, strides: strides,
                dType: self.dtype)
        }
    }

    /// Return the contents as contiguous bytes in the native ``dtype``.
    ///
    /// > If you can guarantee the lifetime of the ``MLXArray`` will exceed the Data and that
    /// the array will not be mutated (e.g. using indexing or other means) it is possible to pass `noCopy: true`
    /// to reference the backing bytes.
    ///
    /// **Replaced with** ``asData(access:)``
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    @available(*, deprecated, message: "use asData(acccess: .copy)")
    public func asData(noCopy: Bool = false, disambiguate: Bool = false) -> Data {
        self.eval()

        return asData(access: noCopy ? .noCopyIfContiguous : .copy)
            .data
    }

    /// Return the contents as a Metal buffer in the native ``dtype``.
    ///
    /// > If you can guarantee the lifetime of the ``MLXArray`` will exceed the MTLBuffer and that
    /// the array will not be mutated (e.g. using indexing or other means) it is possible to pass `noCopy: true`
    /// to reference the backing bytes.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asData(noCopy:)``
    public func asMTLBuffer(device: any MTLDevice, noCopy: Bool = false) -> (any MTLBuffer)? {
        self.eval()

        if noCopy && self.contiguousToDimension() == 0 {
            // the backing is contiguous, we can provide a wrapper
            // for the contents without a copy (if requested)
            let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
            return device.makeBuffer(bytesNoCopy: source, length: self.nbytes)
        } else {
            let source = UnsafeRawBufferPointer(
                start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)
            return device.makeBuffer(bytes: source.baseAddress!, length: self.nbytes)
        }
    }

}

/// Return the strides for contiguous memory
func contiguousStrides(shape: [Int]) -> [Int] {
    var result = [Int]()
    var current = 1
    for d in shape.reversed() {
        result.append(current)
        current *= d
    }
    result.reverse()
    return result
}
