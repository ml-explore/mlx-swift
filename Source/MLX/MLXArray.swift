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
        precondition(self.size == 1)

        if type.dtype != self.dtype {
            return self.asType(type).item(type)
        }

        self.eval()

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

    /// Return the dimension where the storage is contiguous.
    ///
    /// If this returns 0 then the whole storage is contiguous.  If it returns ndmin + 1 then none of it is contiguous.
    func contiguousToDimension() -> Int {
        let shape = self.shape
        let strides = self.strides

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

    /// Return the physical size of the backing (assuming it is evaluated) in elements
    var physicalSize: Int {
        // nbytes is the logical size of the input, not the physical size
        return zip(self.shape, self.strides)
            .map { Swift.abs($0.0 * $0.1) }
            .max()
            ?? self.size
    }

    func copy(from: UnsafeRawBufferPointer, to output: UnsafeMutableRawBufferPointer) {
        let contiguousDimension = self.contiguousToDimension()

        if contiguousDimension == 0 {
            // entire backing is contiguous
            from.copyBytes(to: output)

        } else {
            // only part of the backing is contiguous (possibly a single element)
            // iterate the non-contiguous parts and copy the contiguous chunks into
            // the output.

            // these are the parts to iterate
            let shape = self.shape.prefix(upTo: contiguousDimension)
            let strides = self.strides.prefix(upTo: contiguousDimension)
            let ndim = contiguousDimension
            let itemSize = self.itemSize

            // the size of each chunk that we copy.  this computes the stride of
            // (contiguousDimension - 1) if it were contiguous
            let destItemSize: Int
            if contiguousDimension == self.ndim {
                // nothing contiguous
                destItemSize = itemSize
            } else {
                destItemSize =
                    self.strides[contiguousDimension] * self.shape[contiguousDimension] * itemSize
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
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        if type.dtype != self.dtype {
            return self.asType(type).asArray(type)
        }

        self.eval()

        return [T](unsafeUninitializedCapacity: self.size) { destination, initializedCount in
            let source = UnsafeRawBufferPointer(
                start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)
            copy(from: source, to: UnsafeMutableRawBufferPointer(destination))
            initializedCount = self.size
        }
    }

    /// Return the contents as contiguous bytes in the native ``dtype``.
    ///
    /// > If you can guarantee the lifetime of the ``MLXArray`` will exceed the Data and that
    /// the array will not be mutated (e.g. using indexing or other means) it is possible to pass `noCopy: true`
    /// to reference the backing bytes.
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    public func asData(noCopy: Bool = false) -> Data {
        self.eval()

        if noCopy && self.contiguousToDimension() == 0 {
            // the backing is contiguous, we can provide a wrapper
            // for the contents without a copy (if requested)
            let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
            return Data(
                bytesNoCopy: source, count: self.nbytes,
                deallocator: .none)
        } else {
            let source = UnsafeRawBufferPointer(
                start: mlx_array_data_uint8(self.ctx), count: physicalSize * itemSize)

            var data = Data(count: self.nbytes)
            data.withUnsafeMutableBytes { destination in
                copy(from: source, to: destination)
            }
            return data
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
