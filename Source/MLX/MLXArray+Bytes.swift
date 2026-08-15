// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

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

    /// Copy `self`'s (possibly non-contiguous) backing, starting at `base`, into a
    /// contiguous `output` buffer.
    ///
    /// `base` is treated as a bare starting point, not a pre-sized region: for the
    /// non-contiguous case the actual reachable byte range (which can extend *before*
    /// `base` when strides are negative) is computed internally and bounds-checked via
    /// `RawSpan`, rather than trusting unchecked pointer arithmetic to stay in bounds.
    func copy(from base: UnsafeRawPointer, toContiguous output: UnsafeMutableRawBufferPointer) {
        guard !output.isEmpty else { return }

        let contiguousDimension = self.contiguousToDimension()
        let shape = self.shape
        let strides = self.internalStrides

        if contiguousDimension == 0 {
            // entire backing is contiguous
            let source = unsafe RawSpan(
                _unsafeBytes: UnsafeRawBufferPointer(start: base, count: output.count))
            var destination = unsafe MutableRawSpan(_unsafeBytes: output)
            source.withUnsafeBytes { src in
                destination.withUnsafeMutableBytes { dest in
                    dest.copyMemory(from: src)
                }
            }

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

            // Compute the true reachable range of sourceIndex (in elements). This can be
            // negative when strides are negative, e.g.
            // asStrided(a, [3, 3], strides: [-3, -1], offset: 8) -- the base pointer we
            // have will have the offset already applied, and indices approaching the
            // "backward" direction can land before `base`. A per-dimension index ranges
            // 0...(shape-1), so each dimension's contribution to sourceIndex ranges from
            // min(0, (shape-1)*stride) to max(0, (shape-1)*stride); summing these across
            // dimensions gives the true min/max reachable sourceIndex for the whole loop.
            var minSourceIndex = 0
            var maxSourceIndex = 0
            for dimension in 0 ..< ndim {
                let contribution = (shape[dimension] - 1) * strides[dimension]
                minSourceIndex += Swift.min(0, contribution)
                maxSourceIndex += Swift.max(0, contribution)
            }

            // Bootstrap a RawSpan covering exactly the reachable byte range, anchored so
            // that byte offset 0 corresponds to minSourceIndex -- this correctly extends
            // "backward" from `base` when strides are negative, instead of assuming (as
            // unchecked pointer arithmetic silently did before) that everything reachable
            // lies forward of `base`.
            let spanByteCount = (maxSourceIndex - minSourceIndex) * itemSize + destItemSize
            let spanBase = base + minSourceIndex * itemSize
            let source = unsafe RawSpan(
                _unsafeBytes: UnsafeRawBufferPointer(start: spanBase, count: spanByteCount))
            var destination = unsafe MutableRawSpan(_unsafeBytes: output)

            // the index of the current source item
            var index = Array.init(repeating: 0, count: ndim)

            // Keep this in step with the odometer below. Recomputing it from every index and
            // stride made traversal O(numberOfChunks * rank).
            var sourceIndex = 0

            // output byte offset
            var destOffset = 0

            while true {
                // offset relative to the span's own start -- always >= 0 by construction,
                // since minSourceIndex is the true minimum reachable sourceIndex
                let spanOffset = (sourceIndex - minSourceIndex) * itemSize
                let chunk = source.extracting(spanOffset ..< (spanOffset + destItemSize))

                chunk.withUnsafeBytes { src in
                    // `_mutatingExtracting` (not `extracting`) is the current non-deprecated
                    // spelling for MutableRawSpan's bounds-checked sub-range extraction in
                    // this SDK -- the leading underscore signals this corner of the Span API
                    // is still pre-stabilization and may be renamed again in a future SDK.
                    var chunkDestination = destination._mutatingExtracting(
                        destOffset ..< (destOffset + destItemSize))
                    chunkDestination.withUnsafeMutableBytes { dest in
                        dest.copyMemory(from: src)
                    }
                }

                // next output offset
                destOffset += destItemSize

                // increment the index
                for dimension in Swift.stride(from: ndim - 1, through: 0, by: -1) {
                    // do we need to "carry" into the next dimension?
                    if index[dimension] == (shape[dimension] - 1) {
                        if dimension == 0 {
                            // all done
                            return
                        }

                        index[dimension] = 0
                        sourceIndex -= (shape[dimension] - 1) * strides[dimension]
                    } else {
                        // just increment the dimension and we are done
                        index[dimension] += 1
                        sourceIndex += strides[dimension]
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
    /// - ``asData(access:)``
    /// - ``asData(noCopy:disambiguate:)``
    /// - ``asMTLBuffer(device:noCopy:)``
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        if type.dtype != self.dtype {
            return self.asType(type).asArray(type)
        }

        self.eval()

        return [T](unsafeUninitializedCapacity: self.size) { destination, initializedCount in
            copy(
                from: UnsafeRawPointer(mlx_array_data_uint8(self.ctx))!,
                toContiguous: UnsafeMutableRawBufferPointer(destination))
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

    /// Container for `Data` backing of ``MLXArray``.
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
    internal func asDataCopy() -> MLXArrayData {
        // point into the possibly non-contiguous backing
        let source = UnsafeRawPointer(mlx_array_data_uint8(self.ctx))!

        var data = Data(count: self.nbytes)
        data.withUnsafeMutableBytes { destination in
            copy(from: source, toContiguous: destination)
        }
        return MLXArrayData(
            data: data,
            shape: self.shape, strides: contiguousStrides(shape: self.shape),
            dType: self.dtype)
    }

    /// Return the contents as `Data` bytes in the native ``dtype``.
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
