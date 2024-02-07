// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

private func arange(_ count: Int) -> [Int32] {
    Array(0 ..< Int32(count))
}

private func arange(_ start: Int, _ end: Int) -> [Int32] {
    Array(Int32(start) ..< Int32(end))
}

private func arange(_ start: Int32, _ end: Int32) -> [Int32] {
    Array(Int32(start) ..< Int32(end))
}

private func ones(_ count: Int) -> [Int32] {
    Array(repeating: 1, count: count)
}

@inlinable
func resolve(axis: Int, ndim: Int) -> Int {
    if axis < 0 {
        return axis + ndim
    } else {
        return axis
    }
}

extension MLXArray: Sequence {

    private struct MLXArrayIterator: IteratorProtocol {
        fileprivate var index = -1
        fileprivate let count: Int
        fileprivate let array: MLXArray

        public mutating func next() -> MLXArray? {
            index += 1
            return index < count ? array[index] : nil
        }
    }

    /// Return an iterator along axis 0:
    ///
    /// ```swift
    /// for row in array {
    ///     row...
    /// }
    /// ```
    public func makeIterator() -> some IteratorProtocol<MLXArray> {
        MLXArrayIterator(count: self.dim(0), array: self)
    }
}

extension MLXArray {

    /// Replace the interior ctx (`mlx_array` pointer) with a new value by transferring ownership
    @inline(__always)
    func update(ctx: OpaquePointer) {
        if ctx != self.ctx {
            mlx_free(self.ctx)
            self.ctx = ctx
        }
    }

    /// allow addressing as a positive index or negative (from end) using given axis
    @inlinable
    func resolve(index: Int, axis: Int) -> MLXArray {
        if index < 0 {
            return MLXArray(Int32(index + dim(axis)))
        } else {
            return MLXArray(Int32(index))
        }
    }

    private func resolve(_ indices: [Int]) -> [MLXArray] {
        indices
            .enumerated()
            .map {
                resolve(index: $1, axis: $0)
            }
    }

    private func resolve(_ rangeExpression: any RangeExpression<Int>, _ axis: Int) -> (Int32, Int32)
    {
        func resolve(_ index: Int, _ axis: Int) -> Int32 {
            if index < 0 {
                return Int32(index + dim(axis))
            } else {
                return Int32(index)
            }
        }

        switch rangeExpression {
        case let r as Range<Int>:
            return (resolve(r.lowerBound, axis), resolve(r.upperBound, axis))
        case let r as ClosedRange<Int>:
            return (resolve(r.lowerBound, axis), resolve(r.upperBound, axis) + 1)
        case let r as PartialRangeUpTo<Int>:
            return (0, resolve(r.upperBound, axis))
        case let r as PartialRangeThrough<Int>:
            return (0, resolve(r.upperBound, axis) + 1)
        case let r as PartialRangeFrom<Int>:
            return (resolve(r.lowerBound, axis), dim(axis).int32)
        default:
            fatalError("Unable to handle rangeExpression: \(rangeExpression)")
        }
    }

    // TODO: error checking? subscript out of range, too many indices

    /// Single index subscript.
    ///
    /// This provides access to the given index on axis 0.
    ///
    /// ```swift
    /// // print the contents of row 1
    /// print(array[1])
    ///
    /// // set the value of row 1 to [23] (with broadcast)
    /// array[1] = 23
    /// ```
    public subscript(index: Int, stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            // see mlx_get_item_int
            return take(resolve(index: index, axis: 0), axis: 0, stream: stream)
        }
        set {
            // see mlx_set_item_int()

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            // trim off any singleton dimensions
            let updateShape = Array(newValue.shape.trimmingPrefix { $0 == 1 })

            // this is the shape we will set via broadcast.
            var broadcastShape = self.shape
            broadcastShape[0] = 1

            let expanded = newValue.reshaped(updateShape).broadcast(to: broadcastShape.asInt32)

            let indices = [resolve(index: index, axis: 0)]
            self.update(scattered(indices: indices, updates: expanded, axes: [0]))
        }
    }

    public subscript(index: Int, axis axis: Int, stream stream: StreamOrDevice = .default)
        -> MLXArray
    {
        get {
            // see mlx_get_item_int
            let axis = MLX.resolve(axis: axis, ndim: ndim)
            return take(resolve(index: index, axis: axis), axis: axis, stream: stream)
        }
        set {
            // see mlx_set_item_int()

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            // trim off any singleton dimensions
            let updateShape = Array(newValue.shape.trimmingPrefix { $0 == 1 })

            // this is the shape we will set via broadcast.
            let axis = MLX.resolve(axis: axis, ndim: ndim)
            var broadcastShape = self.shape
            broadcastShape[axis] = 1

            let expanded = newValue.reshaped(updateShape).broadcast(to: broadcastShape.asInt32)

            let indices = [resolve(index: index, axis: axis)]
            self.update(scattered(indices: indices, updates: expanded, axes: [axis.int32]))
        }
    }

    @inlinable
    public subscript(indices: Int..., stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            self[indices, stream: stream]
        }
        set {
            self[indices, stream: stream] = newValue
        }
    }

    /// Multiple axis subscript.
    ///
    /// ```swift
    /// // print the contents of [1, 3]
    /// print(array[1, 3])
    ///
    /// // set the value of [0, 7] to otherArray (with broadcast)
    /// array[0, 7] = otherArray
    /// ```
    public subscript(indices: [Int], stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            // see mlx_get_item_nd

            let axes = arange(indices.count)

            // sliceSizes are 1 for any axes that we specify, full range otherwise
            let sliceSizes = arange(ndim).map {
                $0 < indices.count ? 1 : dim($0)
            }

            // gather the data -- this will be in the same dimensions as the source
            let i = resolve(indices)
            let i_vector_array = new_mlx_vector_array(i)
            defer { mlx_free(i_vector_array) }

            let g = mlx_gather(
                ctx, i_vector_array, axes, axes.count, sliceSizes, sliceSizes.count, stream.ctx)!
            defer {
                mlx_free(g)
            }

            // squeeze down any of the dimensions that were indexed
            let s = mlx_squeeze(g, arange(indices.count), indices.count, stream.ctx)!

            return MLXArray(s)
        }
        set {
            // see mlx_set_item_nd

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            // the shape of the broadcast value
            let broadcastShape = arange(indices.count, self.ndim).map { dim($0) }

            // extended to the dimensions of the array
            let updateShape = MLX.ones(indices.count) + broadcastShape

            let update = newValue.broadcast(to: broadcastShape).reshaped(updateShape)

            let i = resolve(indices)
            let axes = arange(indices.count)
            self.update(scattered(indices: i, updates: update, axes: axes))
        }
    }

    @inlinable
    public subscript(ranges: any RangeExpression<Int>..., stream stream: StreamOrDevice = .default)
        -> MLXArray
    {
        get {
            self[ranges, stream: stream]
        }
        set {
            self[ranges, stream: stream] = newValue
        }
    }

    public subscript(ranges: [any RangeExpression<Int>], stream stream: StreamOrDevice = .default)
        -> MLXArray
    {
        get {
            // see mlx_get_item_nd

            var starts = [Int32]()
            var stops = [Int32]()
            var strides = [Int32]()

            for (i, r) in ranges.enumerated() {
                // we could use r.relative(to:) but that gives no opportunity to
                // handle negatives (offset from end)
                let (lower, upper) = resolve(r, i)
                starts.append(lower)
                stops.append(upper)
                strides.append(1)
            }

            // add any missing dimensions as full range
            for i in starts.count ..< ndim {
                starts.append(0)
                stops.append(dim(i).int32)
                strides.append(1)
            }

            return MLXArray(
                mlx_slice(
                    ctx, starts, starts.count, stops, stops.count, strides, strides.count,
                    stream.ctx))
        }
        set {
            // see mlx_set_item_nd

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            var arrayIndices = [MLXArray]()
            for (i, r) in ranges.enumerated() {
                let (lower, upper) = resolve(r, i)
                let indices = arange(lower, upper)
                var indexShape = MLX.ones(ranges.count).map { Int($0) }
                indexShape[i] = indices.count
                arrayIndices.append(MLXArray(indices, indexShape))
            }

            // conform the shapes to each other
            arrayIndices = MLX.broadcast(arrays: arrayIndices)

            // the shape of the broadcast value
            let broadcastShape =
                arrayIndices[0].shape.asInt32 + arange(ranges.count, self.ndim).map { dim($0) }

            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: MLX.ones(ranges.count), at: arrayIndices[0].ndim)

            let update = newValue.broadcast(to: broadcastShape).reshaped(updateShape)

            let axes = arange(ranges.count)
            self.update(scattered(indices: arrayIndices, updates: update, axes: axes))
        }
    }

    public subscript(indices: MLXArray..., stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

            if indices.count == 1 {
                // fast path for single array
                return take(indices[0], axis: 0)
            }

            let axes = arange(indices.count)
            let sliceSizes = arange(ndim).map {
                $0 < indices.count ? 1 : dim($0)
            }

            let arrayIndices = MLX.broadcast(arrays: indices)

            // shape of the result is based on the shape of the broadcast inputs
            // and padded by any remaining dimensions in the array
            let resultShape =
                arrayIndices[0].shape.asInt32 + arange(indices.count, self.ndim).map { dim($0) }

            let arrayIndices_vec = new_mlx_vector_array(arrayIndices)
            defer { mlx_free(arrayIndices_vec) }

            return MLXArray(
                mlx_gather(
                    ctx, arrayIndices_vec, axes, axes.count, sliceSizes, sliceSizes.count,
                    stream.ctx)
            ).reshaped(resultShape)
        }
        set {
            // see mlx_set_item_nd

            // conform the shapes to each other
            let arrayIndices = MLX.broadcast(arrays: indices)

            // the shape of the broadcast value
            let broadcastShape =
                arrayIndices[0].shape.asInt32 + arange(indices.count, self.ndim).map { dim($0) }

            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: MLX.ones(indices.count), at: arrayIndices[0].ndim)

            let update = newValue.broadcast(to: broadcastShape).reshaped(updateShape)

            let axes = arange(indices.count)
            self.update(scattered(indices: arrayIndices, updates: update, axes: axes))
        }
    }

    public subscript(range: any RangeExpression<Int>, axis axis: Int,
        stream stream: StreamOrDevice = .default
    ) -> MLXArray {
        get {
            var starts = [Int32]()
            var stops = [Int32]()
            var strides = [Int32]()

            for i in 0 ..< ndim {
                starts.append(0)
                stops.append(dim(i).int32)
                strides.append(1)
            }

            let axis = MLX.resolve(axis: axis, ndim: ndim)
            let (lower, upper) = resolve(range, axis)
            starts[axis] = lower
            stops[axis] = upper
            strides[axis] = 1

            return MLXArray(
                mlx_slice(
                    ctx, starts, starts.count, stops, stops.count, strides, strides.count,
                    stream.ctx))
        }
        set {
            // this is [0 ..., 0 ..., range] where the number of full range leading expressions
            let axis = MLX.resolve(axis: axis, ndim: ndim)
            let prefix: [any RangeExpression<Int>] = (0 ..< axis).map { 0 ..< dim($0) }

            self[prefix + [range], stream: stream] = newValue
        }
    }

    /// Indexing using a stride.
    ///
    /// This method supports strided access, similar to python:
    ///
    /// ```python
    /// # access strided contents of array
    /// evens = array[..., ::2]
    /// ```
    ///
    /// can be done in swift with:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< (2 * 3 * 4), [2, 3, 4])
    ///
    /// // array([[[0, 2],
    /// //         [4, 6],
    /// //         [8, 10]],
    /// //        [[12, 14],
    /// //         [16, 18],
    /// //         [20, 22]]], dtype=int64)
    /// let evens = a[stride: 2, axis: -1]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    /// - ``asStrided(_:_:strides:offset:stream:)``
    public subscript(from from: Int? = nil, to to: Int? = nil, stride stride: Int,
        axis axis: Int = -1, stream stream: StreamOrDevice = .default
    ) -> MLXArray {
        get {
            // see mlx_get_item_nd & the RangeExpression code above

            var starts = [Int32]()
            var stops = [Int32]()
            var strides = [Int32]()

            // start with full range
            for i in 0 ..< ndim {
                starts.append(0)
                stops.append(dim(i).int32)
                strides.append(1)
            }

            // update the one we are striding over
            let axis = MLX.resolve(axis: axis, ndim: ndim)

            if stride > 0 {
                starts[axis] = Int32(from ?? 0)
                stops[axis] = Int32(to ?? dim(axis))
            } else {
                // this logic per get_slice_params -- numpy style
                starts[axis] = Int32(from ?? (dim(axis) - 1))
                stops[axis] = Int32(to ?? (-dim(axis) - 1))
            }
            strides[axis] = stride.int32

            return MLXArray(
                mlx_slice(
                    ctx, starts, starts.count, stops, stops.count, strides, strides.count,
                    stream.ctx))
        }
        set {
            // see mlx_set_item_nd

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            var arrayIndices = [MLXArray]()

            // add full range slices up to the axis
            let axis = MLX.resolve(axis: axis, ndim: ndim)

            for i in 0 ..< axis {
                let indices = arange(0, dim(i))
                var indexShape = MLX.ones(axis + 1).map { Int($0) }
                indexShape[i] = indices.count
                arrayIndices.append(MLXArray(indices, indexShape))
            }

            // add the given range
            var start: Int
            var end: Int
            if stride > 0 {
                start = from ?? 0
                end = to ?? dim(axis)
            } else {
                // this logic per get_slice_params -- numpy style
                start = from ?? (dim(axis) - 1)
                end = to ?? (-dim(axis) - 1)
            }

            // handle negative indices
            if start < 0 {
                start = start + dim(axis)
            }
            if end < 0 {
                end = end + dim(axis)
            }

            // build the indices for the slice -- this is different than the Range<T>
            // indices because of the stride
            var indexShape = MLX.ones(axis + 1).map { Int($0) }
            let indices = Array(Swift.stride(from: start.int32, to: end.int32, by: stride))
            indexShape[axis] = indices.count
            arrayIndices.append(MLXArray(indices, indexShape))

            // conform the shapes to each other
            arrayIndices = MLX.broadcast(arrays: arrayIndices)

            // the shape of the broadcast value
            let broadcastShape =
                arrayIndices[0].shape.asInt32 + arange(axis + 1, self.ndim).map { dim($0) }

            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: MLX.ones(axis + 1), at: arrayIndices[0].ndim)

            let update = newValue.broadcast(to: broadcastShape).reshaped(updateShape)

            let axes = arange(axis + 1)
            self.update(scattered(indices: arrayIndices, updates: update, axes: axes))
        }
    }

}
