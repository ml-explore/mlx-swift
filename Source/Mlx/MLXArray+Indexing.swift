import Foundation
import Cmlx

// TODO: will these collide with free function names? do we need them outside this file?  find a home

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

extension MLXArray : Sequence {
    
    public struct MLXArrayIterator : IteratorProtocol {
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
    /// ```
    /// for row in array {
    ///     row...
    /// }
    /// ```
    public func makeIterator() -> MLXArrayIterator {
        MLXArrayIterator(count: self.dim(0), array: self)
    }
}

extension MLXArray {
    
    /// Replace the interior ctx (`mlx_array` pointer) with a new value
    @inline(__always)
    private func update(array: MLXArray) {
        mlx_free(ctx)
        mlx_retain(array.ctx)
        self.ctx = array.ctx
    }
    
    /// Replace the interior ctx (`mlx_array` pointer) with a new value by transferring ownership
    @inline(__always)
    private func update(ctx: OpaquePointer) {
        if ctx != self.ctx {
            mlx_free(self.ctx)
            self.ctx = ctx
        }
    }
    
    /// allow addressing as a positive index or negative (from end)
    private func resolve(_ index: Int, _ axis: Int) -> MLXArray {
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
                resolve($1, $0)
            }
    }
    
    private func resolve(_ rangeExpression: any RangeExpression<Int>, _ axis: Int) -> (Int32, Int32) {
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
    /// ```
    /// // print the contents of row 1
    /// print(array[1])
    ///
    /// // set the value of row 1 to [23] (with broadcast)
    /// array[1] = 23
    /// ```
    public subscript(index: Int, stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            // see mlx_get_item_int
            return take(resolve(index, 0), axis: 0, stream: stream)
        }
        set {
            // see mlx_set_item_int()
            
            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")
                        
            // trim off any singleton dimensions
            let updateShape = Array(newValue.shape.trimmingPrefix { $0 == 1 })

            // this is the shape we will set via broadcast.
            var broadcastShape = self.shape
            broadcastShape[0] = 1
            
            let expanded = newValue.reshape(updateShape).broadcast(to: broadcastShape)

            let indices = [resolve(index, 0)]
            self.update(array: scatter(indices: indices, updates: expanded, axes: [0]))
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
    /// ```
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
            let g = mlx_gather(ctx, i.map { $0.ctx }, i.count, axes, axes.count, sliceSizes, sliceSizes.count, stream.ctx)!
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
            let updateShape = Mlx.ones(indices.count) + broadcastShape
            
            let update = newValue.broadcast(to: broadcastShape).reshape(updateShape)
                   
            let i = resolve(indices)
            let axes = arange(indices.count)
            self.update(array: scatter(indices: i, updates: update, axes: axes))
        }
    }
    
    @inlinable
    public subscript(ranges: any RangeExpression<Int>..., stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            self[ranges, stream: stream]
        }
        set {
            self[ranges, stream: stream] = newValue
        }
    }
    
    public subscript(ranges: [any RangeExpression<Int>], stream stream: StreamOrDevice = .default) -> MLXArray {
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
            
            return MLXArray(mlx_slice(ctx, starts, starts.count, stops, stops.count, strides, strides.count, stream.ctx))
        }
        set {
            // see mlx_set_item_nd
            
            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            var arrayIndices = [MLXArray]()
            for (i, r) in ranges.enumerated() {
                let (lower, upper) = resolve(r, i)
                let indices = arange(lower, upper)
                var indexShape = Mlx.ones(ranges.count).map { Int($0) }
                indexShape[i] = indices.count
                arrayIndices.append(MLXArray(indices, indexShape))
            }
            
            // conform the shapes to each other
            arrayIndices = Mlx.broadcast(arrays: arrayIndices)
            
            // the shape of the broadcast value
            let broadcastShape = arrayIndices[0].shape.asInt32 + arange(ranges.count, self.ndim).map { dim($0) }
            
            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: Mlx.ones(ranges.count), at: arrayIndices[0].ndim)
            
            let update = newValue.broadcast(to: broadcastShape).reshape(updateShape)
                   
            let axes = arange(ranges.count)
            self.update(array: scatter(indices: arrayIndices, updates: update, axes: axes))
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

            let arrayIndices = Mlx.broadcast(arrays: indices)

            // shape of the result is based on the shape of the broadcast inputs
            // and padded by any remaining dimensions in the array
            let resultShape = arrayIndices[0].shape.asInt32 + arange(indices.count, self.ndim).map { dim($0) }

            return MLXArray(mlx_gather(ctx, arrayIndices.map { $0.ctx }, indices.count, axes, axes.count, sliceSizes, sliceSizes.count, stream.ctx)).reshape(resultShape)
        }
        set {
            // see mlx_set_item_nd
            
            // conform the shapes to each other
            let arrayIndices = Mlx.broadcast(arrays: indices)
            
            // the shape of the broadcast value
            let broadcastShape = arrayIndices[0].shape.asInt32 + arange(indices.count, self.ndim).map { dim($0) }
            
            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: Mlx.ones(indices.count), at: arrayIndices[0].ndim)
            
            let update = newValue.broadcast(to: broadcastShape).reshape(updateShape)
                   
            let axes = arange(indices.count)
            self.update(array: scatter(indices: arrayIndices, updates: update, axes: axes))
        }
    }
}
