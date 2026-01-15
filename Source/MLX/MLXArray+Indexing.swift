// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

private func int32Range(_ count: Int) -> [Int32] {
    Array(0 ..< Int32(count))
}

private func int32Range(_ start: Int, _ end: Int) -> [Int32] {
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

    func leadingSingletonDimensionsRemoved(stream: StreamOrDevice = .default) -> MLXArray {
        let shape = self.shape.asInt32
        let newShape = Array(shape.trimmingPrefix { $0 == 1 })
        if shape != newShape {
            return self.reshaped(newShape.isEmpty ? [1] : newShape, stream: stream)
        } else {
            return self
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

    @inlinable
    func resolve(index: Int32, axis: Int) -> MLXArray {
        if index < 0 {
            return MLXArray(index + dim(axis).int32)
        } else {
            return MLXArray(index)
        }
    }

    private func resolve(_ rangeExpression: some RangeExpression<Int>, _ axis: Int) -> (
        Int32, Int32
    ) {
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

    /// Indexing using an integer index with an axis (deprecated).
    ///
    /// This method supports a range expression with an axis, for example this python code:
    ///
    /// ```python
    /// # python
    /// array[..., 3]
    /// ```
    ///
    /// This can be translated to swift as:
    ///
    /// ```swift
    /// array[.ellipsis, 3]
    /// ```
    ///
    /// This particular subscript operator is deprecated in favor of that, but for
    /// completeness the equivalent would be:
    ///
    /// ```swift
    /// array[3, axis: -1]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    @available(*, deprecated, message: "please use subscript(.ellipsis, 3) or equivalent")
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

            // this is the shape we will set via broadcast.
            let axis = MLX.resolve(axis: axis, ndim: ndim)
            var broadcastShape = self.shape
            broadcastShape[axis] = 1

            let expanded =
                newValue
                .leadingSingletonDimensionsRemoved(stream: stream)
                .broadcast(to: broadcastShape.asInt32)

            let indices = [resolve(index: index, axis: axis)]
            self._updateInternal(scattered(indices: indices, updates: expanded, axes: [axis.int32]))
        }
    }

    /// Indexing using a range expression (e.g. `0 ..< 5`) with an axis (deprecated).
    ///
    /// This method supports a range expression with an axis, for example this python code:
    ///
    /// ```python
    /// # python
    /// array[..., 0:3]
    /// ```
    ///
    /// This can be translated to swift as:
    ///
    /// ```swift
    /// array[.ellipsis, 0 ..< 3]
    /// ```
    ///
    /// This particular subscript operator is deprecated in favor of that, but for
    /// completeness the equivalent would be:
    ///
    /// ```swift
    /// array[0 ..< 3, axis: -1]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    @available(*, deprecated, message: "please use subscript(.ellipsis, 0 ..< 3) or equivalent")
    public subscript(range: some RangeExpression<Int>, axis axis: Int,
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

            var result = mlx_array_new()
            mlx_slice(
                &result,
                ctx, starts, starts.count, stops, stops.count, strides, strides.count,
                stream.ctx)
            return MLXArray(result)
        }
        set {
            // this is [0 ..., 0 ..., range] where the number of full range leading expressions
            let axis = MLX.resolve(axis: axis, ndim: ndim)
            let prefix = (0 ..< axis).map { 0 ..< dim($0) }.map { $0.mlxArrayIndexOperation }
            let range = (range as! MLXArrayIndex).mlxArrayIndexOperation

            self[operations: prefix + [range], stream: stream] = newValue
        }
    }

    /// Indexing using a stride (deprecated).
    ///
    /// This method supports strided access, similar to python:
    ///
    /// ```python
    /// # python, access strided contents of array
    /// evens = array[..., ::2]
    /// ```
    ///
    /// This can be translated to swift as:
    ///
    /// ```swift
    /// evens = array[.ellipsis, .stride(by: 2)]
    /// ```
    ///
    /// This particular subscript operator is deprecated in favor of that, but for
    /// completeness the equivalent would be:
    ///
    /// ```swift
    /// let evens = array[stride: 2, axis: -1]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    /// - ``asStrided(_:_:strides:offset:stream:)``
    @available(
        *, deprecated, message: "please use subscript(.ellipsis, .stride(by: 2)) or equivalent"
    )
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

            var result = mlx_array_new()
            mlx_slice(
                &result,
                ctx, starts, starts.count, stops, stops.count, strides, strides.count,
                stream.ctx)
            return MLXArray(result)
        }
        set {
            // see mlx_set_item_nd

            precondition(dtype == newValue.dtype, "\(dtype) != \(newValue.dtype)")

            var arrayIndices = [MLXArray]()

            // add full range slices up to the axis
            let axis = MLX.resolve(axis: axis, ndim: ndim)

            for i in 0 ..< axis {
                let indices = int32Range(0, dim(i))
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
                arrayIndices[0].shape.asInt32 + int32Range(axis + 1, self.ndim).map { dim($0) }

            // compute the scatter shape
            var updateShape = broadcastShape
            updateShape.insert(contentsOf: MLX.ones(axis + 1), at: arrayIndices[0].ndim)

            let update = newValue.broadcast(to: broadcastShape).reshaped(updateShape)

            let axes = int32Range(axis + 1)
            self._updateInternal(scattered(indices: arrayIndices, updates: update, axes: axes))
        }
    }

    /// General array indexing with encoded operations.
    ///
    /// Typically used via the general array indexing subscript:
    ///
    /// ```swift
    /// array[0, .ellipsis, 0 ..< 5]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    subscript(
        operations operations: [MLXArrayIndexOperation],
        stream stream: StreamOrDevice = .default
    ) -> MLXArray {
        get {
            switch operations.count {
            case 0:
                return self
            case 1:
                return getItem(src: self, operation: operations[0], stream: stream)
            default:
                return getItemND(src: self, operations: operations, stream: stream)
            }
        }
        set {
            if let result = updateSlice(src: self, operations: operations, update: newValue) {
                self._updateInternal(result)
                return
            }

            let (indices, update, axes) = scatterArguments(
                src: self, operations: operations, update: newValue, stream: stream)
            if !indices.isEmpty {
                let indices_vector = new_mlx_vector_array(indices)
                defer { mlx_vector_array_free(indices_vector) }

                var result = mlx_array_new()
                mlx_scatter(
                    &result, self.ctx, indices_vector, update.ctx, axes, axes.count, stream.ctx)
                mlx_array_set(&self.ctx, result)
                mlx_array_free(result)
                return
            } else {
                self._updateInternal(update)
                return
            }
        }
    }

    /// General array indexing.
    ///
    /// This implements general array indexing in the same way that the python version of MLX implements it.  This
    /// is equivalent to numpy style indexing.
    ///
    /// ```swift
    /// // index by integer
    /// array[1]
    ///
    /// // index by multiple integers
    /// array[1, 3]
    ///
    /// // index by range expression
    /// // python: [1:5]
    /// array[1 ..< 5]
    ///
    /// // full range slice
    /// // python: [:]
    /// array[0 ...]
    ///
    /// // slice with stride of 2
    /// // python: [::2]
    /// array[.stride(by: 2)]
    ///
    /// // ellipsis operator (consume all remaining axes)
    /// // python: [..., 3]
    /// array[.ellipsis, 3]
    ///
    /// // newaxis operator (insert a new axis of size 1)
    /// // python: [None]
    /// array[.newAxis]
    ///
    /// // using another MLXArray as an index
    /// let i: MLXArray
    /// array[i]
    /// ```
    ///
    /// These can be combined in any way with the following restrictions:
    ///
    /// - `.ellipsis` can only be used once in an indexing operation
    /// - `.newAxis` cannot be used in a set operation, e.g. `array[.newAxis] = MLXArray(1)` is invalid
    /// - the number of axes given must be equal or less than the number of axes in the source array
    ///
    /// ### See Also
    /// - <doc:indexing>
    /// - <doc:converting-python>
    /// - ``MLXArrayIndex/ellipsis``
    /// - ``MLXArrayIndex/newAxis``
    /// - ``MLXArrayIndex/stride(from:to:by:)``
    /// - ``MLXArray/at``
    public subscript(indices: any MLXArrayIndex..., stream stream: StreamOrDevice = .default)
        -> MLXArray
    {
        get {
            // convert the e.g. Int, MLXArray, Range, etc. into operations
            self[operations: indices.map { $0.mlxArrayIndexOperation }, stream: stream]
        }
        set {
            self[operations: indices.map { $0.mlxArrayIndexOperation }, stream: stream] = newValue
        }
    }

    /// General array indexing.
    ///
    /// See ``MLXArray/subscript(_:stream:)-375a0``
    public subscript(indices: [any MLXArrayIndex], stream stream: StreamOrDevice = .default)
        -> MLXArray
    {
        get {
            // convert the e.g. Int, MLXArray, Range, etc. into operations
            self[operations: indices.map { $0.mlxArrayIndexOperation }, stream: stream]
        }
        set {
            self[operations: indices.map { $0.mlxArrayIndexOperation }, stream: stream] = newValue
        }
    }

}

// MARK: - Support

func countNonNewAxisOperations(_ operations: some Sequence<MLXArrayIndexOperation>) -> Int {
    operations
        .filter { !$0.isNewAxis }
        .count
}

/// Replace `.ellipsis` operations with `.slice` operations for the expanded axes.
///
/// - Parameters:
///   - shape: array shape
///   - operations: operations
/// - Returns: new operations with `.ellipsis` replaced
func expandEllipsisOperations(shape: [Int32], operations: [MLXArrayIndexOperation])
    -> [MLXArrayIndexOperation]
{
    // equivalent of mlx_expand_ellipsis

    let ellipsisCount = operations.lazy.filter { $0.isEllipsis }.count
    if ellipsisCount == 0 {
        return operations
    }
    if ellipsisCount > 1 {
        fatalError("multiple .ellipsis not allowed: \(operations)")
    }

    let prefix = operations.prefix { !$0.isEllipsis }
    let suffix = operations.suffix(from: prefix.count + 1)

    // inject full range slices in place of the ellipsis
    let expandRange =
        countNonNewAxisOperations(prefix) ..< (shape.count - countNonNewAxisOperations(suffix))
    let expand = expandRange.map {
        MLXArrayIndexOperation.slice(.init(start: 0, end: shape[$0], stride: 1))
    }

    return prefix + expand + suffix
}

// MARK: - index get

/// Single operation MLXArray subscript get.
///
/// This provides optimized operations for simple forms of subscripts.
///
/// - Parameters:
///   - src: input array
///   - operation: single index operation
/// - Returns: MLXArray get result
func getItem(src: MLXArray, operation: MLXArrayIndexOperation, stream: StreamOrDevice = .default)
    -> MLXArray
{
    switch operation {
    case .ellipsis:
        return src

    case .newAxis:
        return src.expandedDimensions(axis: 0, stream: stream)

    case .index(let index):
        return src.take(src.resolve(index: index, axis: 0), axis: 0, stream: stream)

    case .slice(let slice):
        let ndim = src.ndim
        var starts = [Int32](repeating: 0, count: ndim)
        var ends = src.shape.asInt32
        var strides = [Int32](repeating: 1, count: ndim)

        let size = ends[0]
        starts[0] = slice.start(size)
        ends[0] = slice.end(size)
        strides[0] = slice.stride

        var result = mlx_array_new()
        mlx_slice(
            &result,
            src.ctx, starts, starts.count, ends, ends.count, strides, strides.count,
            stream.ctx)
        return MLXArray(result)

    case .array(let indices):
        return src.take(indices, axis: 0, stream: stream)
    }
}

/// Full implementation of python/numpy indexing.
///
/// - Parameters:
///   - array: source array
///   - operations: array of index operations
/// - Returns: array with the indices applied
func getItemND(
    src: MLXArray, operations: [MLXArrayIndexOperation], stream: StreamOrDevice = .default
) -> MLXArray {
    // See mlx_get_item_nd

    var src = src

    // The plan is as follows:
    // 1. Replace the ellipsis with a series of slice(None)
    // 2. Loop over the indices and calculate the gather indices
    // 3. Calculate the remaining slices and reshapes

    let shape32 = src.shape.asInt32
    let operations = expandEllipsisOperations(shape: shape32, operations: operations)

    // Gather handling

    // compute gatherFirst -- this will be true if there is:
    // - a leading array or index operation followed by
    // - a non index/array (e.g. a slice)
    // - an int/array operation
    //
    // - and there is at least one array operation (hanled below with haveArray)
    var gatherFirst = false
    var haveArrayOrIndex = false
    var haveNonArray = false
    for item in operations {
        if item.isArrayOrIndex {
            if haveArrayOrIndex && haveNonArray {
                gatherFirst = true
                break
            }
            haveArrayOrIndex = true
        } else {
            haveNonArray = haveNonArray || haveArrayOrIndex
        }
    }

    let arrayCount = operations.lazy.filter { $0.isArray }.count
    let haveArray = arrayCount > 0

    var remainingIndices = [MLXArrayIndexOperation]()
    if haveArray {
        // apply all the operations (except for .newAxis) up to and including the
        // final .array operation (array operations are implemented via gather)
        let lastArrayOrIndex = operations.lastIndex { $0.isArrayOrIndex }!
        let gatherIndices = operations.prefix(through: lastArrayOrIndex).filter { !$0.isNewAxis }
        let (maxDimensions, gathered) = gatherND(
            src: src, operations: gatherIndices, gatherFirst: gatherFirst)
        src = gathered

        // Reassemble the indices for the slicing or reshaping if there are any
        if gatherFirst {
            // handle the gathered .array operations
            remainingIndices.append(
                contentsOf: Array(repeating: .slice(.full), count: maxDimensions))

            // copy any newAxis in the gatherIndices through.  any slices get
            // copied in as full range (already applied)
            for item in operations.prefix(through: lastArrayOrIndex) {
                if item.isNewAxis {
                    remainingIndices.append(item)
                } else if item.isSlice {
                    remainingIndices.append(.slice(.full))
                }
            }

            // append the remaining operations
            remainingIndices.append(contentsOf: operations.suffix(from: lastArrayOrIndex + 1))

        } else {
            // !gatherFirst
            for item in operations {
                if item.isArrayOrIndex {
                    break
                } else if item.isNewAxis {
                    remainingIndices.append(item)
                } else {
                    remainingIndices.append(.slice(.full))
                }
            }

            // handle the trailing gathers
            remainingIndices.append(
                contentsOf: Array(repeating: .slice(.full), count: maxDimensions))

            // and the remaining operations
            remainingIndices.append(contentsOf: operations.suffix(from: lastArrayOrIndex + 1))
        }
    }

    if haveArray && remainingIndices.isEmpty {
        return src
    }
    if remainingIndices.isEmpty {
        remainingIndices = operations
    }

    // Slice handling
    let ndim = src.ndim
    var starts = [Int32](repeating: 0, count: ndim)
    var ends = src.shape.asInt32
    var strides = [Int32](repeating: 1, count: ndim)
    var squeezeNeeded = false
    var axis = 0

    for item in remainingIndices {
        switch item {
        case .newAxis:
            continue

        case .index(var index):
            if !haveArray {
                index = index < 0 ? index + src.dim(axis).int32 : index
                starts[axis] = index
                ends[axis] = index + 1
                squeezeNeeded = true
            }

        case .slice(let slice):
            let size = src.dim(axis).int32
            starts[axis] = slice.start(size)
            ends[axis] = slice.end(size)
            strides[axis] = slice.stride

        default:
            fatalError("Unexpected item in remainingIndices: \(item)")
        }
        axis += 1
    }

    var result = mlx_array_new()
    mlx_slice(
        &result,
        src.ctx, starts, starts.count, ends, ends.count, strides, strides.count, stream.ctx)
    src = MLXArray(result)

    // Unsqueeze handling
    if remainingIndices.count > src.ndim || squeezeNeeded {
        var newShape = [Int]()
        var axis = 0
        for item in remainingIndices {
            if item.isNewAxis {
                newShape.append(1)
            } else if squeezeNeeded && item.isIndex {
                axis += 1
            } else {
                newShape.append(src.dim(axis))
                axis += 1
            }
        }
        newShape.append(contentsOf: src.shape.suffix(from: axis))

        src = src.reshaped(newShape)
    }

    return src
}

/// Gather N dimensional.
///
/// - Parameters:
///   - src: the source array
///   - operations: array of ``MLXArrayIndexOperation`` with all
///   ``MLXArrayIndexOperation/ellipsis`` resolved to ``MLXArrayIndexOperation/slice(_:)``
///   and all ``MLXArrayIndexOperation/newAxis`` removed.
///   - gatherFirst: if true put the int/array indices at the front else put them at the back
/// - Returns: maxDimensions of any of the `.array` operations and the result MLXArray
func gatherND(
    src: MLXArray, operations: [MLXArrayIndexOperation], gatherFirst: Bool,
    stream: StreamOrDevice = .default
) -> (Int, MLXArray) {
    var maxDimensions = 0
    var sliceCount = 0
    var isSlice = [Bool](repeating: false, count: operations.count)
    var gatherIndices = [MLXArray]()

    let shape32 = src.shape.asInt32

    // prepare the gatherIndices
    for (i, item) in operations.enumerated() {
        switch item {
        case .ellipsis, .newAxis:
            // the ellipsis has already been converted to .slice operations
            // and .newAxis has been filtered out of the operations
            fatalError("unexpected item: \(item) in gatherND")

        case .index(let index):
            gatherIndices.append(src.resolve(index: index, axis: i))

        case .slice(let slice):
            sliceCount += 1
            isSlice[i] = true

            let size = shape32[i]
            gatherIndices.append(
                MLXArray(
                    stride(
                        from: slice.absoluteStart(size), to: slice.absoluteEnd(size),
                        by: Int(slice.stride))))

        case .array(let array):
            maxDimensions = max(array.ndim, maxDimensions)
            gatherIndices.append(array)
        }
    }

    // reshape them so that the int/array indices are first
    if gatherFirst {

        // if there are slices
        if sliceCount > 0 {
            var sliceIndex = 0
            for (i, item) in gatherIndices.enumerated() {
                if isSlice[i] {
                    var newShape = Array(repeating: 1, count: maxDimensions + sliceCount)
                    newShape[maxDimensions + sliceIndex] = item.dim(0)
                    gatherIndices[i] = item.reshaped(newShape)
                    sliceIndex += 1

                } else {
                    let newShape = item.shape + Array(repeating: 1, count: sliceCount)
                    gatherIndices[i] = item.reshaped(newShape)
                }
            }
        }

    } else {
        // reshape them so that the int/array indices are last
        for (i, item) in gatherIndices.prefix(sliceCount).enumerated() {
            var newShape = Array(repeating: 1, count: maxDimensions + sliceCount)
            newShape[i] = item.dim(0)
            gatherIndices[i] = item.reshaped(newShape)
        }
    }

    // Do the gather
    let indices = new_mlx_vector_array(gatherIndices)
    defer { mlx_vector_array_free(indices) }
    let axes = Array(0 ..< operations.count.int32)
    var sliceSizes = shape32
    for i in 0 ..< operations.count {
        sliceSizes[i] = 1
    }

    var tmp = mlx_array_new()
    mlx_gather(&tmp, src.ctx, indices, axes, axes.count, sliceSizes, sliceSizes.count, stream.ctx)
    let gathered = MLXArray(tmp)
    let gatheredShape = gathered.shape

    // Squeeze the dims
    let outputShape = Array(
        gatheredShape[0 ..< (maxDimensions + sliceCount)]
            + gatheredShape[(maxDimensions + sliceCount + operations.count)...])
    let result = gathered.reshaped(outputShape, stream: stream)

    return (maxDimensions, result)
}

// MARK: - index set (slice)

func updateSlice(
    src: MLXArray, operations: [MLXArrayIndexOperation], update: MLXArray,
    stream: StreamOrDevice = .default
) -> MLXArray? {
    // See mlx_slice_update
    let ndim = src.ndim
    if ndim == 0 || operations.count == 0 {
        return nil
    }

    // Can't route to slice update if any arrays are present
    if operations.contains(where: { $0.isArray }) {
        return nil
    }

    // Remove leading singletons dimensions from the update
    var update = update.leadingSingletonDimensionsRemoved(stream: stream)

    // Build slice update params
    var starts = [Int32](repeating: 0, count: ndim)
    var ends = src.shape.asInt32
    var strides = [Int32](repeating: 1, count: ndim)

    // If it's just a simple slice, just do a slice update and return
    if operations.count == 1, case .slice(let slice) = operations[0] {
        let size = src.dim(0).int32
        starts[0] = slice.start(size)
        ends[0] = slice.end(size)
        strides[0] = slice.stride

        var result = mlx_array_new()
        mlx_slice_update(
            &result,
            src.ctx, update.ctx, starts, starts.count, ends, ends.count, strides, strides.count,
            stream.ctx)
        return MLXArray(result)
    }

    // Expand ellipses into a series of ':' (full slice) slices
    let operations = expandEllipsisOperations(shape: src.shape.asInt32, operations: operations)

    // If no non-None indices return the broadcasted update
    let nonNewAxisOperationCount = countNonNewAxisOperations(operations)
    if nonNewAxisOperationCount == 0 {
        return broadcast(update, to: src.shape)
    }

    // Process entries
    var updateReshape = [Int](repeating: 0, count: src.ndim)
    var axis = src.ndim - 1
    var updateAxis = update.ndim - 1

    while axis >= nonNewAxisOperationCount {
        if updateAxis >= 0 {
            updateReshape[axis] = update.dim(updateAxis)
            updateAxis -= 1
        } else {
            updateReshape[axis] = 1
        }
        axis -= 1
    }

    for item in operations.reversed() {
        switch item {
        case .ellipsis, .array:
            // these were replaced or rejected earlier
            fatalError("unexpected item \(item) in updateSlice")

        case .index(let index):
            let size = src.dim(axis).int32
            let index = index < 0 ? index + size : index
            starts[axis] = index
            ends[axis] = index + 1

            updateReshape[axis] = 1
            axis -= 1

        case .slice(let slice):
            let size = src.dim(axis).int32
            starts[axis] = slice.start(size)
            ends[axis] = slice.end(size)
            strides[axis] = slice.stride

            if updateAxis >= 0 {
                updateReshape[axis] = update.dim(updateAxis)
                updateAxis -= 1
            } else {
                updateReshape[axis] = 1
            }
            axis -= 1

        case .newAxis:
            break
        }
    }

    update = reshaped(update, updateReshape)

    var result = mlx_array_new()
    mlx_slice_update(
        &result,
        src.ctx, update.ctx, starts, starts.count, ends, ends.count, strides, strides.count,
        stream.ctx)
    return MLXArray(result)
}

// MARK: - index set (scatter)

/// Top level scatter computation.
///
/// - Parameters:
///   - src: the input array
///   - operations: array of index operations
///   - update: the update value
/// - Returns: scatter indices, update value, scatter axes
func scatterArguments(
    src: MLXArray, operations: [MLXArrayIndexOperation], update: MLXArray,
    stream: StreamOrDevice = .default
) -> ([MLXArray], MLXArray, [Int32]) {
    // mlx_compute_scatter_args

    if operations.count == 1 {
        switch operations[0] {
        case .ellipsis:
            fatalError("Unable to update array with .ellipsis argument")
        case .newAxis:
            return ([], update.broadcast(to: src.shape.asInt32, stream: stream), [])
        case .index(let index):
            return scatterArguments(src: src, index: index, update: update, stream: stream)
        case .slice(let slice):
            return scatterArguments(src: src, slice: slice, update: update, stream: stream)
        case .array(let array):
            return scatterArguments(src: src, array: array, update: update, stream: stream)
        }
    }

    // mlx_scatter_args_nd

    let shape32 = src.shape.asInt32

    let operations = expandEllipsisOperations(shape: shape32, operations: operations)
    var update = update.leadingSingletonDimensionsRemoved(stream: stream)

    // If no non-newAxis indices return the broadcasted update
    let nonNewAxisOperationCount = countNonNewAxisOperations(operations)
    if nonNewAxisOperationCount == 0 {
        return (
            [],
            update.broadcast(to: shape32, stream: stream),
            []
        )
    }

    // Analyse the types of the indices
    var maxDimensions = 0
    var arraysFirst = false
    var countNewAxis = 0
    var countSlices = 0
    var countArrays = 0
    var countStridedSlices = 0
    var countSimpleSlicesPost = 0

    var haveArray = false
    var haveNonArray = false

    for item in operations {
        switch item {
        case .ellipsis:
            // has been converted to slices
            fatalError("unexpected item: \(item) in scatterArguments")

        case .newAxis:
            haveNonArray = true
            countNewAxis += 1

        case .index:
            // ignore
            break

        case .slice(let slice):
            haveNonArray = haveArray
            countSlices += 1
            if slice.stride != 1 {
                countStridedSlices += 1
                countSimpleSlicesPost = 0
            } else {
                countSimpleSlicesPost += 1
            }

        case .array(let array):
            haveArray = true
            if haveArray && haveNonArray {
                arraysFirst = true
            }
            maxDimensions = max(array.ndim, maxDimensions)
            countArrays += 1
            countSimpleSlicesPost = 0
        }
    }

    // We have index dims for the arrays, strided slices (implemented as arrays), none
    var indexDimensions = maxDimensions + countNewAxis + countSlices - countSimpleSlicesPost

    // If we have simple non-strided slices, we also attach an index for that
    if indexDimensions == 0 {
        indexDimensions = 1
    }

    // Go over each index type and translate to the needed scatter args
    var arrayIndices = [MLXArray]()
    var sliceNumber = 0
    var arrayNumber = 0
    var axis = 0

    // We collect the shapes of the slices and updates during this process
    var updateShape = Array(repeating: 1, count: nonNewAxisOperationCount)
    var sliceShapes = [Int]()

    for item in operations {
        switch item {
        case .ellipsis:
            // has been converted to slices
            fatalError("unexpected item: \(item) in scatterArguments")

        case .newAxis:
            // We only use the newAxis's for bookeeping dimensions
            sliceNumber += 1

        case .index(let index):
            arrayIndices.append(src.resolve(index: index, axis: axis))
            updateShape[axis] = 1
            axis += 1

        case .slice(let slice):
            let size = src.dim(axis).int32
            let start = slice.absoluteStart(size)
            let end = slice.absoluteEnd(size)
            let stride = slice.stride

            var indexShape = Array(repeating: 1, count: indexDimensions)

            // If it's a simple slice, we only need to add the start index
            if arrayNumber >= countArrays && countStridedSlices <= 0 && stride == 1 {
                let index = MLXArray(start).reshaped(indexShape, stream: stream)
                sliceShapes.append(Int(end - start))
                arrayIndices.append(index)

                // Add the shape to the update
                updateShape[axis] = Int(sliceShapes.last!)

            } else {
                // Otherwise we expand the slice into indices using arange
                let index = MLXArray(Swift.stride(from: start, to: end, by: Int(stride)))
                let location = sliceNumber + (arraysFirst ? maxDimensions : 0)
                indexShape[location] = index.size
                arrayIndices.append(index.reshaped(indexShape, stream: stream))

                sliceNumber += 1
                countStridedSlices -= 1

                // Add the shape to the update
                updateShape[axis] = 1
            }

            axis += 1

        case .array(let array):
            // Place the arrays in the correct dimension
            let start = (arraysFirst ? 0 : sliceNumber) + maxDimensions - array.ndim
            var newShape = Array(repeating: 1, count: indexDimensions)

            for j in 0 ..< array.ndim {
                newShape[start + j] = array.dim(j)
            }

            arrayIndices.append(array.reshaped(newShape, stream: stream))
            arrayNumber += 1
            if !arraysFirst && arrayNumber == countArrays {
                sliceNumber += maxDimensions
            }

            // Add the shape to the update
            updateShape[axis] = 1
            axis += 1
        }
    }

    // Broadcast the update to the indices and slices
    arrayIndices = broadcast(arrays: arrayIndices)
    let updateShapeBroadcast =
        arrayIndices[0].shape + sliceShapes + src.shape.dropFirst(nonNewAxisOperationCount)
    update = broadcast(update, to: updateShapeBroadcast, stream: stream)

    // Reshape the update with the size-1 dims for the int and array indices
    let updateReshape =
        arrayIndices[0].shape + updateShape + src.shape.dropFirst(nonNewAxisOperationCount)

    update = update.reshaped(updateReshape, stream: stream)

    return (
        arrayIndices,
        update,
        Array(0 ..< Int32(arrayIndices.count))
    )
}

/// Scatter operation for a single integer index.
///
/// - Parameters:
///   - src: the input array
///   - operations: array of index operations
///   - update: the update value
/// - Returns: scatter indices, update value, scatter axes
func scatterArguments(
    src: MLXArray, index: Int32, update: MLXArray, stream: StreamOrDevice = .default
) -> ([MLXArray], MLXArray, [Int32]) {
    // mlx_scatter_args_int

    // Remove any leading singleton dimensions from the update
    // and then broadcast update to shape of src[0, ...]
    let update = update.leadingSingletonDimensionsRemoved(stream: stream)

    var shape = src.shape.asInt32
    shape[0] = 1

    return (
        [src.resolve(index: index, axis: 0)],
        update.broadcast(to: shape, stream: stream),
        [0]
    )
}

/// Scatter operation for a single MLXArray index.
///
/// - Parameters:
///   - src: the input array
///   - operations: array of index operations
///   - update: the update value
/// - Returns: scatter indices, update value, scatter axes
func scatterArguments(
    src: MLXArray, array: MLXArray, update: MLXArray, stream: StreamOrDevice = .default
) -> ([MLXArray], MLXArray, [Int32]) {
    // mlx_scatter_args_array

    // trim leading singleton dimensions
    var update = update.leadingSingletonDimensionsRemoved(stream: stream)

    // The update shape must broadcast with indices.shape + [1] + src.shape[1:]
    var updateShape = (array.shape + src.shape.dropFirst()).asInt32
    update = update.broadcast(to: updateShape, stream: stream)

    updateShape.insert(1, at: array.ndim)
    update = update.reshaped(updateShape)

    return (
        [array],
        update,
        [0]
    )
}

/// Scatter operation for a single MLXSlice index.
///
/// - Parameters:
///   - src: the input array
///   - operations: array of index operations
///   - update: the update value
/// - Returns: scatter indices, update value, scatter axes
func scatterArguments(
    src: MLXArray, slice: MLXSlice, update: MLXArray, stream: StreamOrDevice = .default
) -> ([MLXArray], MLXArray, [Int32]) {
    // mlx_scatter_args_slice

    // If none slice is requested broadcast the update
    // to the src size and return it.
    if slice.isFull {
        return (
            [],
            update
                .leadingSingletonDimensionsRemoved(stream: stream)
                .broadcast(to: src.shape.asInt32, stream: stream),
            []
        )
    }

    let size = src.dim(0).int32
    let start = slice.start(size)
    let end = slice.end(size)
    let stride = slice.stride

    // If simple stride
    if stride == 1 {
        var update = update.leadingSingletonDimensionsRemoved(stream: stream)

        // Broadcast update to slice size
        let updateBroadcastShape = [1, end - start] + src.shape.dropFirst().asInt32
        update = update.broadcast(to: updateBroadcastShape, stream: stream)

        let indices = MLXArray(start).reshaped([1])
        return (
            [indices],
            update,
            [0]
        )
    } else {
        // stide != 1, convert the slice to an array
        return scatterArguments(
            src: src, array: MLXArray(Swift.stride(from: start, to: end, by: Int(stride))),
            update: update, stream: stream)
    }
}

// MARK: - MLXArrayIndexOperation and MLXArrayIndex

/// MLXArray index operations.
///
/// Typically produced internally by normal index operations:
///
/// ```swift
/// let a = MLXArray(...)
///
/// print(a[1 ..< 3, .ellipsis, 3])
/// ```
///
/// ### See Also
/// - <doc:indexing>
public enum MLXArrayIndexOperation: CustomStringConvertible {
    /// `...` or `Ellipsis` in python -- this will expand to be full range slices of all collected axes
    case ellipsis

    /// `None` or `newaxis` in python
    case newAxis

    /// A single integer index
    case index(Int32)

    /// A slice, e.g. from `0 ..< 3` or `.stride(by: 2)`
    case slice(MLXSlice)

    /// An MLXArray index, see https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    case array(MLXArray)

    var isEllipsis: Bool {
        switch self {
        case .ellipsis: true
        default: false
        }
    }

    var isNewAxis: Bool {
        switch self {
        case .newAxis: true
        default: false
        }
    }

    var isIndex: Bool {
        switch self {
        case .index: true
        default: false
        }
    }

    var isSlice: Bool {
        switch self {
        case .slice: true
        default: false
        }
    }

    var isArray: Bool {
        switch self {
        case .array: true
        default: false
        }
    }

    var isArrayOrIndex: Bool {
        switch self {
        case .index, .array: true
        default: false
        }
    }

    public var description: String {
        switch self {
        case .ellipsis:
            return ".ellipsis"
        case .newAxis:
            return ".newAxis"
        case .index(let v):
            return "\(v)"
        case .slice(let v):
            return "\(v)"
        case .array(let v):
            return "\(v.shape)(\(v.dtype))"
        }
    }
}

/// Protocol for values usable as an index with ``MLXArray``.
///
/// ### See Also
/// - <doc:indexing>
/// - ``MLXArray/subscript(_:stream:)-375a0``
public protocol MLXArrayIndex {

    /// Provide the represeting ``MLXArrayIndexOperation``
    var mlxArrayIndexOperation: MLXArrayIndexOperation { get }
}

extension Int: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .index(self.int32)
    }
}

extension MLXArray: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .array(self)
    }
}

public struct MLXEllipsisIndex: MLXArrayIndex, Sendable {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .ellipsis
    }
}

extension MLXArrayIndex where Self == MLXEllipsisIndex {

    /// Implementation of the `.ellipsis` MLXArray index.
    ///
    /// This is the equivalent of the `None` index in python:
    ///
    /// ```python
    /// # python
    /// array[0, ..., 1]
    /// ```
    ///
    /// Written as this in swift:
    ///
    /// ```swift
    /// // swift
    /// array[0, .ellipsis, 1]
    /// ```
    ///
    /// This is equivalent to inserting full range slices for all intermediate axes such that the
    /// `1` index fell on the last axis.
    ///
    /// ### See Also
    /// - <doc:indexing>
    public static var ellipsis: Self { Self() }
}

public struct MLXNewAxisIndex: MLXArrayIndex, Sendable {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .newAxis
    }
}

extension MLXArrayIndex where Self == MLXNewAxisIndex {

    /// Implementation of the `.newAxis`MLXArray index.
    ///
    /// This is the equivalent of the `None` index in python:
    ///
    /// ```python
    /// # python
    /// array[0, None]
    /// ```
    ///
    /// Written as this in swift:
    ///
    /// ```swift
    /// // swift
    /// array[0, .newAxis]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    public static var newAxis: Self { Self() }
}

// MARK: - MLXSlice

/// Representation of a slice in ``MLXArray`` indexing operations.
///
/// This is typically not used directly -- the various range expressions implement
/// ``MLXArrayIndex`` and can produce these, e.g.
///
/// ```swift
/// let a = MLXArray(...)
///
/// print(a[1 ..< 3])
/// print(a[.stride(to: 8, by: 2)])
/// ```
///
/// ### See Also
/// - <doc:indexing>
/// - ``MLXArrayIndex/stride(from:to:by:)``
public struct MLXSlice: Equatable, CustomStringConvertible, Sendable {

    private let _start: Int32?
    private let _end: Int32?
    private let _stride: Int32?

    /// Initialize an MLXSlice with its optional parameters.
    ///
    /// This is typically caled by the range expressions which implement ``MLXArrayIndex``:
    ///
    /// ```swift
    /// let a = MLXArray(...)
    ///
    /// print(a[1 ..< 3])
    /// print(a[.stride(to: 8, by: 2)])
    /// ```
    ///
    /// - Parameters:
    ///   - start: optional start index -- nil means 0
    ///   - end: optional end index -- nil means end of axis
    ///   - stride: optional stride -- nil means 1
    public init(start: Int32? = nil, end: Int32? = nil, stride: Int32? = nil) {
        self._start = start
        self._end = end
        self._stride = stride
    }

    /// A full range slice
    static let full = MLXSlice()

    /// `true` if this is a full range (unrestricted) slice
    public var isFull: Bool {
        (_stride == nil || _stride == 1) && (_start == nil || _start == 0) && (_end == nil)
    }

    /// Given `stride` or `1`
    public var stride: Int32 { _stride ?? 1 }

    /// Start index using numpy conventions.
    ///
    /// If `start` is not given it will be `size - 1` if ``stride`` is negative or `0` if positive.
    public func start(_ size: Int32) -> Int32 {
        _start ?? (stride < 0 ? size - 1 : 0)
    }

    /// Start index resolving negative values (e.g. for gather operations)
    public func absoluteStart(_ size: Int32) -> Int32 {
        let start = self.start(size)
        return start < 0 ? start + size : start
    }

    /// End index using numpy conventions.
    ///
    /// If `end` is not given it will be `-size - 1` if ``stride`` is negative or `size` if positive.
    public func end(_ size: Int32) -> Int32 {
        _end ?? (stride < 0 ? -size - 1 : size)
    }

    /// End index resolving negative values (e.g. for gather operations)
    public func absoluteEnd(_ size: Int32) -> Int32 {
        let end = self.end(size)
        return end < 0 ? end + size : end
    }

    public var description: String {
        if stride == 1 {
            return "\(_start?.description ?? "") ..< \(_end?.description ?? "")"
        } else {
            return "\(_start?.description ?? "") ..< \(_end?.description ?? "") : \(stride)"
        }
    }
}

extension MLXArrayIndex where Self == MLXSlice {
    /// MLXArray index for general slicing.
    ///
    /// ```swift
    /// // full range
    /// // swift: 0...
    /// // python: :
    /// array[.stride()]
    ///
    /// // full range with a stride of 2
    /// // swift: N/A
    /// // python: ::2
    /// array[.stride(by: 2)]
    ///
    /// // range
    /// // swift: 0 ..< 3
    /// // python: 0:3
    /// array[.stride(from: 0, to: 3)]
    ///
    /// // range + stride
    /// // swift: N/A
    /// // python: 0:10:3
    /// array[.stride(from: 0, to: 10, by: 3)]
    ///
    /// // negative stride (reverse)
    /// // swift: N/A
    /// // python: ::-1
    /// array[.stride(by: -1)]
    /// ```
    ///
    /// ### See Also
    /// - <doc:indexing>
    public static func stride(from start: Int? = nil, to end: Int? = nil, by stride: Int? = nil)
        -> MLXSlice
    {
        MLXSlice(start: start?.int32, end: end?.int32, stride: stride?.int32)
    }
}

/// Replacement for `Swift.stride(from:to:by:)` for MLXArray indexing.
///
/// Typicaly written with `.stride`:
///
/// ```swift
/// let a = MLXArray(...)
///
/// print(a[.stride(to: 8, by: 2)])
/// ```
///
/// ### See Also
/// - <doc:indexing>
/// - ``MLXArrayIndex/stride(from:to:by:)``
@inlinable public func stride(from start: Int? = nil, to end: Int? = nil, by stride: Int? = nil)
    -> MLXSlice
{
    MLXSlice(start: start?.int32, end: end?.int32, stride: stride?.int32)
}

extension MLXSlice: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(self)
    }
}

extension Range<Int>: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32, end: self.upperBound.int32))
    }
}

extension ClosedRange<Int>: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32, end: self.upperBound.int32 + 1))
    }
}

extension PartialRangeUpTo<Int>: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: 0, end: self.upperBound.int32))
    }
}

extension PartialRangeThrough<Int>: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: 0, end: self.upperBound.int32 + 1))
    }
}

extension PartialRangeFrom<Int>: MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32))
    }
}
