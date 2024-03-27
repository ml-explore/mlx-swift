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

    @inlinable
    func resolve(index: Int32, axis: Int) -> MLXArray {
        if index < 0 {
            return MLXArray(index + dim(axis).int32)
        } else {
            return MLXArray(index)
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

    // TODO
    // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

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
            let prefix = (0 ..< axis).map { 0 ..< dim($0) }.map { $0.mlxArrayIndexOperation }
            let range = (range as! MLXArrayIndex).mlxArrayIndexOperation

            self[operations: prefix + [range], stream: stream] = newValue
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
        
    public subscript(operations operations: [MLXArrayIndexOperation], stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            switch operations.count {
            case 0:
                return self
            case 1:
                return getItem(array: self, operation: operations[0], stream: stream)
            default:
                return getItemND(array: self, operations: operations, stream: stream)
            }
        }
        set {
            // TODO
        }
    }

    public subscript(indexes: MLXArrayIndex..., stream stream: StreamOrDevice = .default) -> MLXArray {
        get {
            self[operations: indexes.map { $0.mlxArrayIndexOperation }, stream: stream]
        }
        set {
            self[operations: indexes.map { $0.mlxArrayIndexOperation }, stream: stream] = newValue
        }
    }
    
}

func countNonNewAxisOperations(_ operations: any Sequence<MLXArrayIndexOperation>) -> Int {
    operations
        .filter { !$0.isNewAxis }
        .count
}

func expandEllipsisOperations(shape: [Int32], operations: [MLXArrayIndexOperation]) -> [MLXArrayIndexOperation] {
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
    let expandRange = countNonNewAxisOperations(prefix) ..< (shape.count - countNonNewAxisOperations(suffix))
    let expand = expandRange.map {
        MLXArrayIndexOperation.slice(.init(start: 0, end: shape[$0], stride: 1))
    }
    
    return prefix + expand + suffix
}

func getItem(array: MLXArray, operation: MLXArrayIndexOperation, stream: StreamOrDevice = .default) -> MLXArray {
    switch operation {
    case .ellipsis:
        return array
        
    case .newAxis:
        return array.expandedDimensions(axis: 0, stream: stream)
        
    case .index(let index):
        return array.take(array.resolve(index: index, axis: 0), axis: 0, stream: stream)
        
    case .slice(let slice):
        let ndim = array.ndim
        var starts = [Int32](repeating: 0, count: ndim)
        var ends = array.shape.asInt32
        var strides = [Int32](repeating: 1, count: ndim)
        
        let size = ends[0]
        starts[0] = slice.start(size)
        ends[0] = slice.end(size)
        strides[0] = slice.stride

        return MLXArray(
            mlx_slice(
                array.ctx, starts, starts.count, ends, ends.count, strides, strides.count,
                stream.ctx))

    case .array(let indices):
        return array.take(indices, axis: 0, stream: stream)
    }
}

func getItemND(array: MLXArray, operations: [MLXArrayIndexOperation], stream: StreamOrDevice = .default) -> MLXArray {
    var array = array
        
    // The plan is as follows:
    // 1. Replace the ellipsis with a series of slice(None)
    // 2. Loop over the indices and calculate the gather indices
    // 3. Calculate the remaining slices and reshapes
    
    let shape32 = array.shape.asInt32
    let operations = expandEllipsisOperations(shape: shape32, operations: operations)
    
    // Gather handling
    //
    // Check whether we have arrays or integer indices and delegate to gather_nd
    // after removing the slices at the end and all Nones (.expand)
    
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
        let lastArrayOrIndex = operations.lastIndex { $0.isArrayOrIndex }!
        
        let gatherIndices = operations.prefix(through: lastArrayOrIndex).filter { !$0.isNewAxis }
        let (maxDimensions, gathered) = gatherND(array: array, operations: gatherIndices, gatherFirst: gatherFirst)
        array = gathered
        
        // Reassemble the indices for the slicing or reshaping if there are any
        if gatherFirst {
            remainingIndices.append(contentsOf: Array(repeating: .slice(.full), count: maxDimensions))
            for item in operations.prefix(upTo: lastArrayOrIndex) {
                if item.isNewAxis {
                    remainingIndices.append(item)
                } else if item.isSlice {
                    remainingIndices.append(.slice(.full))
                }
            }
            remainingIndices.append(contentsOf: operations.suffix(from: lastArrayOrIndex + 1))
            
        } else {
            for item in operations {
                if item.isArrayOrIndex {
                    break
                } else if item.isNewAxis {
                    remainingIndices.append(item)
                } else {
                    remainingIndices.append(.slice(.full))
                }
            }
            remainingIndices.append(contentsOf: Array(repeating: .slice(.full), count: maxDimensions))
            remainingIndices.append(contentsOf: operations.suffix(from: lastArrayOrIndex + 1))
        }
    }
    
    if haveArray && remainingIndices.isEmpty {
        return array
    }
    if remainingIndices.isEmpty {
        remainingIndices = operations
    }
    
    // Slice handling
    let ndim = array.ndim
    var starts = [Int32](repeating: 0, count: ndim)
    var ends = array.shape.asInt32
    var strides = [Int32](repeating: 1, count: ndim)
    var squeezeNeeded = false
    var axis = 0
    
    for item in remainingIndices {
        switch item {
        case .newAxis:
            continue
            
        case .index(var index):
            if !haveArray {
                index = index < 0 ? index + array.dim(axis).int32 : index
                starts[axis] = index
                ends[axis] = index + 1
                squeezeNeeded = true
            }
            
        case .slice(let slice):
            let size = array.dim(axis).int32
            starts[axis] = slice.start(size)
            ends[axis] = slice.end(size)
            strides[axis] = slice.stride

        default:
            fatalError("Unexpected item in remainingIndices: \(item)")
        }
        axis += 1
    }
    
    array = MLXArray(mlx_slice(array.ctx, starts, starts.count, ends, ends.count, strides, strides.count, stream.ctx))

    // Unsqueeze handling
    if remainingIndices.count > array.ndim || squeezeNeeded {
        var newShape = [Int]()
        var axis = 0
        for item in remainingIndices {
            if item.isNewAxis {
                newShape.append(1)
            } else if squeezeNeeded && item.isIndex {
                axis += 1
            } else {
                newShape.append(array.dim(axis))
                axis += 1
            }
        }
        newShape.append(contentsOf: array.shape.suffix(from: axis))
        
        array = array.reshaped(newShape)
    }

    return array
}

func gatherND(array: MLXArray, operations: [MLXArrayIndexOperation], gatherFirst: Bool, stream: StreamOrDevice = .default) -> (Int, MLXArray) {
    var maxDimensions = 0
    var sliceCount = 0
    var isSlice = [Bool](repeating: false, count: operations.count)
    var gatherIndices = [MLXArray]()
    
    let shape32 = array.shape.asInt32
    
    // gather all the arrays
    for (i, item) in operations.enumerated() {
        switch item {
        case .ellipsis, .newAxis:
            break
        case .index(let index):
            gatherIndices.append(array.resolve(index: index, axis: i))
            
        case .slice(let slice):
            sliceCount += 1
            isSlice[i] = true
            
            let size = shape32[i]
            gatherIndices.append(MLXArray(stride(from: slice.absoluteStart(size), to: slice.absoluteEnd(size), by: Int(slice.stride))))
            
        case .array(let array):
            maxDimensions = max(array.ndim, maxDimensions)
            gatherIndices.append(array)
        }
    }
    
    // reshape them so that the int/array indices are first
    if gatherFirst {
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
    defer { mlx_free(indices) }
    let axes = Array(0 ..< operations.count.int32)
    var sliceSizes = shape32
    for i in 0 ..< operations.count {
        sliceSizes[i] = 1
    }
    
    let gathered = MLXArray(mlx_gather(array.ctx, indices, axes, axes.count, sliceSizes, sliceSizes.count, stream.ctx))
    let gatheredShape = gathered.shape
    
    // Squeeze the dims
    let outputShape = Array(gatheredShape[0 ..< (maxDimensions + sliceCount)] + gatheredShape[(maxDimensions + sliceCount + operations.count)...])
    let result = gathered.reshaped(outputShape, stream: stream)
    
    return (maxDimensions, result)
}

public enum MLXArrayIndexOperation : CustomStringConvertible {
    /// `...` or `Ellipsis` in python -- this will expand to be full range slices of all collected axes
    case ellipsis
    
    /// `None` or `newaxis` in python
    case newAxis
    case index(Int32)
    case slice(MLXSlice)
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
            return "ellipsis"
        case .newAxis:
            return "None"
        case .index(let v):
            return "\(v)"
        case .slice(let v):
            return "\(v)"
        case .array(let v):
            return "\(v.shape)(\(v.dtype))"
        }
    }
}

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
public struct MLXSlice : Equatable, CustomStringConvertible {
    private let _start: Int32?
    private let _end: Int32?
    private let _stride: Int32?
    
    /// Intialize an MLXSlice with its optional parameters.
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
    public static let full = MLXSlice()
    
    /// Given `stride` or `1`
    public var stride: Int32 { _stride ?? 1 }
    
    /// Start index using numpy conventions.
    ///
    /// If `start` is not given it will be `size - 1` if ``stride`` is negative or `0` if positive.
    public func start(_ size: Int32) -> Int32 {
        _start ??
        (stride < 0 ? size - 1 : 0)
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
        _end ??
        (stride < 0 ? -size - 1 : size)
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

extension MLXSlice : MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(self)
    }
}

public protocol MLXArrayIndex {
    var mlxArrayIndexOperation: MLXArrayIndexOperation { get }
}

extension Int : MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .index(self.int32)
    }
}

extension Range : MLXArrayIndex where Bound == Int {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32, end: self.upperBound.int32))
    }
}

extension ClosedRange : MLXArrayIndex where Bound == Int {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32, end: self.upperBound.int32 + 1))
    }
}

extension PartialRangeUpTo : MLXArrayIndex where Bound == Int {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: 0, end: self.upperBound.int32))
    }
}

extension PartialRangeThrough : MLXArrayIndex where Bound == Int {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: 0, end: self.upperBound.int32 + 1))
    }
}

extension PartialRangeFrom : MLXArrayIndex where Bound == Int {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .slice(.init(start: self.lowerBound.int32))
    }
}

extension MLXArray : MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .array(self)
    }
}

// TODO remove?
struct MLXEllipsisHelper {
    static postfix func ... (x: MLXEllipsisHelper) -> MLXArrayIndex {
        MLXEllipsisIndex()
    }
    static prefix func ... (x: MLXEllipsisHelper) -> MLXArrayIndex {
        MLXEllipsisIndex()
    }
}

public struct MLXEllipsisIndex : MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .ellipsis
    }
}

extension MLXArrayIndex where Self == MLXEllipsisIndex {
    public static var ellipsis: Self { Self() }
}

public struct MLXNewAxisIndex : MLXArrayIndex {
    public var mlxArrayIndexOperation: MLXArrayIndexOperation {
        .newAxis
    }
}

extension MLXArrayIndex where Self == MLXNewAxisIndex {
    public static var newAxis: Self { Self() }
}

@inlinable public func stride(from start: Int? = nil, to end: Int? = nil, by stride: Int? = nil) -> MLXSlice {
    MLXSlice(start: start?.int32, end: end?.int32, stride: stride?.int32)
}

extension MLXArrayIndex where Self == MLXSlice {
    public static func stride(from start: Int? = nil, to end: Int? = nil, by stride: Int? = nil) -> MLXSlice {
        MLXSlice(start: start?.int32, end: end?.int32, stride: stride?.int32)
    }
}
