import Foundation
import Cmlx

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
    /// let a = MLXArray(7)
    /// ```
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(_ value: Int32) {
        self.init(mlx_array_from_int(value))
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
        self.init(withUnsafePointer(to: value) { ptr in
            mlx_array_from_data(ptr, [], 0, T.dtype.cmlxDtype)
        })
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
        self.init(value.withUnsafeBufferPointer { ptr in
            let shape = shape ?? [value.count]
            return mlx_array_from_data(ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype)
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
        self.init(floats.withUnsafeBufferPointer { ptr in
            let shape = shape ?? [floats.count]
            return mlx_array_from_data(ptr.baseAddress!, shape.asInt32, shape.count.int32, Float.dtype.cmlxDtype)
        })
    }
    
    /// Unavailable init to redirect for initializing with a `[Double]`
    @available(*, unavailable, renamed: "MLXArray(converting:shape:)", message: "Use MLXArray(converting: [1.0, 2.0, ...]) instead")
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
    /// ### See Also
    /// - <doc:initialization>
    public convenience init<S: Sequence>(_ sequence: S, _ shape: [Int]? = nil) where S.Element: HasDType {
        let value = Array(sequence)
        shapePrecondition(shape: shape, count: value.count)
        self.init(value.withUnsafeBufferPointer { ptr in
            let shape = shape ?? [value.count]
            return mlx_array_from_data(ptr.baseAddress!, shape.asInt32, shape.count.int32, S.Element.dtype.cmlxDtype)
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
        self.init(mlx_array_from_data(ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype))
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
    public convenience init<T: HasDType>(_ ptr: UnsafeRawBufferPointer, _ shape: [Int]? = nil, type: T.Type) {
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
        self.init(data.withUnsafeBytes { ptr in
            let buffer = ptr.assumingMemoryBound(to: type)
            shapePrecondition(shape: shape, count: buffer.count)
            let shape = shape ?? [buffer.count]
            return mlx_array_from_data(ptr.baseAddress!, shape.asInt32, shape.count.int32, T.dtype.cmlxDtype)
        })
    }

}

// MARK: - Expressible by literals

extension MLXArray: ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
    
    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a literal.
    ///
    /// ```swift
    /// let a: MLXArray = 7
    /// ```
    ///
    /// This is convenient for calling methods that take `MLXArray` parameters using literals.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(integerLiteral value: Int32) {
        self.init(mlx_array_from_int(value))
    }
    
    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a literal.
    ///
    /// ```swift
    /// let a: MLXArray = false
    /// ```
    ///
    /// This is convenient for calling methods that take `MLXArray` parameters using literals.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(booleanLiteral value: Bool) {
        self.init(mlx_array_from_bool(value))
    }
    
    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a literal.
    ///
    /// ```swift
    /// let a: MLXArray = 35.1
    /// ```
    ///
    /// This is convenient for calling methods that take `MLXArray` parameters using literals.
    ///
    /// ### See Also
    /// - <doc:initialization>
    public convenience init(floatLiteral value: Float) {
        self.init(mlx_array_from_float(value))
    }
    
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
            return mlx_array_from_data(ptr.baseAddress!, shape, Int32(shape.count), Int32.dtype.cmlxDtype)!
        }
        self.init(ctx)
    }
}

// MARK: - Factory Methods

extension MLXArray {
    
    /// Construct an array of zeros.
    ///
    /// Example:
    ///
    /// ```swift
    /// let z = MLXArray.zeros([5, 10], type: Int.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - type: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``zeros(like:stream:)``
    /// - ``ones(_:type:stream:)``
    static public func zeros<T: HasDType>(_ shape: [Int], type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_zeros(shape.map { Int32($0) }, shape.count, T.dtype.cmlxDtype, stream.ctx))
    }
    
    /// Construct an array of zeros.
    ///
    /// Example:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [4, 3])
    /// let z = MLXArray.zeros(like: array)
    /// ```
    ///
    /// - Parameters:
    ///     - like: array to copy shape and dtype from
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``zeros(_:type:stream:)``
    /// - ``ones(_:type:stream:)``
    static public func zeros(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_zeros_like(array.ctx, stream.ctx))
    }

    /// Construct an array of ones.
    ///
    /// Example:
    ///
    /// ```swift
    /// let r = MLXArray.ones([5, 10], type: Int.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: desired shape
    ///     - type: dtype of the values
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ones(like:stream:)``
    /// - ``zeros(_:type:stream:)``
    static public func ones<T: HasDType>(_ shape: [Int], type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_ones(shape.map { Int32($0) }, shape.count, T.dtype.cmlxDtype, stream.ctx))
    }

    /// Construct an array of ones.
    ///
    /// Example:
    ///
    /// ```swift
    /// let array = MLXArray(0 ..< 12, [4, 3])
    /// let r = MLXArray.ones(like: array)
    /// ```
    ///
    /// - Parameters:
    ///     - like: array to copy shape and dtype from
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``ones(_:type:stream:)``
    /// - ``zeros(_:type:stream:)``
    static public func ones(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_ones_like(array.ctx, stream.ctx))
    }
    
    /// Create an identity matrix or a general diagonal matrix.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.eye(10)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``identity(_:type:stream:)``
    static public func eye<T: HasDType>(_ n: Int, m: Int? = nil, k: Int = 0, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_eye(n.int32, (m ?? n).int32, k.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `vals`. If `vals`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [5, 4] array filled with 7
    /// let r = MLXArray.full([5, 4], values: 7, type: Float.self)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: shape of the output array
    ///     - values: values to be bradcast into the array
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``full(_:values:stream:)``
    /// - ``repeat(_:count:axis:stream:)``
    static public func full<T: HasDType>(_ shape: [Int], values: MLXArray, type: T.Type, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_full(shape.asInt32, shape.count, values.ctx, T.dtype.cmlxDtype, stream.ctx))
    }
    
    /// Construct an array with the given value.
    ///
    /// Constructs an array of size `shape` filled with `vals`. If `vals`
    /// is an :obj:`array` it must be <doc:broadcasting> to the given `shape`.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [5, 4] array filled with 7
    /// let r = MLXArray.full([5, 4], values: 7)
    /// ```
    ///
    /// - Parameters:
    ///     - shape: shape of the output array
    ///     - values: values to be bradcast into the array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``full(_:values:type:stream:)``
    /// - ``repeat(_:count:axis:stream:)``
    static public func full(_ shape: [Int], values: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_full(shape.asInt32, shape.count, values.ctx, values.dtype.cmlxDtype, stream.ctx))
    }

    /// Create a square identity matrix.
    ///
    /// Example:
    ///
    /// ```swift
    /// //  create [10, 10] array with 1's on the diagonal.
    /// let r = MLXArray.identity(10)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows and columns in the output
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``eye(_:m:k:type:stream:)``
    static public func identity<T: HasDType>(_ n: Int, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_identity(n.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    /// Generate `num` evenly spaced numbers over interval `[start, stop]`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create a 50 element 1-D array with values from 0 to 50
    /// let r = MLXArray.linSpace(0, 50)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - count: number of samples
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``linspace(_:_:count:stream:)-32sbl``
    static public func linspace<T: HasDType>(_ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default) -> MLXArray where T : BinaryInteger {
        MLXArray(mlx_linspace(Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    /// Generate `num` evenly spaced numbers over interval `[start, stop]`.
    ///
    /// Example:
    ///
    /// ```swift
    /// // Create a 50 element 1-D array with values from 0 to 1
    /// let r = MLXArray.linSpace(0.0, 1.0)
    /// ```
    ///
    /// - Parameters:
    ///     - start: start value
    ///     - stop: stop value
    ///     - count: number of samples
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    static public func linspace<T: HasDType>(_ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default) -> MLXArray where T : BinaryFloatingPoint {
        MLXArray(mlx_linspace(Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx))
    }

    /// Repeat an array along a specified axis.
    ///
    /// Example:
    ///
    /// ```swift
    /// // repeat a [2, 2] array 4 times along axis 1
    /// let r = MLXArray.repeat(MLXArray(0 ..< 4, [2, 2]), count: 4, axis: 1)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to repeat
    ///     - count: number of times to repeat
    ///     - axis: axis to repeat along
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeat(_:count:stream:)``
    /// - ``full(_:values:stream:)``
    static public func `repeat`(_ array: MLXArray, count: Int, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_repeat(array.ctx, count.int32, axis.int32, stream.ctx))
    }
    
    /// Repeat a flattened array along axis 0.
    ///
    /// Example:
    ///
    /// ```swift
    /// // repeat a 4 element array 4 times along axis 0
    /// let r = MLXArray.repeat(MLXArray(0 ..< 4, [2, 2]), count: 4)
    /// ```
    ///
    /// - Parameters:
    ///     - array: array to repeat
    ///     - count: number of times to repeat
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``repeat(_:count:axis:stream:)``
    /// - ``full(_:values:stream:)``
    static public func `repeat`(_ array: MLXArray, count: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_repeat_all(array.ctx, count.int32, stream.ctx))
    }

    /// An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// Example:
    ///
    /// ```swift
    /// // [5, 5] array with the lower triangle filled with 1s
    /// let r = MLXArray.triangle(5)
    /// ```
    ///
    /// - Parameters:
    ///     - n: number of rows in the output
    ///     - m: number of columns in the output -- equal to `n` if not specified
    ///     - k: index of the diagonal
    ///     - type: data type of the output array
    ///     - stream: stream or device to evaluate on
    ///
    /// ### See Also
    /// - <doc:initialization>
    static public func triangle<T: HasDType>(_ n: Int, m: Int? = nil, k: Int = 0, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_tri(n.int32, (m ?? n).int32, k.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
}

