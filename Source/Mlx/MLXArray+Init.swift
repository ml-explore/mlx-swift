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
    public convenience init(_ value: Int32) {
        self.init(mlx_array_from_int(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `Bool`.
    ///
    /// ```swift
    /// let a = MLXArray(true)
    /// ```
    public convenience init(_ value: Bool) {
        self.init(mlx_array_from_bool(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `Float`.
    ///
    /// ```swift
    /// let a = MLXArray(35.1)
    /// ```
    public convenience init(_ value: Float) {
        self.init(mlx_array_from_float(value))
    }

    /// Initalizer allowing creation of scalar (0-dimension) `MLXArray` from a `HasDType` value.
    ///
    /// ```swift
    /// let a = MLXArray(UInt64(7))
    /// ```
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
    
    static public func zeros<T: HasDType>(_ shape: [Int], type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_zeros(shape.map { Int32($0) }, shape.count, T.dtype.cmlxDtype, stream.ctx))
    }
    
    static public func zeros(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_zeros_like(array.ctx, stream.ctx))
    }

    static public func ones<T: HasDType>(_ shape: [Int], type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_ones(shape.map { Int32($0) }, shape.count, T.dtype.cmlxDtype, stream.ctx))
    }

    static public func ones(like array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_ones_like(array.ctx, stream.ctx))
    }
    
    static public func eye<T: HasDType>(_ n: Int, m: Int? = nil, k: Int = 0, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_eye(n.int32, (m ?? n).int32, k.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    static public func full<T: HasDType>(_ shape: [Int], values: MLXArray, type: T.Type, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_full(shape.asInt32, shape.count, values.ctx, T.dtype.cmlxDtype, stream.ctx))
    }
    
    static public func full(_ shape: [Int], values: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_full(shape.asInt32, shape.count, values.ctx, values.dtype.cmlxDtype, stream.ctx))
    }

    static public func identity<T: HasDType>(_ n: Int, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_identity(n.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    static public func linspace<T: HasDType>(_ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default) -> MLXArray where T : BinaryInteger {
        MLXArray(mlx_linspace(Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
    static public func linspace<T: HasDType>(_ start: T, _ stop: T, count: Int = 50, stream: StreamOrDevice = .default) -> MLXArray where T : BinaryFloatingPoint {
        MLXArray(mlx_linspace(Double(start), Double(stop), count.int32, T.dtype.cmlxDtype, stream.ctx))
    }

    static public func `repeat`(_ array: MLXArray, repeats: Int, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_repeat(array.ctx, repeats.int32, axis.int32, stream.ctx))
    }
    
    static public func `repeat`(_ array: MLXArray, repeats: Int, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_repeat_all(array.ctx, repeats.int32, stream.ctx))
    }

    static public func triangle<T: HasDType>(_ n: Int, m: Int? = nil, k: Int = 0, type: T.Type = Float.self, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_tri(n.int32, (m ?? n).int32, k.int32, T.dtype.cmlxDtype, stream.ctx))
    }
    
}

// MARK: - Factory Methods (Internal)

extension MLXArray {
    
    /// Return an array of MLXArray from an `mlx_vector_array`.  The caller retains ownership
    /// of the `mlx_vector_array` and its contents.
    static func fromVector(_ vec: mlx_vector_array) -> [MLXArray] {
        (0 ..< vec.size)
            .map { index in
                // transfer ownership to the result arrays
                let ctx = vec.arrays[index]!
                mlx_retain(ctx)
                return MLXArray(ctx)
            }
    }
    
}
