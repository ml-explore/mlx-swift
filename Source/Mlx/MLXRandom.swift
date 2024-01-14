import Foundation
import Cmlx

/// Collection of functions related to random number generation.
///
///
public enum MLXRandom {
    
    /// Container for PRNG keys.
    ///
    /// A `Key` can be passed to any of the random generators to provider repeatable generation.
    ///
    /// ### See Also
    /// - [JAX Documentation](https://jax.readthedocs.io/en/latest/jep/263-prng.html)
    public struct Key {
        var value: MLXArray
        fileprivate var ctx: OpaquePointer { value.ctx }
        
        /// Split a PRNG key into sub keys.
        ///
        /// Convenience for `MLXRandom.split()`
        public func split(into num: Int = 2, stream: StreamOrDevice = .default) -> [Key] {
            MLXRandom.split(key: self, into: num, stream: stream)
        }
    }

    /// Seed the global PRNG.
    ///
    /// See also ``key(_:)``
    public static func seed(_ seed: UInt64) {
        mlx_random_seed(seed)
    }
    
    /// Get a PRNG key from a seed.
    ///
    /// Return a value that can be used as a PRNG key.
    public static func key(_ seed: UInt64) -> Key {
        Key(value: MLXArray(mlx_random_key(seed)))
    }

    /// Split a PRNG key into sub keys.
    ///
    /// Array of `MLXRandom.Key` values.
    public static func split(key: Key, into num: Int = 2, stream: StreamOrDevice = .default) -> [Key] {
        let keys = MLXArray(mlx_random_split_equal_parts(key.value.ctx, num.int32, stream.ctx))
        return keys.map { Key(value: $0) }
    }
    
    /// Generate uniformly distributed random numbers.
    ///
    /// The values are sampled uniformly in the range.  An optional shape can be used to broadcast into
    /// a larger array.  An optional `Key` can be specified to control the PRNG.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // create an array of shape [50] type Float values in the range 0 ..< 10
    /// let array = MLXRandom.uniform(0 ..< 10, [50], key: key)
    ///
    /// // same, but in range 0.5 ..< 1
    /// let array = MLXRandom.uniform(0.5 ..< 1, [50], key: key)
    /// ```
    public static func uniform<R: HasDType, T>(_ range: Range<R>, _ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        return MLXArray(mlx_random_uniform(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }
    
    /// Generate uniformly distributed random numbers.
    ///
    /// Specialization to make it easy to call with `Float`:
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    /// let array = MLXRandom.uniform(0.5 ..< 1, [50], key: key)
    /// ```
    public static func uniform<T>(_ range: Range<Float>, _ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        return MLXArray(mlx_random_uniform(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }
    
    /// Generate uniformly distributed random numbers.
    ///
    /// The values are sampled uniformly in the half-open interval `[lb, ub)`.
    /// The lower and upper bound can be scalars or arrays and must be
    /// broadcastable to the optional `shape` (it will be the shape of the `lb`
    /// if not specified).
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate an array of two Float values, one in the range 0 ..< 10
    /// // and one in the range 10 ..< 100
    /// let value = MLXRandom.uniform([0, 10], [10, 100], key: key)
    /// ```
    public static func uniform<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_uniform(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate normally distributed random numbers.
    ///
    /// Generate an array of random numbers using the optional shape.  The result
    /// will be of the given `type`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate a single Float with normal distribution
    /// let value = MLXRandom.normal(key: key).item(Float.self)
    ///
    /// // generate an array of Float with normal distribution in shape [10, 5]
    /// let array = MLXRandom.normal([10, 5], key: key)
    /// ```
    public static func normal<T>(_ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        MLXArray(mlx_random_normal(shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }
    
    /// Generate random integers from the given interval.
    ///
    /// The values are sampled with equal probability from the integers in
    /// half-open interval `[low, high)`. The lower and upper bound can be
    /// scalars or arrays and must be roadcastable to `shape`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate a single random int
    /// let value = MLXRandom.randInt(0 ..< 100, key: key).item(Int.self)
    ///
    /// // generate an array of shape [50] random Int32
    /// let array = MLXRandom.randInt(Int32(0) ..< 100, [50], key: key)
    /// ```
    public static func randInt<T>(_ range: Range<T>, _ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryInteger {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        return MLXArray(mlx_random_randint(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate random integers from the given interval.
    ///
    /// The values are sampled with equal probability from the integers in
    /// half-open interval `[lb, ub)`. The lower and upper bound can be
    /// scalars or arrays and must be roadcastable to `shape`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate an array of Int values, one in the range 0 ..< 10
    /// // and one in the range 10 ..< 100
    /// let array = MLXRandom.randInt([0, 10], [10, 100], key: key)
    /// ```
    public static func randInt(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_randint(lb.ctx, ub.ctx, shape.asInt32, shape.count, lb.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate random integers from the given interval.
    ///
    /// The values are sampled with equal probability from the integers in
    /// half-open interval `[lb, ub)`. The lower and upper bound can be
    /// scalars or arrays and must be roadcastable to `shape`.  The
    /// integer results will be of the given `type`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate an array of Int8 values, one in the range 0 ..< 10
    /// // and one in the range 10 ..< 100
    /// let array = MLXRandom.randInt([0, 10], [10, 100], type: Int8.self, key: key)
    /// ```
    public static func randInt<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryInteger {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_randint(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate Bernoulli random values.
    ///
    /// The values are sampled from the bernoulli distribution with parameter `p` using the default value of 0.5.
    /// The result will be of size `shape`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate a single random Bool with p = 0.5
    /// let value = MLXRandom.bernoulli(key: key).item(Bool.self)
    ///
    /// // generate an array of shape [50, 2] of random Bool
    /// let array = MLXRandom.bernoulli([50, 2], key: key)
    /// ```
    public static func bernoulli(_ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let p = MLXArray(0.5)
        return MLXArray(mlx_random_bernoulli(p.ctx, shape.asInt32, shape.count, key?.ctx, stream.ctx))
    }

    /// Generate Bernoulli random values.
    ///
    /// The values are sampled from the bernoulli distribution with parameter
    /// `p`. The parameter `p` must have a floating point type and
    /// must be broadcastable to `shape`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate a single random Bool with p = 0.8
    /// let value = MLXRandom.bernoulli(0.8, key: key).item(Bool.self)
    ///
    /// // generate an array of shape [50, 2] of random Bool with p = 0.8
    /// let array = MLXRandom.bernoulli(0.8, [50, 2], key: key)
    ///
    /// // generate an array of [3] Bool with the given p values
    /// let array = MLXRandom.bernoulli(MLXArray(convert: [0.1, 0.5, 0.8]), key: key)
    /// ```
    public static func bernoulli(_ p: MLXArray, _ shape: [Int]? = nil, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let shape = shape ?? p.shape
        return MLXArray(mlx_random_bernoulli(p.ctx, shape.asInt32, shape.count, key?.ctx, stream.ctx))
    }
    
    /// Generate values from a truncated normal distribution.
    ///
    /// The values are sampled from the truncated normal distribution in the range.
    /// An optional shape can be used to broadcast into
    /// a larger array.  An optional `Key` can be specified to control the PRNG.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // create an array of shape [50] type Float values in the range 0 ..< 10
    /// let array = MLXRandom.truncatedNormal(0 ..< 10, [50], key: key)
    ///
    /// // same, but in range 0.5 ..< 1
    /// let array = MLXRandom.truncatedNormal(0.5 ..< 1, [50], key: key)
    /// ```
    ///
    /// ### See also
    /// - [JAX Documentation](https://jax.readthedocs.io/en/latest/_modules/jax/_src/random.html#truncated_normal)
    public static func truncatedNormal<R: HasDType, T>(_ range: Range<R>, _ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        return MLXArray(mlx_random_truncated_normal(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }
    
    /// Generate values from a truncated normal distribution.
    ///
    /// Specialization to make it easy to call with `Float`:
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    /// let array = MLXRandom.truncatedNormal(0.5 ..< 1, [50], key: key)
    /// ```
    public static func truncatedNormal<T>(_ range: Range<Float>, _ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        return MLXArray(mlx_random_truncated_normal(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate values from a truncated normal distribution.
    ///
    /// The values are sampled from the truncated normal distribution
    /// on the domain `(lower, upper)`. The bounds `lower` and `upper`
    /// can be scalars or arrays and must be broadcastable to `shape`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate an array of two Float values, one in the range 0 ..< 10
    /// // and one in the range 10 ..< 100
    /// let value = MLXRandom.truncatedNormal([0, 10], [10, 100], key: key)
    /// ```
    public static func truncatedNormal<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_truncated_normal(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Sample from the standard Gumbel distribution.
    ///
    /// The values are sampled from a standard Gumbel distribution
    /// which CDF `exp(-exp(-x))`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// // generate a single Float with Gumbel distribution
    /// let value = MLXRandom.gumbel(key: key).item(Float.self)
    ///
    /// // generate an array of Float with Gumbel distribution in shape [10, 5]
    /// let array = MLXRandom.gumbel([10, 5], key: key)
    /// ```
    public static func gumbel<T>(_ shape: [Int] = [], type: T.Type = Float.self, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        MLXArray(mlx_random_gumbel(shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Sample from a categorical distribution.
    ///
    /// The values are sampled from the categorical distribution specified by
    /// the unnormalized values in `logits`.   If the `shape` is not specified
    /// the result shape will be the same shape as `logits` with the `axis`
    /// dimension removed.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// let logits = MLXArray.zeros([5, 20])
    ///
    /// // produces MLXArray of UInt32 shape [5]
    /// let result = MLXRandom.categorical(logits, key: key)
    /// ```
    ///
    /// - Parameters:
    ///     - logits: The *unnormalized* categorical distribution(s).
    public static func categorical(_ logits: MLXArray, axis: Int = -1, shape: [Int]? = nil, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        if let shape {
            return MLXArray(mlx_random_categorical_shape(logits.ctx, axis.int32, shape.asInt32, shape.count, key?.ctx, stream.ctx))
        } else {
            return MLXArray(mlx_random_categorical(logits.ctx, axis.int32, key?.ctx, stream.ctx))
        }
    }

    /// Sample from a categorical distribution.
    ///
    /// The values are sampled from the categorical distribution specified by
    /// the unnormalized values in `logits`.
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    ///
    /// let logits = MLXArray.zeros([5, 20])
    ///
    /// // produces MLXArray of UInt32 shape [5, 2]
    /// let result = MLXRandom.categorical(logits, count: 2, key: key)
    /// ```
    ///
    /// - Parameters:
    ///     - logits: The *unnormalized* categorical distribution(s).
    public static func categorical(_ logits: MLXArray, axis: Int = -1, count: Int, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_random_categorical_num_samples(logits.ctx, axis.int32, count.int32, key?.ctx, stream.ctx))
    }

}
