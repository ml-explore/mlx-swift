import Foundation
import Cmlx

public enum MLXRandom {
    
    /// https://jax.readthedocs.io/en/latest/jep/263-prng.html
    public struct Key {
        var value: MLXArray
        fileprivate var ctx: OpaquePointer { value.ctx }
    }

    /// Seed the global PRNG.
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
    public static func uniform<T>(_ range: Range<T>, _ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
    /// The type of  `lb` must be a floating point type.
    public static func uniform(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_uniform(lb.ctx, ub.ctx, shape.asInt32, shape.count, lb.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate uniformly distributed random numbers.
    ///
    /// The values are sampled uniformly in the half-open interval `[lb, ub)`.
    /// The lower and upper bound can be scalars or arrays and must be
    /// broadcastable to the optional `shape` (it will be the shape of the `lb`
    /// if not specified).
    ///
    /// This will generate values of the given `type`.
    public static func uniform<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_uniform(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate normally distributed random numbers.
    ///
    /// Generate an array of random numbers using the optional shape.  The result
    /// will be of the given `type`.
    public static func normal<T>(_ shape: [Int] = [], type: T.Type, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        MLXArray(mlx_random_normal(shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }
    
    /// Generate normally distributed random numbers.
    ///
    /// Generate an array of `Float` random numbers using the optional shape.
    public static func normal(_ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_random_normal(shape.asInt32, shape.count, Float.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate random integers from the given interval.
    ///
    /// The values are sampled with equal probability from the integers in
    /// half-open interval ``[low, high)``. The lower and upper bound can be
    /// scalars or arrays and must be roadcastable to ``shape``.
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
    public static func randInt<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryInteger {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_randint(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate Bernoulli random values.
    ///
    /// The values are sampled from the bernoulli distribution with parameter `p` using the default value of 0.5.
    /// The result will be of size `shape`.
    public static func bernoulli(_ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let p = MLXArray(0.5)
        return MLXArray(mlx_random_bernoulli(p.ctx, shape.asInt32, shape.count, key?.ctx, stream.ctx))
    }

    /// Generate Bernoulli random values.
    ///
    /// The values are sampled from the bernoulli distribution with parameter `p`.  The result will
    /// be of size `shape`.
    public static func bernoulli<T>(_ p: T, _ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let p = MLXArray(p)
        return MLXArray(mlx_random_bernoulli(p.ctx, shape.asInt32, shape.count, key?.ctx, stream.ctx))
    }

    /// Generate Bernoulli random values.
    ///
    /// The values are sampled from the bernoulli distribution with parameter
    /// `p`. The parameter `p` must have a floating point type and
    /// must be broadcastable to `shape`.
    public static func bernoulli(_ p: MLXArray, _ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        MLXArray(mlx_random_bernoulli(p.ctx, shape.asInt32, shape.count, key?.ctx, stream.ctx))
    }
    
    /// Generate values from a truncated normal distribution.
    ///
    /// The values are sampled from the truncated normal distribution in the range.
    /// An optional shape can be used to broadcast into
    /// a larger array.  An optional `Key` can be specified to control the PRNG.
    public static func truncatedNormal<T>(_ range: Range<T>, _ shape: [Int] = [], key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
    /// The type of  `lb` must be a floating point type.
    public static func truncatedNormal(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_truncated_normal(lb.ctx, ub.ctx, shape.asInt32, shape.count, lb.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

    /// Generate values from a truncated normal distribution.
    ///
    /// The values are sampled from the truncated normal distribution
    /// on the domain `(lower, upper)`. The bounds `lower` and `upper`
    /// can be scalars or arrays and must be broadcastable to `shape`.
    public static func truncatedNormal<T>(_ lb: MLXArray, _ ub: MLXArray, _ shape: [Int]? = nil, type: T.Type, key: Key? = nil, stream: StreamOrDevice = .default) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
        let shape = shape ?? lb.shape
        return MLXArray(mlx_random_truncated_normal(lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key?.ctx, stream.ctx))
    }

}
