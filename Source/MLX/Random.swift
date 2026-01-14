// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Collection of functions related to random number generation.
///
/// Following [JAX’s PRNG design](https://jax.readthedocs.io/en/latest/jep/263-prng.html)
/// we use a splittable version of Threefry, which is a counter-based PRNG.
///
/// Random sampling functions in MLX use an implicit global PRNG state by default.
/// However, all functions take an optional key keyword argument for when more fine-grained
/// control or explicit state management is needed.  Callers can also arrange for `Task` local
/// random state -- useful in multithreaded situations.
///
/// For example, you can generate random numbers with:
///
/// ```swift
/// for _ in 0 ..< 3 {
///   print(MLXRandom.uniform())
/// }
/// ```
///
/// which will print a sequence of unique pseudo random numbers. Alternatively you can explicitly set the key:
///
/// ```swift
/// let key = MLXRandom.key(0)
/// for _ in 0 ..< 3 {
///   print(MLXRandom.uniform(key: key))
/// }
/// ```
///
/// which will yield the same pseudo random number at each iteration as the key doesn't change.
///
/// To get a new random number for each call you would ``split(key:stream:)`` the key:
///
/// ```swift
/// var key = MLXRandom.key(0)
/// for _ in 0 ..< 3 {
///   let (a, b) = MLXRandom.split(key: key)
///
///   // use b to generate a different value each time
///   print(MLXRandom.uniform(key: b))
///
///   // new random state is a
///   key = a
/// }
/// ```
///
/// This will generate the same sequence of numbers each time (same starting key) but
/// different values for each call.
///
/// As a convenience you can use ``RandomState`` to manage the key splitting and:
///
/// ```swift
/// let state = RandomState(seed: 0)
/// for _ in 0 ..< 3 {
///   print(MLXRandom.uniform(key: state))
/// }
/// ```
///
/// Finally, if you need to control random state in deeply nested calls to `MLXRandom` or you need
/// thread-safe random state for multi-threaded evaluation you can use ``withRandomState(_:body:)-18ob4``.
///
/// ```swift
/// await withTaskGroup { group in
///     for i in 0 ..< 10 {
///         group.addTask {
///             let state = MLXRandom.RandomState(seed: UInt64(i))
///             return withRandomState(state) {
///                 var t: Float = 0.0
///                 for _ in 0 ..< 100 {
///                     t += uniform(0 ..< 1, [10, 10]).sum().item(Float.self)
///                 }
///                 return t
///             }
///         }
///     }
///
///     for await v in group {
///         ...
///     }
/// }
/// ```
///
/// Each task will have separate ``RandomState`` that will be used implicitly (if no other key is passed in).
public enum MLXRandom {

    /// Seed the global PRNG.
    ///
    /// ### See Also
    /// - ``key(_:)``
    /// - ``RandomState``
    /// - ``globalState``
    public static func seed(_ seed: UInt64) {
        globalState.seed(seed)
    }

    /// Get a PRNG key from a seed.
    ///
    /// Return a value that can be used as a PRNG key.  All ``MLXRandom``
    /// functions take an optional key -- this will let you control the
    /// random number generation.
    public static func key(_ seed: UInt64) -> MLXArray {
        var result = mlx_array_new()
        mlx_random_key(&result, seed)
        return MLXArray(result)
    }

    /// Split a PRNG key into sub keys.
    ///
    /// ### See Also
    /// - ``split(key:stream:)``
    public static func split(key: MLXArray, into num: Int, stream: StreamOrDevice = .default)
        -> [MLXArray]
    {
        var keys = mlx_array_new()
        mlx_random_split_num(&keys, key.ctx, num.int32, stream.ctx)

        return MLXArray(keys).map { $0 }
    }

    /// Split a PRNG key into two keys and return a tuple.
    ///
    /// ### See Also
    /// - ``split(key:into:stream:)``
    public static func split(key: MLXArray, stream: StreamOrDevice = .default) -> (
        MLXArray, MLXArray
    ) {
        var r0 = mlx_array_new()
        var r1 = mlx_array_new()
        mlx_random_split(&r0, &r1, key.ctx, stream.ctx)
        return (MLXArray(r0), MLXArray(r1))
    }

    /// Generate uniformly distributed random numbers with a `RangeExpression`.
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
    public static func uniform(
        _ range: Range<some HasDType>, _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_uniform(
            &result, lb.ctx, ub.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate uniformly distributed random numbers with a `RangeExpression<Float>` (specialization).
    ///
    /// Specialization to make it easy to call with `Float`:
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    /// let array = MLXRandom.uniform(0.5 ..< 1, [50], key: key)
    /// ```
    public static func uniform(
        _ range: Range<Float> = 0 ..< 1, _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_uniform(
            &result, lb.ctx, ub.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate uniformly distributed random numbers between `low` and `high`.
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
    /// let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)
    /// ```
    public static func uniform(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none,
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_uniform(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate uniformly distributed random numbers between `low` and `high` with a given `DType`.
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
    /// let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)
    /// ```
    public static func uniform(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_uniform(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, dtype.cmlxDtype, key.ctx,
            stream.ctx
        )

        return MLXArray(result)
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
    ///
    /// - Parameters:
    ///   - shape: shape of the output
    ///   - type: type of the output
    ///   - loc: mean of the distribution
    ///   - scale: standard deviation of the distribution
    ///   - key: PRNG key
    ///   - stream: stream or device to evaluate on
    public static func normal(
        _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self, loc: Float = 0,
        scale: Float = 1, key: (some RandomStateOrKey)? = MLXArray?.none,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_normal(
            &result, shape.asInt32, shape.count, type.dtype.cmlxDtype, loc, scale, key.ctx,
            stream.ctx)

        return MLXArray(result)
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
    ///
    /// - Parameters:
    ///   - shape: shape of the output
    ///   - dtype: type of the output
    ///   - loc: mean of the distribution
    ///   - scale: standard deviation of the distribution
    ///   - key: PRNG key
    ///   - stream: stream or device to evaluate on
    public static func normal(
        _ shape: some Collection<Int> = [], dtype: DType, loc: Float = 0, scale: Float = 1,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_normal(
            &result, shape.asInt32, shape.count, dtype.cmlxDtype, loc, scale, key.ctx, stream.ctx)

        return MLXArray(result)
    }

    /// Generate jointly-normal random samples given a mean and covariance.
    ///
    /// The matrix `covariance` must be positive semi-definite. The behavior is
    /// undefined if it is not.  The only supported `dtype` is `.float32`.
    ///
    /// - Parameters:
    ///   - mean: array of shape `[..., n]`, the mean of the distribution.
    ///   - covariance: array  of shape `[..., n, n]`, the covariance
    /// matrix of the distribution. The batch shape `...` must be
    /// broadcast-compatible with that of `mean`.
    ///   - shape:  The output shape must be
    /// broadcast-compatible with `mean.shape.dropLast()` and `covariance.shape.dropLast(2)`.
    /// If empty, the result shape is determined by broadcasting the batch
    /// shapes of `mean` and `covariance`.
    ///   - dtype: DType of the result
    ///   - key: PRNG key
    ///   - stream: stream or device to evaluate on
    public static func multivariateNormal(
        mean: MLXArray, covariance: MLXArray, shape: some Collection<Int> = [], dtype: DType,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_multivariate_normal(
            &result, mean.ctx, covariance.ctx, shape.asInt32, shape.count,
            dtype.cmlxDtype, key.ctx, stream.ctx)

        return MLXArray(result)
    }

    /// Generate random integers from the given interval using a `RangeExpression<Int>`.
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
    public static func randInt<T>(
        _ range: Range<T>, _ shape: some Collection<Int> = [],
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray where T: HasDType, T: BinaryInteger {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_randint(
            &result, lb.ctx, ub.ctx, shape.asInt32, shape.count, T.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate random integers from the given interval (`low:` and `high:`).
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
    /// let array = MLXRandom.randInt(low: [0, 10], high: [10, 100], key: key)
    /// ```
    public static func randInt(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_randint(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, low.dtype.cmlxDtype, key.ctx,
            stream.ctx
        )
        return MLXArray(result)
    }

    /// Generate random integers from the given interval (`low:` and `high:`) with a given type, e.g. `Int8.self`.
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
    /// let array = MLXRandom.randInt(low: [0, 10], high: [10, 100], type: Int8.self, key: key)
    /// ```
    public static func randInt(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none, type: (some HasDType & BinaryInteger).Type,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_randint(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate Bernoulli random values with a `p` value of 0.5.
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
    public static func bernoulli(
        _ shape: some Collection<Int> = [], key: (some RandomStateOrKey)? = MLXArray?.none,
        stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        let p = MLXArray(0.5)
        let key = resolve(key: key)
        var result = mlx_array_new()
        mlx_random_bernoulli(&result, p.ctx, shape.asInt32, shape.count, key.ctx, stream.ctx)

        return MLXArray(result)
    }

    /// Generate Bernoulli random values with a given `p` value.
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
    public static func bernoulli(
        _ p: some ScalarOrArray, _ shape: (some Collection<Int>)? = [Int]?.none,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let p = p.asMLXArray(dtype: .float32)
        let shape = shape.map { Array($0) } ?? p.shape
        let key = resolve(key: key)
        var result = mlx_array_new()
        mlx_random_bernoulli(&result, p.ctx, shape.asInt32, shape.count, key.ctx, stream.ctx)

        return MLXArray(result)
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
    public static func truncatedNormal(
        _ range: Range<some HasDType>, _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_truncated_normal(
            &result, lb.ctx, ub.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate values from a truncated normal distribution in a given `RangeExpression<Float>`.
    ///
    /// Specialization to make it easy to call with `Float`:
    ///
    /// ```swift
    /// let key = MLXRandom.key(0)
    /// let array = MLXRandom.truncatedNormal(0.5 ..< 1, [50], key: key)
    /// ```
    public static func truncatedNormal(
        _ range: Range<Float>, _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let lb = MLXArray(range.lowerBound)
        let ub = MLXArray(range.upperBound)
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_truncated_normal(
            &result, lb.ctx, ub.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate values from a truncated normal distribution between `low` and `high`.
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
    public static func truncatedNormal(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none,
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_truncated_normal(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx,
            stream.ctx)

        return MLXArray(result)
    }

    /// Generate values from a truncated normal distribution between `low` and `high` with a given `DType`.
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
    public static func truncatedNormal(
        low: some ScalarOrArray, high: some ScalarOrArray,
        _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let (low, high) = toArrays(low, high)
        let shape = shape.map { Array($0) } ?? low.shape
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_truncated_normal(
            &result, low.ctx, high.ctx, shape.asInt32, shape.count, dtype.cmlxDtype, key.ctx,
            stream.ctx
        )

        return MLXArray(result)
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
    public static func gumbel(
        _ shape: some Collection<Int> = [],
        type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_gumbel(
            &result, shape.asInt32, shape.count, type.dtype.cmlxDtype, key.ctx, stream.ctx)

        return MLXArray(result)
    }

    /// Sample from the standard Gumbel distribution with a given `DType`.
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
    public static func gumbel(
        _ shape: some Collection<Int> = [], dtype: DType,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_gumbel(&result, shape.asInt32, shape.count, dtype.cmlxDtype, key.ctx, stream.ctx)

        return MLXArray(result)
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
    ///   - logits: The *unnormalized* categorical distribution(s).
    ///   - axis: axis that specifies the distribution
    ///   - shape: optional shape of the output -- this must be broadcast compatible with the shape of logits
    ///   - key: optional PRNG key
    ///   - stream: stream or device to evaluate on
    public static func categorical(
        _ logits: MLXArray, axis: Int = -1, shape: (some Collection<Int>)? = [Int]?.none,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        if let shape {
            var result = mlx_array_new()

            mlx_random_categorical_shape(
                &result, logits.ctx, axis.int32, shape.asInt32, shape.count, key.ctx, stream.ctx)

            return MLXArray(result)
        } else {
            var result = mlx_array_new()
            mlx_random_categorical(&result, logits.ctx, axis.int32, key.ctx, stream.ctx)

            return MLXArray(result)
        }
    }

    /// Sample `count` values from a categorical distribution.
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
    ///   - logits: The *unnormalized* categorical distribution(s).
    ///   - axis: axis that specifies the distribution
    ///   - count: number of samples to draw from logits
    ///   - key: optional PRNG key
    ///   - stream: stream or device to evaluate on
    public static func categorical(
        _ logits: MLXArray, axis: Int = -1, count: Int,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_categorical_num_samples(
            &result, logits.ctx, axis.int32, count.int32, key.ctx, stream.ctx)

        return MLXArray(result)
    }

    /// Sample numbers from a Laplace distribution.
    ///
    /// - Parameters:
    ///   - shape: shape of the output
    ///   - dtype: type of the output
    ///   - loc: mean of the distribution
    ///   - scale: scale "b" of the distribution
    public static func laplace(
        _ shape: some Collection<Int> = [], dtype: DType, loc: Float = 0, scale: Float = 1,
        key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let key = resolve(key: key)
        var result = mlx_array_new()

        mlx_random_laplace(
            &result, shape.asInt32, shape.count, dtype.cmlxDtype, loc, scale, key.ctx, stream.ctx)

        return MLXArray(result)
    }

}  // MLXRandom

/// Seed the global PRNG.
///
/// ### See Also
/// - ``key(_:)``
/// - ``MLXRandom/RandomState``
/// - ``MLXRandom/globalState``
public func seed(_ seed: UInt64) {
    return MLXRandom.seed(seed)
}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``MLXRandom``
/// functions take an optional key -- this will let you control the
/// random number generation.
public func key(_ seed: UInt64) -> MLXArray {
    return MLXRandom.key(seed)
}

/// Split a PRNG key into sub keys.
///
/// ### See Also
/// - ``split(key:stream:)``
public func split(key: MLXArray, into num: Int, stream: StreamOrDevice = .default) -> [MLXArray] {
    return MLXRandom.split(key: key, into: num, stream: stream)
}

/// Split a PRNG key into two keys and return a tuple.
///
/// ### See Also
/// - ``split(key:into:stream:)``
public func split(key: MLXArray, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray) {
    return MLXRandom.split(key: key, stream: stream)
}

/// Generate uniformly distributed random numbers with a `RangeExpression`.
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
public func uniform(
    _ range: Range<some HasDType>, _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.uniform(range, shape, type: type, key: key, stream: stream)
}

/// Generate uniformly distributed random numbers with a `RangeExpression<Float>` (specialization).
///
/// Specialization to make it easy to call with `Float`:
///
/// ```swift
/// let key = MLXRandom.key(0)
/// let array = MLXRandom.uniform(0.5 ..< 1, [50], key: key)
/// ```
public func uniform(
    _ range: Range<Float> = 0 ..< 1, _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.uniform(range, shape, type: type, key: key, stream: stream)
}

/// Generate uniformly distributed random numbers between `low` and `high`.
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
/// let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)
/// ```
public func uniform(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none,
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.uniform(low: low, high: high, shape, type: type, key: key, stream: stream)
}

/// Generate uniformly distributed random numbers between `low` and `high` with a given `DType`.
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
/// let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)
/// ```
public func uniform(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.uniform(low: low, high: high, shape, dtype: dtype, key: key, stream: stream)
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
///
/// - Parameters:
///   - shape: shape of the output
///   - type: type of the output
///   - loc: mean of the distribution
///   - scale: standard deviation of the distribution
///   - key: PRNG key
///   - stream: stream or device to evaluate on
public func normal(
    _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self, loc: Float = 0, scale: Float = 1,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.normal(shape, type: type, loc: loc, scale: scale, key: key, stream: stream)
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
///
/// - Parameters:
///   - shape: shape of the output
///   - dtype: type of the output
///   - loc: mean of the distribution
///   - scale: standard deviation of the distribution
///   - key: PRNG key
///   - stream: stream or device to evaluate on
public func normal(
    _ shape: some Collection<Int> = [], dtype: DType, loc: Float = 0, scale: Float = 1,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.normal(shape, dtype: dtype, loc: loc, scale: scale, key: key, stream: stream)
}

/// Generate jointly-normal random samples given a mean and covariance.
///
/// The matrix `covariance` must be positive semi-definite. The behavior is
/// undefined if it is not.  The only supported `dtype` is `.float32`.
///
/// - Parameters:
///   - mean: array of shape `[..., n]`, the mean of the distribution.
///   - covariance: array  of shape `[..., n, n]`, the covariance
/// matrix of the distribution. The batch shape `...` must be
/// broadcast-compatible with that of `mean`.
///   - shape:  The output shape must be
/// broadcast-compatible with `mean.shape.dropLast()` and `covariance.shape.dropLast(2)`.
/// If empty, the result shape is determined by broadcasting the batch
/// shapes of `mean` and `covariance`.
///   - dtype: DType of the result
///   - key: PRNG key
///   - stream: stream or device to evaluate on
public func multivariateNormal(
    mean: MLXArray, covariance: MLXArray, shape: some Collection<Int> = [], dtype: DType,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.multivariateNormal(
        mean: mean, covariance: covariance, shape: shape, dtype: dtype, key: key, stream: stream)
}

/// Generate random integers from the given interval using a `RangeExpression<Int>`.
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
public func randInt(
    _ range: Range<some HasDType & BinaryInteger>, _ shape: some Collection<Int> = [],
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.randInt(range, shape, key: key, stream: stream)
}

/// Generate random integers from the given interval (`low:` and `high:`).
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
/// let array = MLXRandom.randInt(low: [0, 10], high: [10, 100], key: key)
/// ```
public func randInt(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none, key: (some RandomStateOrKey)? = MLXArray?.none,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.randInt(low: low, high: high, shape, key: key, stream: stream)
}

/// Generate random integers from the given interval (`low:` and `high:`) with a given type, e.g. `Int8.self`.
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
/// let array = MLXRandom.randInt(low: [0, 10], high: [10, 100], type: Int8.self, key: key)
/// ```
public func randInt(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none, type: (some HasDType & BinaryInteger).Type,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.randInt(low: low, high: high, shape, type: type, key: key, stream: stream)
}

/// Generate Bernoulli random values with a `p` value of 0.5.
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
public func bernoulli(
    _ shape: some Collection<Int> = [], key: (some RandomStateOrKey)? = MLXArray?.none,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.bernoulli(shape, key: key, stream: stream)
}

/// Generate Bernoulli random values with a given `p` value.
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
public func bernoulli(
    _ p: some ScalarOrArray, _ shape: (some Collection<Int>)? = [Int]?.none,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.bernoulli(p, shape, key: key, stream: stream)
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
public func truncatedNormal(
    _ range: Range<some HasDType>, _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.truncatedNormal(range, shape, type: type, key: key, stream: stream)
}

/// Generate values from a truncated normal distribution in a given `RangeExpression<Float>`.
///
/// Specialization to make it easy to call with `Float`:
///
/// ```swift
/// let key = MLXRandom.key(0)
/// let array = MLXRandom.truncatedNormal(0.5 ..< 1, [50], key: key)
/// ```
public func truncatedNormal(
    _ range: Range<Float>, _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.truncatedNormal(range, shape, type: type, key: key, stream: stream)
}

/// Generate values from a truncated normal distribution between `low` and `high`.
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
public func truncatedNormal(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none,
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.truncatedNormal(
        low: low, high: high, shape, type: type, key: key, stream: stream)
}

/// Generate values from a truncated normal distribution between `low` and `high` with a given `DType`.
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
public func truncatedNormal(
    low: some ScalarOrArray, high: some ScalarOrArray,
    _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.truncatedNormal(
        low: low, high: high, shape, dtype: dtype, key: key, stream: stream)
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
public func gumbel(
    _ shape: some Collection<Int> = [],
    type: (some HasDType & BinaryFloatingPoint).Type = Float.self,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.gumbel(shape, type: type, key: key, stream: stream)
}

/// Sample from the standard Gumbel distribution with a given `DType`.
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
public func gumbel(
    _ shape: some Collection<Int> = [], dtype: DType,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.gumbel(shape, dtype: dtype, key: key, stream: stream)
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
///   - logits: The *unnormalized* categorical distribution(s).
///   - axis: axis that specifies the distribution
///   - shape: optional shape of the output -- this must be broadcast compatible with the shape of logits
///   - key: optional PRNG key
///   - stream: stream or device to evaluate on
public func categorical(
    _ logits: MLXArray, axis: Int = -1, shape: (some Collection<Int>)? = [Int]?.none,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.categorical(logits, axis: axis, shape: shape, key: key, stream: stream)
}

/// Sample `count` values from a categorical distribution.
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
///   - logits: The *unnormalized* categorical distribution(s).
///   - axis: axis that specifies the distribution
///   - count: number of samples to draw from logits
///   - key: optional PRNG key
///   - stream: stream or device to evaluate on
public func categorical(
    _ logits: MLXArray, axis: Int = -1, count: Int, key: (some RandomStateOrKey)? = MLXArray?.none,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.categorical(logits, axis: axis, count: count, key: key, stream: stream)
}

/// Sample numbers from a Laplace distribution.
///
/// - Parameters:
///   - shape: shape of the output
///   - dtype: type of the output
///   - loc: mean of the distribution
///   - scale: scale "b" of the distribution
///   - key: optional PRNG key
///   - stream: stream or device to evaluate on
public func laplace(
    _ shape: some Collection<Int> = [], dtype: DType, loc: Float = 0, scale: Float = 1,
    key: (some RandomStateOrKey)? = MLXArray?.none, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.laplace(shape, dtype: dtype, loc: loc, scale: scale, key: key, stream: stream)
}
