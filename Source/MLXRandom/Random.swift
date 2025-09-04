// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX

@available(
    *, deprecated,
    message: "`import MLXRandom` is deprecated. All methods are now available through `import MLX"
)
public let deprecationWarning: Void = ()

/// Seed the global PRNG.
///
/// ### See Also
/// - ``key(_:)``
@available(*, deprecated, message: "seed is now available in the main MLX module")
@_disfavoredOverload
public func seed(_ seed: UInt64) {
    return MLXRandom.seed(seed)
}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``MLXRandom``
/// functions take an optional key -- this will let you control the
/// random number generation.
@available(*, deprecated, message: "key is now available in the main MLX module")
@_disfavoredOverload
public func key(_ seed: UInt64) -> MLXArray {
    return MLXRandom.key(seed)
}

/// Split a PRNG key into sub keys.
///
/// ### See Also
/// - ``split(key:stream:)``
@available(*, deprecated, message: "split is now available in the main MLX module")
@_disfavoredOverload
public func split(key: MLXArray, into num: Int, stream: StreamOrDevice = .default) -> [MLXArray] {
    return MLXRandom.split(key: key, into: num, stream: stream)
}

/// Split a PRNG key into two keys and return a tuple.
///
/// ### See Also
/// - ``split(key:into:stream:)``
@available(*, deprecated, message: "split is now available in the main MLX module")
@_disfavoredOverload
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
@available(*, deprecated, message: "uniform is now available in the main MLX module")
@_disfavoredOverload
public func uniform<R: HasDType, T>(
    _ range: Range<R>, _ shape: [Int] = [], type: T.Type = Float.self, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "uniform is now available in the main MLX module")
@_disfavoredOverload
public func uniform<T>(
    _ range: Range<Float> = 0 ..< 1, _ shape: [Int] = [], type: T.Type = Float.self,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "uniform is now available in the main MLX module")
@_disfavoredOverload
public func uniform<T>(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, type: T.Type = Float.self,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "uniform is now available in the main MLX module")
@_disfavoredOverload
public func uniform(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, dtype: DType = .float32,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "normal is now available in the main MLX module")
@_disfavoredOverload
public func normal<T>(
    _ shape: [Int] = [], type: T.Type = Float.self, loc: Float = 0, scale: Float = 1,
    key: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "normal is now available in the main MLX module")
@_disfavoredOverload
public func normal(
    _ shape: [Int] = [], dtype: DType = .float32, loc: Float = 0, scale: Float = 1,
    key: MLXArray? = nil,
    stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "multivariateNormal is now available in the main MLX module")
@_disfavoredOverload
public func multivariateNormal(
    mean: MLXArray, covariance: MLXArray, shape: [Int] = [], dtype: DType = .float32,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "randInt is now available in the main MLX module")
@_disfavoredOverload
public func randInt<T>(
    _ range: Range<T>, _ shape: [Int] = [], key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryInteger {
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
@available(*, deprecated, message: "randInt is now available in the main MLX module")
@_disfavoredOverload
public func randInt(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, key: MLXArray? = nil,
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
@available(*, deprecated, message: "randInt is now available in the main MLX module")
@_disfavoredOverload
public func randInt<T>(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, type: T.Type,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryInteger {
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
@available(*, deprecated, message: "bernoulli is now available in the main MLX module")
@_disfavoredOverload
public func bernoulli(_ shape: [Int] = [], key: MLXArray? = nil, stream: StreamOrDevice = .default)
    -> MLXArray
{
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
@available(*, deprecated, message: "bernoulli is now available in the main MLX module")
@_disfavoredOverload
public func bernoulli(
    _ p: ScalarOrArray, _ shape: [Int]? = nil, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "truncatedNormal is now available in the main MLX module")
@_disfavoredOverload
public func truncatedNormal<R: HasDType, T>(
    _ range: Range<R>, _ shape: [Int] = [], type: T.Type = Float.self, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "truncatedNormal is now available in the main MLX module")
@_disfavoredOverload
public func truncatedNormal<T>(
    _ range: Range<Float>, _ shape: [Int] = [], type: T.Type = Float.self, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "truncatedNormal is now available in the main MLX module")
@_disfavoredOverload
public func truncatedNormal<T>(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, type: T.Type = Float.self,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "truncatedNormal is now available in the main MLX module")
@_disfavoredOverload
public func truncatedNormal(
    low: ScalarOrArray, high: ScalarOrArray, _ shape: [Int]? = nil, dtype: DType = .float32,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
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
@available(*, deprecated, message: "gumbel is now available in the main MLX module")
@_disfavoredOverload
public func gumbel<T>(
    _ shape: [Int] = [], type: T.Type = Float.self, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray where T: HasDType, T: BinaryFloatingPoint {
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
@available(*, deprecated, message: "gumbel is now available in the main MLX module")
@_disfavoredOverload
public func gumbel(
    _ shape: [Int] = [], dtype: DType = .float32, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
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
///     - logits: The *unnormalized* categorical distribution(s).
@available(*, deprecated, message: "categorical is now available in the main MLX module")
@_disfavoredOverload
public func categorical(
    _ logits: MLXArray, axis: Int = -1, shape: [Int]? = nil, key: MLXArray? = nil,
    stream: StreamOrDevice = .default
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
///     - logits: The *unnormalized* categorical distribution(s).
@available(*, deprecated, message: "categorical is now available in the main MLX module")
@_disfavoredOverload
public func categorical(
    _ logits: MLXArray, axis: Int = -1, count: Int, key: MLXArray? = nil,
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
@available(*, deprecated, message: "laplace is now available in the main MLX module")
@_disfavoredOverload
public func laplace(
    _ shape: [Int] = [], dtype: DType = .float32, loc: Float = 0, scale: Float = 1,
    key: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXRandom.laplace(shape, dtype: dtype, loc: loc, scale: scale, key: key, stream: stream)
}
