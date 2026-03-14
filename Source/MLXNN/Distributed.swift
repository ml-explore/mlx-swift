// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - sumGradients Helper

/// Cache of `sumGradients` closures keyed by group identity (ObjectIdentifier).
///
/// Each closure uses `CustomFunction` with an identity forward pass and an
/// `allSum` VJP so that gradients are aggregated across the distributed group
/// during backpropagation.
private var _sumGradientsCache = [ObjectIdentifier: (MLXArray) -> MLXArray]()
private let _sumGradientsCacheLock = NSLock()

/// Returns a closure that is the identity in the forward pass but performs
/// `allSum` on the cotangents during the backward pass.
///
/// The result is cached per group instance.
///
/// - Parameter group: the distributed group to aggregate gradients over
/// - Returns: a closure `(MLXArray) -> MLXArray` that is identity forward,
///   allSum backward
public func sumGradients(group: DistributedGroup) -> (MLXArray) -> MLXArray {
    let key = ObjectIdentifier(group)

    return _sumGradientsCacheLock.withLock {
        if let cached = _sumGradientsCache[key] {
            return cached
        }

        if group.size == 1 {
            // Optimization: on a size-1 group, just return identity
            let fn: (MLXArray) -> MLXArray = { x in x }
            _sumGradientsCache[key] = fn
            return fn
        }

        // Build a CustomFunction with identity forward and allSum VJP
        let cf = CustomFunction {
            Forward { inputs in inputs }
            VJP { _, cotangents in
                cotangents.map { MLXDistributed.allSum($0, group: group) }
            }
        }

        let fn: (MLXArray) -> MLXArray = { x in
            cf([x])[0]
        }
        _sumGradientsCache[key] = fn
        return fn
    }
}

// MARK: - AllToShardedLinear

/// Each member of the group applies part of the affine transformation such
/// that the result is sharded across the group.
///
/// The gradients are automatically aggregated from each member of the group
/// via ``sumGradients(group:)``.
///
/// ### See Also
/// - ``ShardedToAllLinear``
open class AllToShardedLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?

    /// The distributed group. Stored as a plain property so it is excluded
    /// from `parameters()` and `children()`.
    public let group: DistributedGroup

    /// Initialize an ``AllToShardedLinear`` layer.
    ///
    /// Validates that `outputDimensions` is divisible by the group size.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions (must be divisible by group size)
    ///   - bias: if `true`, apply a bias
    ///   - group: the distributed group (defaults to `MLXDistributed.init()`)
    public init(
        inputDimensions: Int, outputDimensions: Int, bias: Bool = true,
        group: DistributedGroup? = nil
    ) {
        let group = group ?? MLXDistributed.`init`()!
        self.group = group
        let N = group.size

        precondition(
            outputDimensions % N == 0,
            "Cannot shard the output of size \(outputDimensions) across \(N) devices."
        )

        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale, [outputDimensions / N, inputDimensions])
        if bias {
            self.bias = MLXRandom.uniform(
                low: -scale, high: scale, [outputDimensions / N])
        } else {
            self.bias = nil
        }
        super.init()
    }

    /// Internal initializer for providing weight and bias directly (used by `fromLinear`).
    init(weight: MLXArray, bias: MLXArray?, group: DistributedGroup) {
        self.weight = weight
        self.bias = bias
        self.group = group
        super.init()
    }

    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let N = group.size
        return
            "(inputDimensions=\(inDims), outputDimensions=\(outDims * N), bias=\(bias != nil))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Aggregate the gradients coming from each shard
        var x = sumGradients(group: group)(x)

        // Compute the affine projection
        if let bias {
            x = addMM(bias, x, weight.T)
        } else {
            x = matmul(x, weight.T)
        }
        return x
    }

    /// Create an ``AllToShardedLinear`` from an existing ``Linear`` layer.
    ///
    /// For a size-1 group, the sharded weights are identical to the original.
    ///
    /// - Parameters:
    ///   - linear: the linear layer to convert
    ///   - segments: number of segments for fused weights (e.g. 3 for QKV). Default is 1.
    ///   - group: the distributed group
    /// - Returns: a new ``AllToShardedLinear`` layer with sharded weights
    public class func fromLinear(
        _ linear: Linear, segments: Int = 1, group: DistributedGroup? = nil
    ) -> AllToShardedLinear {
        let group = group ?? MLXDistributed.`init`()!
        let (outputDimensions, inputDimensions) = linear.weight.shape2

        let layer = AllToShardedLinear(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions,
            bias: linear.bias != nil, group: group)

        // Shard the parameters from the original linear layer
        let shardedParams = shardParameterTree(
            linear.parameters(), predicate: allToShardedPredicate(segments: segments),
            group: group)
        layer.update(parameters: shardedParams)

        return layer
    }
}

// MARK: - ShardedToAllLinear

/// Each member of the group applies part of the affine transformation and
/// then aggregates the results via `allSum`.
///
/// All nodes will have the same exact result after this layer.
///
/// ### See Also
/// - ``AllToShardedLinear``
open class ShardedToAllLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?

    /// The distributed group. Stored as a plain property so it is excluded
    /// from `parameters()` and `children()`.
    public let group: DistributedGroup

    /// Initialize a ``ShardedToAllLinear`` layer.
    ///
    /// Validates that `inputDimensions` is divisible by the group size.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions (must be divisible by group size)
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true`, apply a bias
    ///   - group: the distributed group (defaults to `MLXDistributed.init()`)
    public init(
        inputDimensions: Int, outputDimensions: Int, bias: Bool = true,
        group: DistributedGroup? = nil
    ) {
        let group = group ?? MLXDistributed.`init`()!
        self.group = group
        let N = group.size

        precondition(
            inputDimensions % N == 0,
            "The input of size \(inputDimensions) cannot be sharded across \(N) devices."
        )

        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale, [outputDimensions, inputDimensions / N])
        if bias {
            self.bias = MLXRandom.uniform(
                low: -scale, high: scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()
    }

    /// Internal initializer for providing weight and bias directly (used by `fromLinear`).
    init(weight: MLXArray, bias: MLXArray?, group: DistributedGroup) {
        self.weight = weight
        self.bias = bias
        self.group = group
        super.init()
    }

    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let N = group.size
        return
            "(inputDimensions=\(inDims * N), outputDimensions=\(outDims), bias=\(bias != nil))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = matmul(x, weight.T)

        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }

    /// Create a ``ShardedToAllLinear`` from an existing ``Linear`` layer.
    ///
    /// For a size-1 group, the sharded weights are identical to the original.
    ///
    /// - Parameters:
    ///   - linear: the linear layer to convert
    ///   - segments: number of segments for fused weights (e.g. 3 for QKV). Default is 1.
    ///   - group: the distributed group
    /// - Returns: a new ``ShardedToAllLinear`` layer with sharded weights
    public class func fromLinear(
        _ linear: Linear, segments: Int = 1, group: DistributedGroup? = nil
    ) -> ShardedToAllLinear {
        let group = group ?? MLXDistributed.`init`()!
        let (outputDimensions, inputDimensions) = linear.weight.shape2

        let layer = ShardedToAllLinear(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions,
            bias: linear.bias != nil, group: group)

        // Shard the parameters from the original linear layer
        let shardedParams = shardParameterTree(
            linear.parameters(), predicate: shardedToAllPredicate(segments: segments),
            group: group)
        layer.update(parameters: shardedParams)

        return layer
    }
}

// MARK: - Internal Sharding Helpers

/// Sharding predicate result: axis to shard on, and number of segments.
/// Returns `nil` if the parameter should not be sharded.
private typealias ShardInfo = (axis: Int, segments: Int)

/// Returns a sharding predicate for "all-to-sharded" conversion.
///
/// For bias: shard along last axis (-1). For weight: shard along axis 0
/// (max(ndim - 2, 0) in Python, which is axis 0 for 2D weights).
private func allToShardedPredicate(segments: Int) -> (String, MLXArray) -> ShardInfo? {
    return { path, weight in
        if path.hasSuffix("bias") {
            return (axis: -1, segments: segments)
        }
        // For 2D weight [outDims, inDims], max(ndim - 2, 0) = 0
        return (axis: max(weight.ndim - 2, 0), segments: segments)
    }
}

/// Returns a sharding predicate for "sharded-to-all" conversion.
///
/// For bias: don't shard (return nil). For weight: shard along last axis (-1).
private func shardedToAllPredicate(segments: Int) -> (String, MLXArray) -> ShardInfo? {
    return { path, weight in
        if path.hasSuffix("bias") {
            return nil
        }
        return (axis: -1, segments: segments)
    }
}

/// Shard a flat parameter tree according to the given predicate and group.
///
/// This mirrors the Python `_shard` function using `tree_map_with_path`.
/// For each parameter, the predicate determines the sharding axis and segments.
/// The weight is split into segments, each segment is split across the group,
/// and the rank-local shard is taken and concatenated.
private func shardParameterTree(
    _ parameters: ModuleParameters,
    predicate: (String, MLXArray) -> ShardInfo?,
    group: DistributedGroup
) -> ModuleParameters {
    let N = group.size
    let r = group.rank

    // Flatten to get (path, MLXArray) pairs
    let flat = parameters.flattened()

    // Shard each parameter
    let sharded = flat.map { (path, value) -> (String, MLXArray) in
        guard let info = predicate(path, value) else {
            return (path, value)
        }

        var axis = info.axis
        let segments = info.segments

        // Normalize negative axis
        if axis < 0 {
            axis = value.ndim + axis
        }

        // Split into segments, then split each segment across group, take rank-th part
        let segmentParts: [MLXArray]
        if segments > 1 {
            segmentParts = value.split(parts: segments, axis: axis)
        } else {
            segmentParts = [value]
        }

        let shardedParts = segmentParts.map { part -> MLXArray in
            let groupParts = part.split(parts: N, axis: axis)
            return groupParts[r]
        }

        let result: MLXArray
        if shardedParts.count > 1 {
            result = concatenated(shardedParts, axis: axis).contiguous()
        } else {
            result = shardedParts[0].contiguous()
        }

        return (path, result)
    }

    return ModuleParameters.unflattened(sharded)
}
