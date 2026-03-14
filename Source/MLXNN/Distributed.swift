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

// MARK: - QuantizedAllToShardedLinear

/// Each member of the group applies part of the affine transformation with
/// a quantized matrix such that the result is sharded across the group.
///
/// It is the quantized equivalent of ``AllToShardedLinear``.
/// Similar to ``QuantizedLinear``, its parameters are frozen and will not be
/// included in any gradient computation.
///
/// ### See Also
/// - ``AllToShardedLinear``
/// - ``QuantizedShardedToAllLinear``
open class QuantizedAllToShardedLinear: Module, UnaryLayer, Quantized {

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray?
    public let bias: MLXArray?

    /// The distributed group. Stored as a plain property so it is excluded
    /// from `parameters()` and `children()`.
    public let group: DistributedGroup

    /// Initialize a ``QuantizedAllToShardedLinear`` layer.
    ///
    /// Validates that `outputDimensions` is divisible by the group size.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions (must be divisible by group size)
    ///   - bias: if `true`, apply a bias
    ///   - groupSize: the group size used for quantization. Default is 64.
    ///   - bits: the bit width used for quantization. Default is 4.
    ///   - mode: the quantization mode. Default is `.affine`.
    ///   - group: the distributed group (defaults to `MLXDistributed.init()`)
    public init(
        inputDimensions: Int, outputDimensions: Int, bias: Bool = true,
        groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine,
        group: DistributedGroup? = nil
    ) {
        let group = group ?? MLXDistributed.`init`()!
        self.group = group
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let N = group.size

        precondition(
            outputDimensions % N == 0,
            "Cannot shard the output of size \(outputDimensions) across \(N) devices."
        )

        let scale = sqrt(1.0 / Float(inputDimensions))
        let w = MLXRandom.uniform(
            low: -scale, high: scale, [outputDimensions / N, inputDimensions])
        let (quantizedWeight, scales, biases) = MLX.quantized(
            w, groupSize: groupSize, bits: bits, mode: mode)
        self.weight = quantizedWeight
        self.scales = scales
        self.biases = biases

        if bias {
            self.bias = MLXArray.zeros([outputDimensions / N])
        } else {
            self.bias = nil
        }
        super.init()

        self.freeze()
    }

    /// Internal initializer for providing arrays directly (used by `fromQuantizedLinear`).
    init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        group: DistributedGroup
    ) {
        self.weight = weight
        self.bias = bias
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.group = group
        super.init()

        self.freeze()
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let inDimsReal = (inDims * 32) / bits
        let outDimsReal = outDims * group.size
        return
            "(inputDimensions=\(inDimsReal), outputDimensions=\(outDimsReal), bias=\(bias != nil), groupSize=\(groupSize), bits=\(bits))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Aggregate the gradients coming from each shard
        var x = sumGradients(group: group)(x)

        x = quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        if let bias {
            x = x + bias
        }
        return x
    }

    /// Create a ``QuantizedAllToShardedLinear`` from an existing ``QuantizedLinear`` layer.
    ///
    /// For a size-1 group, the sharded weights are identical to the original.
    ///
    /// - Parameters:
    ///   - quantizedLinear: the quantized linear layer to convert
    ///   - segments: number of segments for fused weights (e.g. 3 for QKV). Default is 1.
    ///   - group: the distributed group
    /// - Returns: a new ``QuantizedAllToShardedLinear`` layer with sharded weights
    public class func fromQuantizedLinear(
        _ quantizedLinear: QuantizedLinear, segments: Int = 1,
        group: DistributedGroup? = nil
    ) -> QuantizedAllToShardedLinear {
        let group = group ?? MLXDistributed.`init`()!
        let (outputDimensions, inputDimensions) = quantizedLinear.weight.shape2
        let inputDimsReal = (inputDimensions * 32) / quantizedLinear.bits

        let layer = QuantizedAllToShardedLinear(
            inputDimensions: inputDimsReal, outputDimensions: outputDimensions,
            bias: quantizedLinear.bias != nil,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits,
            mode: quantizedLinear.mode,
            group: group)

        // Shard the parameters from the original quantized linear layer
        let shardedParams = shardParameterTree(
            quantizedLinear.parameters(), predicate: allToShardedPredicate(segments: segments),
            group: group)
        layer.update(parameters: shardedParams)

        return layer
    }
}

// MARK: - QuantizedShardedToAllLinear

/// Each member of the group applies part of the affine transformation using
/// the quantized matrix and then aggregates the results.
///
/// All nodes will have the same exact result after this layer.
///
/// It is the quantized equivalent of ``ShardedToAllLinear``.
/// Similar to ``QuantizedLinear``, its parameters are frozen and will not be
/// included in any gradient computation.
///
/// ### See Also
/// - ``ShardedToAllLinear``
/// - ``QuantizedAllToShardedLinear``
open class QuantizedShardedToAllLinear: Module, UnaryLayer, Quantized {

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray?
    public let bias: MLXArray?

    /// The distributed group. Stored as a plain property so it is excluded
    /// from `parameters()` and `children()`.
    public let group: DistributedGroup

    /// Initialize a ``QuantizedShardedToAllLinear`` layer.
    ///
    /// Validates that `inputDimensions` is divisible by the group size.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input dimensions (must be divisible by group size)
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true`, apply a bias
    ///   - groupSize: the group size used for quantization. Default is 64.
    ///   - bits: the bit width used for quantization. Default is 4.
    ///   - mode: the quantization mode. Default is `.affine`.
    ///   - group: the distributed group (defaults to `MLXDistributed.init()`)
    public init(
        inputDimensions: Int, outputDimensions: Int, bias: Bool = true,
        groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine,
        group: DistributedGroup? = nil
    ) {
        let group = group ?? MLXDistributed.`init`()!
        self.group = group
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let N = group.size

        precondition(
            inputDimensions % N == 0,
            "The input of size \(inputDimensions) cannot be sharded across \(N) devices."
        )

        let scale = sqrt(1.0 / Float(inputDimensions))
        let w = MLXRandom.uniform(
            low: -scale, high: scale, [outputDimensions, inputDimensions / N])
        let (quantizedWeight, scales, biases) = MLX.quantized(
            w, groupSize: groupSize, bits: bits, mode: mode)
        self.weight = quantizedWeight
        self.scales = scales
        self.biases = biases

        if bias {
            self.bias = MLXArray.zeros([outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()

        self.freeze()
    }

    /// Internal initializer for providing arrays directly (used by `fromQuantizedLinear`).
    init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        group: DistributedGroup
    ) {
        self.weight = weight
        self.bias = bias
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.group = group
        super.init()

        self.freeze()
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    open override func describeExtra(_ indent: Int) -> String {
        let (outDims, inDims) = weight.shape2
        let inDimsReal = (inDims * 32) / bits * group.size
        return
            "(inputDimensions=\(inDimsReal), outputDimensions=\(outDims), bias=\(bias != nil), groupSize=\(groupSize), bits=\(bits))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }

    /// Create a ``QuantizedShardedToAllLinear`` from an existing ``QuantizedLinear`` layer.
    ///
    /// For a size-1 group, the sharded weights are identical to the original.
    ///
    /// - Parameters:
    ///   - quantizedLinear: the quantized linear layer to convert
    ///   - segments: number of segments for fused weights (e.g. 3 for QKV). Default is 1.
    ///   - group: the distributed group
    /// - Returns: a new ``QuantizedShardedToAllLinear`` layer with sharded weights
    public class func fromQuantizedLinear(
        _ quantizedLinear: QuantizedLinear, segments: Int = 1,
        group: DistributedGroup? = nil
    ) -> QuantizedShardedToAllLinear {
        let group = group ?? MLXDistributed.`init`()!
        let (outputDimensions, inputDimensions) = quantizedLinear.weight.shape2
        let inputDimsReal = (inputDimensions * 32) / quantizedLinear.bits

        let layer = QuantizedShardedToAllLinear(
            inputDimensions: inputDimsReal, outputDimensions: outputDimensions,
            bias: quantizedLinear.bias != nil,
            groupSize: quantizedLinear.groupSize,
            bits: quantizedLinear.bits,
            mode: quantizedLinear.mode,
            group: group)

        // Shard the parameters from the original quantized linear layer
        let shardedParams = shardParameterTree(
            quantizedLinear.parameters(), predicate: shardedToAllPredicate(segments: segments),
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

// MARK: - ShardingType

/// Describes the type of sharding for distributed linear layers.
///
/// - ``allToSharded``: Common (replicated) input is projected into a sharded
///   representation. Each rank holds a slice of the output features.
/// - ``shardedToAll``: Sharded input is projected and then aggregated so that
///   every rank obtains the full (common) output.
///
/// ### See Also
/// - ``shardLinear(module:sharding:segments:group:)``
/// - ``shardInPlace(module:sharding:segments:group:)``
public enum ShardingType {
    case allToSharded
    case shardedToAll
}

// MARK: - shardLinear

/// Create a new distributed linear layer from an existing ``Linear`` or
/// ``QuantizedLinear``.
///
/// The returned layer has its parameters sharded across the group and
/// performs distributed communication in either the forward or backward pass
/// depending on the sharding type.
///
/// - Parameters:
///   - module: the ``Linear`` or ``QuantizedLinear`` layer to shard
///   - sharding: the type of sharding (``ShardingType/allToSharded`` or
///     ``ShardingType/shardedToAll``)
///   - segments: number of segments for fused weights (e.g. 3 for QKV).
///     Default is 1.
///   - group: the distributed group. If `nil`, uses `MLXDistributed.init()`.
/// - Returns: a new distributed ``Module`` with sharded parameters
///
/// ### See Also
/// - ``shardInPlace(module:sharding:segments:group:)``
/// - ``AllToShardedLinear``
/// - ``ShardedToAllLinear``
public func shardLinear(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
) -> Module {
    // QuantizedLinear must be checked before Linear because QuantizedLinear
    // is a subclass of Linear and would otherwise match the Linear case.
    switch (sharding, module) {
    case (.allToSharded, let quantized as QuantizedLinear):
        return QuantizedAllToShardedLinear.fromQuantizedLinear(
            quantized, segments: segments, group: group)
    case (.allToSharded, let linear as Linear):
        return AllToShardedLinear.fromLinear(linear, segments: segments, group: group)
    case (.shardedToAll, let quantized as QuantizedLinear):
        return QuantizedShardedToAllLinear.fromQuantizedLinear(
            quantized, segments: segments, group: group)
    case (.shardedToAll, let linear as Linear):
        return ShardedToAllLinear.fromLinear(linear, segments: segments, group: group)
    default:
        preconditionFailure(
            "shardLinear: unsupported module type \(type(of: module)). "
                + "Expected Linear or QuantizedLinear.")
    }
}

// MARK: - shardInPlace

/// Shard a module's parameters in-place using ``Module/update(parameters:)``.
///
/// Unlike ``shardLinear(module:sharding:segments:group:)`` which returns a new
/// distributed layer type, this function modifies the parameters of the
/// existing module without changing its type. The module itself must
/// natively support distributed communication for the collective ops to
/// take effect.
///
/// - Parameters:
///   - module: the module whose parameters will be sharded in-place
///   - sharding: the type of sharding (``ShardingType/allToSharded`` or
///     ``ShardingType/shardedToAll``), or a custom predicate
///   - segments: number of segments for fused weights (e.g. 3 for QKV).
///     Default is 1.
///   - group: the distributed group. If `nil`, uses `MLXDistributed.init()`.
///
/// ### See Also
/// - ``shardLinear(module:sharding:segments:group:)``
public func shardInPlace(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
) {
    let group = group ?? MLXDistributed.`init`()!
    let predicate: (String, MLXArray) -> ShardInfo?

    switch sharding {
    case .allToSharded:
        predicate = allToShardedPredicate(segments: segments)
    case .shardedToAll:
        predicate = shardedToAllPredicate(segments: segments)
    }

    let shardedParams = shardParameterTree(
        module.parameters(), predicate: predicate, group: group)
    module.update(parameters: shardedParams)
}

// MARK: - averageGradients

/// Average a gradient tree across the processes in the distributed group.
///
/// When the group has a single member the gradients are returned unchanged.
/// Otherwise each gradient array is sum-reduced across the group and divided
/// by the group size.
///
/// This helper supports batching small gradient arrays into larger
/// concatenated chunks before performing the all-reduce, which can improve
/// networking performance.
///
/// - Parameters:
///   - gradients: the gradient tree (typically from ``Module/parameters()``
///     or ``Module/trainableParameters()``)
///   - group: the distributed group. If `nil`, uses `MLXDistributed.init()`.
///   - allReduceSize: maximum byte size for batching gradient arrays into a
///     single all-reduce call. Set to 0 or negative to disable batching.
///     Default is 32 MiB.
///   - communicationType: if provided, cast each gradient to this type before
///     communication and cast back to the original type after. Typically used
///     to cast to a smaller float (e.g. `.float16`) to reduce communication
///     size. Default is `nil`.
///   - communicationStream: optional stream for the communication. If `nil`,
///     the default stream is used.
/// - Returns: the averaged gradient tree with the same structure as the input
///
/// ### See Also
/// - ``shardLinear(module:sharding:segments:group:)``
/// - ``shardInPlace(module:sharding:segments:group:)``
public func averageGradients(
    gradients: ModuleParameters,
    group: DistributedGroup? = nil,
    allReduceSize: Int = 32 * 1024 * 1024,
    communicationType: DType? = nil,
    communicationStream: StreamOrDevice? = nil
) -> ModuleParameters {
    let group = group ?? MLXDistributed.`init`()!
    let N = group.size

    if N == 1 {
        return gradients
    }

    let stream: StreamOrDevice = communicationStream ?? .default

    // Helper to average a single gradient array, optionally casting to
    // communicationType before the all-reduce and back after.
    func average(_ x: MLXArray) -> MLXArray {
        let dt = x.dtype
        let y = communicationType != nil ? x.asType(communicationType!) : x
        return (MLXDistributed.allSum(y, group: group, stream: stream)).asType(dt) / Float(N)
    }

    if allReduceSize <= 0 {
        // No batching: average each gradient independently
        return gradients.mapValues(transform: { array in
            average(array)
        })
    }

    // Batched mode: concatenate small gradients, reduce, split back
    let flat = gradients.flattened()
    if flat.isEmpty {
        return gradients
    }

    // Collect metadata
    let keys = flat.map { $0.0 }
    let values = flat.map { $0.1 }
    let shapes = values.map { $0.shape }
    let sizes = values.map { $0.size }
    let dtypes = values.map { $0.dtype }

    // Check for mixed types -- if mixed, fall back to non-batched
    let firstDtype = dtypes[0]
    if !dtypes.allSatisfy({ $0 == firstDtype }) {
        return averageGradients(
            gradients: gradients, group: group, allReduceSize: 0,
            communicationType: communicationType,
            communicationStream: communicationStream)
    }

    // Use communicationType size for batching threshold if provided,
    // matching Python's behavior
    let itemSize = communicationType?.size ?? firstDtype.size

    // Group gradients into batches that are at least allReduceSize bytes
    var gradGroups = [[Int]]()
    var currentGroup = [Int]()
    var currentSize = 0

    for i in 0 ..< keys.count {
        currentGroup.append(i)
        currentSize += sizes[i] * itemSize
        if currentSize >= allReduceSize {
            gradGroups.append(currentGroup)
            currentGroup = []
            currentSize = 0
        }
    }
    if !currentGroup.isEmpty {
        gradGroups.append(currentGroup)
    }

    // Concatenate-reduce-split for each group
    var newFlat = [(String, MLXArray)]()
    for group in gradGroups {
        // Flatten each gradient to 1D and concatenate
        let flatArrays = group.map { values[$0].reshaped(-1) }
        let bigGrad = concatenated(flatArrays, axis: 0)

        // Average the concatenated gradient
        let averaged = average(bigGrad)

        // Split back using cumulative sizes as indices
        var indices = [Int]()
        var cumulative = 0
        for (i, idx) in group.enumerated() {
            cumulative += sizes[idx]
            if i < group.count - 1 {
                indices.append(cumulative)
            }
        }

        let splitGrads: [MLXArray]
        if indices.isEmpty {
            splitGrads = [averaged]
        } else {
            splitGrads = split(averaged, indices: indices, axis: 0)
        }

        for (i, idx) in group.enumerated() {
            let reshaped = splitGrads[i].reshaped(shapes[idx])
            newFlat.append((keys[idx], reshaped))
        }
    }

    return ModuleParameters.unflattened(newFlat)
}
