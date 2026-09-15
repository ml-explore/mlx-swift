// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How a linear layer is sharded across a group.
public enum ShardingType: Sendable {
    /// A common input becomes a sharded output.
    case allToSharded

    /// A sharded input becomes a common output.
    case shardedToAll
}

/// Describes how an unsharded weight is composed.
///
/// A fused QKV matrix, for example, is three segments stacked together and
/// each has to be sharded separately.
public enum Segments: Sendable {
    /// The weight is `count` equally sized segments.
    case count(Int)

    /// The weight is split at these indices.
    case indices([Int])

    /// The weight is split at these fractions of the axis.
    case fractions([Double])

    func split(_ weight: MLXArray, axis: Int) -> [MLXArray] {
        switch self {
        case .count(let count):
            count <= 1 ? [weight] : weight.split(parts: count, axis: axis)
        case .indices(let indices):
            weight.split(indices: indices, axis: axis)
        case .fractions(let fractions):
            weight.split(
                indices: fractions.map { Int($0 * Double(weight.dim(axis))) }, axis: axis)
        }
    }
}

/// Returns a function that is the identity in the forward pass and sums the
/// gradients across the group in the backward pass.
///
/// - Parameter group: the group, or `nil` to use the global group
public func sumGradients(group: MLXDistributed.Group? = nil) -> (MLXArray) -> MLXArray {
    let group = group ?? MLXDistributed.initialize()
    if group.size == 1 {
        return { $0 }
    }

    let f = CustomFunction {
        Forward { inputs in
            inputs
        }
        VJP { _, cotangents in
            cotangents.map { MLXDistributed.allSum($0, group: group) }
        }
    }

    return { f([$0])[0] }
}

/// The sharding axis and segments for a parameter, or `nil` to leave it alone.
private typealias ShardingPredicate = (String, MLXArray) -> (axis: Int, segments: Segments)?

private func allToShardedPredicate(_ segments: Segments) -> ShardingPredicate {
    { path, weight in
        path.hasSuffix("bias") ? (-1, segments) : (max(weight.ndim - 2, 0), segments)
    }
}

private func shardedToAllPredicate(_ segments: Segments) -> ShardingPredicate {
    { path, _ in
        path.hasSuffix("bias") ? nil : (-1, segments)
    }
}

private func predicate(for sharding: ShardingType, segments: Segments) -> ShardingPredicate {
    switch sharding {
    case .allToSharded: allToShardedPredicate(segments)
    case .shardedToAll: shardedToAllPredicate(segments)
    }
}

/// Returns a new parameter tree with the weights sharded across the group.
private func shard(
    _ parameters: ModuleParameters, group: MLXDistributed.Group,
    _ sharding: ShardingPredicate
) -> ModuleParameters {
    let size = group.size
    if size == 1 {
        return parameters
    }
    let rank = group.rank

    let sharded = parameters.flattened().map { path, weight -> (String, MLXArray) in
        guard let (axis, segments) = sharding(path, weight) else {
            return (path, weight)
        }
        let parts = segments.split(weight, axis: axis).map {
            $0.split(parts: size, axis: axis)[rank]
        }
        return (path, concatenated(parts, axis: axis).contiguous())
    }

    return ModuleParameters.unflattened(sharded)
}

/// Shard a module in place by replacing its parameters with sharded ones.
///
/// The module itself is unchanged, so distributed communication only happens
/// if the module supports it natively.
///
/// - Parameters:
///   - module: the module whose parameters are sharded in place
///   - sharding: the kind of sharding to apply
///   - segments: the segments that comprise each unsharded weight
///   - group: the group to shard across, or `nil` to use the global group
public func shardInPlace(
    _ module: Module, sharding: ShardingType, segments: Segments = .count(1),
    group: MLXDistributed.Group? = nil
) {
    let group = group ?? MLXDistributed.initialize()
    _ = module.update(
        parameters: shard(
            module.parameters(), group: group, predicate(for: sharding, segments: segments)))
}

/// Create a new linear layer with sharded parameters that also performs the
/// distributed communication, either in the forward or the backward pass.
///
/// Unlike ``shardInPlace(_:sharding:segments:group:)`` the original layer is
/// not changed.
///
/// - Parameters:
///   - layer: the linear layer to shard
///   - sharding: the kind of sharding to apply
///   - segments: the segments that comprise each unsharded weight
///   - group: the group to shard across, or `nil` to use the global group
public func shardLinear(
    _ layer: Linear, sharding: ShardingType, segments: Segments = .count(1),
    group: MLXDistributed.Group? = nil
) -> Module {
    switch sharding {
    case .allToSharded:
        AllToShardedLinear(layer, segments: segments, group: group)
    case .shardedToAll:
        ShardedToAllLinear(layer, segments: segments, group: group)
    }
}

/// Each member of the group applies part of the affine transformation so that
/// the result is sharded across the group.
///
/// The gradients are automatically aggregated from each member of the group.
open class AllToShardedLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?

    let group: MLXDistributed.Group
    private let aggregateGradients: (MLXArray) -> MLXArray

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions, sharded across the group
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) {
        let group = group ?? MLXDistributed.initialize()
        let size = group.size
        precondition(
            outputDimensions % size == 0,
            "Cannot shard the output of size \(outputDimensions) across \(size) devices.")

        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions / size, inputDimensions])
        self.bias = bias ? MLXRandom.uniform(-scale ..< scale, [outputDimensions / size]) : nil
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
    }

    /// Create a sharded layer from an existing ``Linear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) {
        let group = group ?? MLXDistributed.initialize()
        let (outputDimensions, inputDimensions) = other.shape

        self.init(inputDimensions, outputDimensions, bias: other.bias != nil, group: group)
        _ = update(
            parameters: shard(
                other.parameters(), group: group, allToShardedPredicate(segments)))
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // aggregate the gradients coming from each shard
        let x = aggregateGradients(x)

        if let bias {
            return addMM(bias, x, weight.T)
        } else {
            return x.matmul(weight.T)
        }
    }
}

/// Each member of the group applies part of the affine transformation and the
/// results are then aggregated.
///
/// Every member of the group ends up with the same result.
open class ShardedToAllLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?

    let group: MLXDistributed.Group

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions, sharded across the group
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) {
        let group = group ?? MLXDistributed.initialize()
        let size = group.size
        precondition(
            inputDimensions % size == 0,
            "The input of size \(inputDimensions) cannot be sharded across \(size) devices.")

        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions / size])
        self.bias = bias ? MLXRandom.uniform(-scale ..< scale, [outputDimensions]) : nil
        self.group = group
    }

    /// Create a sharded layer from an existing ``Linear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) {
        let group = group ?? MLXDistributed.initialize()
        let (outputDimensions, inputDimensions) = other.shape

        self.init(inputDimensions, outputDimensions, bias: other.bias != nil, group: group)
        _ = update(
            parameters: shard(
                other.parameters(), group: group, shardedToAllPredicate(segments)))
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x.matmul(weight.T)
        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }
}

/// Average the gradients across the processes in the group.
///
/// Small gradients are concatenated into batches of at least `allReduceSize`
/// bytes so that they are communicated in one step, which is considerably
/// faster than one call per array.
///
/// - Parameters:
///   - gradients: the gradients, which must have the same structure in every process
///   - group: the group to average across, or `nil` to use the global group
///   - allReduceSize: group arrays until their size in bytes exceeds this,
///     or `0` to disable grouping
///   - stream: stream to evaluate on
public func averageGradients(
    _ gradients: ModuleParameters, group: MLXDistributed.Group? = nil,
    allReduceSize: Int = 32 * 1024 * 1024, stream: StreamOrDevice = .cpu
) -> ModuleParameters {
    let group = group ?? MLXDistributed.initialize()
    let size = group.size
    if size == 1 {
        return gradients
    }

    let flat = gradients.flattened()
    if flat.isEmpty {
        return gradients
    }

    // one all reduce per gradient
    if allReduceSize <= 0 {
        return ModuleParameters.unflattened(
            flat.map {
                ($0.0, MLXDistributed.allSum($0.1, group: group, stream: stream) / size)
            })
    }

    // arrays of mixed types cannot be concatenated
    let dtype = flat[0].1.dtype
    guard flat.allSatisfy({ $0.1.dtype == dtype }) else {
        return averageGradients(
            gradients, group: group, allReduceSize: 0, stream: stream)
    }

    // gather the gradients into groups that are at least allReduceSize bytes
    var batches = [[Int]]()
    var batch = [Int]()
    var batchBytes = 0
    for (i, (_, gradient)) in flat.enumerated() {
        batch.append(i)
        batchBytes += gradient.nbytes
        if batchBytes >= allReduceSize {
            batches.append(batch)
            batch = []
            batchBytes = 0
        }
    }
    if !batch.isEmpty {
        batches.append(batch)
    }

    // concatenate, reduce, split
    var result = [(String, MLXArray)]()
    for batch in batches {
        let sizes = batch.map { flat[$0].1.size }
        var big = concatenated(batch.map { flat[$0].1.reshaped([-1]) })
        big = MLXDistributed.allSum(big, group: group, stream: stream) / size

        let indices = sizes.dropLast().reduce(into: [Int]()) { $0.append(($0.last ?? 0) + $1) }
        let parts = indices.isEmpty ? [big] : big.split(indices: indices)

        for (part, i) in zip(parts, batch) {
            result.append((flat[i].0, part.reshaped(flat[i].1.shape)))
        }
    }

    return ModuleParameters.unflattened(result)
}
