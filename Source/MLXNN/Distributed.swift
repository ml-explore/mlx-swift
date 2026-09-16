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

/// A layer cannot be sharded across a group.
public enum ShardingError: Error, CustomStringConvertible {
    /// A dimension is not divisible by the size of the group.
    case indivisible(dimension: String, of: Int, across: Int)

    /// A shard would not hold whole quantization groups.
    case quantizationGroup(inputDimensions: Int, groupSize: Int, across: Int)

    /// The layer being sharded is missing a parameter.
    case missingParameter(String)

    public var description: String {
        switch self {
        case .indivisible(let dimension, let value, let size):
            "Cannot shard the \(dimension) of size \(value) across \(size) processes."
        case .quantizationGroup(let inputDimensions, let groupSize, let size):
            """
            Sharding \(inputDimensions) inputs across \(size) processes splits a quantization \
            group of \(groupSize).
            """
        case .missingParameter(let name):
            "The layer being sharded has no \(name)."
        }
    }
}

/// Returns a function that is the identity in the forward pass and sums the
/// gradients across the group in the backward pass.
///
/// - Parameter group: the group to sum across
public func sumGradients(group: MLXDistributed.Group) -> (MLXArray) -> MLXArray {
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

/// Shard the rows -- the output dimensions -- of every parameter.
///
/// `scales` and `biases` hold one row per output row, so they shard the same
/// way as the weight.  `bias`, the affine bias, is one dimensional.
private func allToShardedPredicate(_ segments: Segments) -> ShardingPredicate {
    { path, weight in
        path.hasSuffix("bias") ? (-1, segments) : (max(weight.ndim - 2, 0), segments)
    }
}

/// Shard the columns -- the input dimensions -- of every parameter.
///
/// The affine bias applies to the summed result, so every process keeps it
/// whole.
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

/// Returns a new parameter tree holding this process' shard of the weights.
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
        // a scalar -- the NVFP4 global scale, for example -- is the same in
        // every process
        guard weight.ndim > 0, let (axis, segments) = sharding(path, weight) else {
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
) throws {
    let group = try group ?? MLXDistributed.initialize()
    _ = module.update(
        parameters: shard(
            module.parameters(), group: group, predicate(for: sharding, segments: segments)))
}

/// Create a new linear layer with sharded parameters that also performs the
/// distributed communication, either in the forward or the backward pass.
///
/// Unlike ``shardInPlace(_:sharding:segments:group:)`` the original layer is
/// not changed.  A ``QuantizedLinear`` yields the quantized flavor of the
/// sharded layer, so a quantized model shards like any other.
///
/// - Parameters:
///   - layer: the linear layer to shard
///   - sharding: the kind of sharding to apply
///   - segments: the segments that comprise each unsharded weight
///   - group: the group to shard across, or `nil` to use the global group
/// - Returns: the sharded layer, which can replace `layer` in a model
public func shardLinear(
    _ layer: Linear, sharding: ShardingType, segments: Segments = .count(1),
    group: MLXDistributed.Group? = nil
) throws -> Linear {
    // QuantizedLinear is a Linear, so it has to be matched first
    if let layer = layer as? QuantizedLinear {
        return switch sharding {
        case .allToSharded:
            try QuantizedAllToShardedLinear(layer, segments: segments, group: group)
        case .shardedToAll:
            try QuantizedShardedToAllLinear(layer, segments: segments, group: group)
        }
    }

    return switch sharding {
    case .allToSharded:
        try AllToShardedLinear(layer, segments: segments, group: group)
    case .shardedToAll:
        try ShardedToAllLinear(layer, segments: segments, group: group)
    }
}

/// Each member of the group applies part of the affine transformation so that
/// the result is sharded across the group.
///
/// The gradients are automatically aggregated from each member of the group.
///
/// This is a ``Linear``, so it can replace one in a model.
open class AllToShardedLinear: Linear {

    /// The group the output dimensions are sharded across.
    public let group: MLXDistributed.Group

    private let aggregateGradients: (MLXArray) -> MLXArray

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions, sharded across the group
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        self.group = group
        self.aggregateGradients = sumGradients(group: group)

        let scale = sqrt(1.0 / Float(inputDimensions))
        let shardedOutput = outputDimensions / group.size
        super.init(
            weight: MLXRandom.uniform(-scale ..< scale, [shardedOutput, inputDimensions]),
            bias: bias ? MLXRandom.uniform(-scale ..< scale, [shardedOutput]) : nil)
    }

    /// Hold parameters that are already this process' shard.
    init(shardedWeight weight: MLXArray, bias: MLXArray?, group: MLXDistributed.Group) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(weight: weight, bias: bias)
    }

    /// Create a sharded layer from an existing ``Linear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (outputDimensions, _) = other.shape
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        let parameters = Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, allToShardedPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }

        self.init(shardedWeight: weight, bias: parameters["bias"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // every shard reads the same input, so their gradients are summed
        let x = aggregateGradients(x)

        if let bias {
            return addMM(bias, x, weight.T)
        } else {
            return matmul(x, weight.T)
        }
    }

    /// Quantizing keeps the layer sharded.
    ///
    /// Without this a quantized model built by ``quantize(model:groupSize:bits:predicate:)``
    /// would hold plain ``QuantizedLinear`` layers that no longer communicate.
    public override func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedAllToShardedLinear(
            shardedWeight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode,
            group: group)
    }
}

/// Each member of the group applies part of the affine transformation and the
/// results are then aggregated.
///
/// Every member of the group ends up with the same result.
///
/// This is a ``Linear``, so it can replace one in a model.
open class ShardedToAllLinear: Linear {

    /// The group the input dimensions are sharded across.
    public let group: MLXDistributed.Group

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions, sharded across the group
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }

        self.group = group

        let scale = sqrt(1.0 / Float(inputDimensions))
        super.init(
            weight: MLXRandom.uniform(
                -scale ..< scale, [outputDimensions, inputDimensions / group.size]),
            bias: bias ? MLXRandom.uniform(-scale ..< scale, [outputDimensions]) : nil)
    }

    /// Hold parameters that are already this process' shard.
    init(shardedWeight weight: MLXArray, bias: MLXArray?, group: MLXDistributed.Group) {
        self.group = group
        super.init(weight: weight, bias: bias)
    }

    /// Create a sharded layer from an existing ``Linear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (_, inputDimensions) = other.shape
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }

        let parameters = Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, shardedToAllPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }

        self.init(shardedWeight: weight, bias: parameters["bias"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // each process holds part of the sum, and the bias belongs to the
        // whole, so it is added after the reduction
        var x = matmul(x, weight.T)
        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }

    /// Quantizing keeps the layer sharded.
    public override func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedShardedToAllLinear(
            shardedWeight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode,
            group: group)
    }
}

/// The quantized flavor of ``AllToShardedLinear``.
///
/// Like ``QuantizedLinear`` its parameters are frozen.
open class QuantizedAllToShardedLinear: QuantizedLinear {

    /// The group the output dimensions are sharded across.
    public let group: MLXDistributed.Group

    private let aggregateGradients: (MLXArray) -> MLXArray

    /// Quantize a weight that is already this process' shard.
    public init(
        shardedWeight weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, group: MLXDistributed.Group
    ) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Hold quantized parameters that are already this process' shard.
    init(
        shardedQuantizedWeight weight: MLXArray, bias: MLXArray?, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode,
        globalScale: MLXArray?, group: MLXDistributed.Group
    ) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, globalScale: globalScale)
        self.freeze()
    }

    /// Create a sharded layer from an existing ``QuantizedLinear``.
    ///
    /// The packed weight, `scales` and `biases` all carry one row per output,
    /// so they shard together along the output dimension.
    public convenience init(
        _ other: QuantizedLinear, segments: Segments = .count(1),
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (outputDimensions, _) = other.shape
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        let parameters = Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, allToShardedPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }
        guard let scales = parameters["scales"] else {
            throw ShardingError.missingParameter("scales")
        }

        self.init(
            shardedQuantizedWeight: weight, bias: parameters["bias"], scales: scales,
            biases: parameters["biases"], groupSize: other.groupSize, bits: other.bits,
            mode: other.mode, globalScale: parameters["global_scale"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // the quantized matmul, the global scale and the bias are all local to
        // this shard, so only the gradient aggregation is added
        super.callAsFunction(aggregateGradients(x))
    }
}

/// The quantized flavor of ``ShardedToAllLinear``.
///
/// Like ``QuantizedLinear`` its parameters are frozen.
open class QuantizedShardedToAllLinear: QuantizedLinear {

    /// The group the input dimensions are sharded across.
    public let group: MLXDistributed.Group

    /// Quantize a weight that is already this process' shard.
    public init(
        shardedWeight weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, group: MLXDistributed.Group
    ) {
        self.group = group
        super.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Hold quantized parameters that are already this process' shard.
    init(
        shardedQuantizedWeight weight: MLXArray, bias: MLXArray?, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode,
        globalScale: MLXArray?, group: MLXDistributed.Group
    ) {
        self.group = group
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, globalScale: globalScale)
        self.freeze()
    }

    /// Create a sharded layer from an existing ``QuantizedLinear``.
    ///
    /// This shards the input dimension, which is the packed one, so a shard has
    /// to hold whole quantization groups.
    public convenience init(
        _ other: QuantizedLinear, segments: Segments = .count(1),
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (_, inputDimensions) = other.shape
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }
        guard (inputDimensions / group.size) % other.groupSize == 0 else {
            throw ShardingError.quantizationGroup(
                inputDimensions: inputDimensions, groupSize: other.groupSize, across: group.size)
        }

        let parameters = Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, shardedToAllPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }
        guard let scales = parameters["scales"] else {
            throw ShardingError.missingParameter("scales")
        }

        self.init(
            shardedQuantizedWeight: weight, bias: parameters["bias"], scales: scales,
            biases: parameters["biases"], groupSize: other.groupSize, bits: other.bits,
            mode: other.mode, globalScale: parameters["global_scale"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // this cannot call super: the bias applies to the summed result, not to
        // this process' partial product
        var x = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
            bits: bits, mode: mode)
        x = applyNVFP4GlobalScale(x, globalScale: globalScale)
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
) throws -> ModuleParameters {
    let group = try group ?? MLXDistributed.initialize()
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
        return try averageGradients(
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
