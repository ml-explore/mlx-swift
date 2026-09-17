// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

// Ports of the layer tests from the Python distributed tests
// (`python/tests/mlx_distributed_tests.py`).  Python runs them for every group
// size under a launcher; each body here runs twice, once in a single process
// where the sharding degenerates but the layers are still built and run, and
// once across real ranks where every rank holds a different slice.

/// Port of `test_shard_linear`.
///
/// - Parameter world: the group to shard across
func shardLinearBody(world: MLXDistributed.Group) throws {
    MLXRandom.seed(0xF0F0_F0F0)

    let lower = world.rank * 1024 / world.size
    let upper = (world.rank + 1) * 1024 / world.size

    let x = MLXRandom.normal([4, 1024])
    let linear = Linear(1024, 1024, bias: true)

    // MARK: - float

    let sharded1 = try shardLinear(linear, sharding: .allToSharded, group: world)
    let sharded2 = try shardLinear(linear, sharding: .shardedToAll, group: world)

    let y = linear(x)
    let y1 = sharded1(x)
    let y2 = sharded2(x[lower ..< upper, axis: 1])
    try checkedEval(y, y1, y2)

    XCTAssertTrue(
        y.allClose(y2, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "sharded-to-all must reproduce the unsharded output")
    XCTAssertTrue(
        y[lower ..< upper, axis: 1].allClose(y1, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "all-to-sharded must reproduce this rank's slice of the output")

    // MARK: - quantized

    // Python: qlin = lin.to_quantized()
    let quantized = QuantizedLinear(linear)
    let qSharded1 = try shardLinear(quantized, sharding: .allToSharded, group: world)
    let qSharded2 = try shardLinear(quantized, sharding: .shardedToAll, group: world)

    XCTAssertTrue(
        qSharded1 is QuantizedAllToShardedLinear,
        "a QuantizedLinear must shard into the quantized layer, not the float one")
    XCTAssertTrue(qSharded2 is QuantizedShardedToAllLinear)

    let qy = quantized(x)
    let qy1 = qSharded1(x)
    let qy2 = qSharded2(x[lower ..< upper, axis: 1])
    try checkedEval(qy, qy1, qy2)

    XCTAssertTrue(
        qy.allClose(qy2, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "quantized sharded-to-all must reproduce the unsharded output")
    XCTAssertTrue(
        qy[lower ..< upper, axis: 1].allClose(qy1, rtol: 1e-5, atol: 1e-8).item(Bool.self),
        "quantized all-to-sharded must reproduce this rank's slice")

    // MARK: - a non affine mode, which carries no quantization biases

    // Python: lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    let mxfp8 = QuantizedLinear(linear, groupSize: 32, bits: 8, mode: .mxfp8)
    XCTAssertEqual(mxfp8.mode, .mxfp8)
    XCTAssertNil(mxfp8.biases, "mxfp8 quantization has no biases")

    let mxSharded1 = try XCTUnwrap(
        try shardLinear(mxfp8, sharding: .allToSharded, group: world) as? QuantizedLinear)
    let mxSharded2 = try XCTUnwrap(
        try shardLinear(mxfp8, sharding: .shardedToAll, group: world) as? QuantizedLinear)

    XCTAssertEqual(mxSharded1.mode, .mxfp8, "the mode must survive sharding")
    XCTAssertEqual(mxSharded2.mode, .mxfp8, "the mode must survive sharding")
    XCTAssertNil(mxSharded1.biases)
    XCTAssertNil(mxSharded2.biases)

    let my = mxfp8(x)
    let my1 = mxSharded1(x)
    let my2 = mxSharded2(x[lower ..< upper, axis: 1])
    try checkedEval(my, my1, my2)

    XCTAssertTrue(
        my.allClose(my2, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "mxfp8 sharded-to-all must reproduce the unsharded output")
    XCTAssertTrue(
        my[lower ..< upper, axis: 1].allClose(my1, rtol: 1e-5, atol: 1e-8).item(Bool.self),
        "mxfp8 all-to-sharded must reproduce this rank's slice")

    // MARK: - quantizing a sharded layer keeps it sharded

    let quantizedShard = sharded1.toQuantized(groupSize: 64, bits: 4, mode: .affine)
    XCTAssertTrue(
        quantizedShard is QuantizedAllToShardedLinear,
        "quantizing a sharded layer must not drop the communication")

    // MARK: - backward

    try shardLinearBackward(world: world)
}

/// The gradients of a sharded model must match the slice of the unsharded
/// model's gradients that this rank owns.
private func shardLinearBackward(world: MLXDistributed.Group) throws {
    MLXRandom.seed(0xF0F0_F0F0)

    let layers = (0 ..< 4).map { _ in Linear(128, 128) }
    let model = Sequential(layers: layers[0], layers[1], layers[2], layers[3])
    let sharded = Sequential(
        layers: try shardLinear(layers[0], sharding: .allToSharded, group: world),
        try shardLinear(layers[1], sharding: .shardedToAll, group: world),
        try shardLinear(layers[2], sharding: .allToSharded, group: world),
        try shardLinear(layers[3], sharding: .shardedToAll, group: world))

    func loss(_ model: Sequential, _ x: MLXArray, _ y: MLXArray) -> MLXArray {
        (model(x) * y).sum()
    }

    let x = MLXRandom.normal([4, 128])
    let y = MLXRandom.normal([4, 128])

    let (l1, g1) = valueAndGrad(model: model, loss)(model, x, y)
    let (l2, g2) = valueAndGrad(model: sharded, loss)(sharded, x, y)
    try checkedEval(l1, g1, l2, g2)

    let lower = world.rank * 128 / world.size
    let upper = (world.rank + 1) * 128 / world.size
    let full = Dictionary(uniqueKeysWithValues: g1.flattened())
    let shard = Dictionary(uniqueKeysWithValues: g2.flattened())

    XCTAssertTrue(l1.allClose(l2, rtol: 1e-4, atol: 1e-6).item(Bool.self), "loss mismatch")

    // all-to-sharded splits the output dimension, so rows of the weight
    // gradient and the bias gradient
    for layer in [0, 2] {
        assertGradientShard(full, shard, "layers.\(layer).weight", lower ..< upper, axis: 0)
        assertGradientShard(full, shard, "layers.\(layer).bias", lower ..< upper, axis: 0)
    }

    // sharded-to-all splits the input dimension -- columns of the weight
    // gradient -- and leaves the bias whole
    for layer in [1, 3] {
        assertGradientShard(full, shard, "layers.\(layer).weight", lower ..< upper, axis: 1)
        assertGradientWhole(full, shard, "layers.\(layer).bias")
    }
}

/// Port of `test_shard_predicate`.
///
/// A module that is not a ``Linear`` is sharded by handing ``shardInPlace`` a
/// predicate.  The module aggregates where the sharding requires it, which the
/// predicate cannot do on its own.
func shardPredicateBody(world: MLXDistributed.Group) throws {
    MLXRandom.seed(0xF0F0_F0F0)

    // even layers shard their output channels; odd layers shard their input
    // channels and keep the bias, which applies to the summed result
    let sharding: ShardingPredicate = { path, _ in
        let parts = path.split(separator: ".")
        guard parts.count > 1, let layer = Int(parts[1]) else { return nil }

        if layer % 2 == 0 {
            return (0, .count(1))
        }
        return parts.last == "bias" ? nil : (-1, .count(1))
    }

    let model = Sequential(
        layers: AggregatingConv(3, 128), AggregatingConv(128, 128),
        AggregatingConv(128, 128), AggregatingConv(128, 3))
    let sharded = Sequential(
        layers: AggregatingConv(3, 128),
        AggregatingConv(128, 128, aggregate: world),
        AggregatingConv(128, 128),
        AggregatingConv(128, 3, aggregate: world))

    _ = sharded.update(parameters: model.parameters())
    try shardInPlace(sharded, predicate: sharding, group: world)

    let x = MLXRandom.normal([4, 16, 16, 3])
    let y1 = model(x)
    let y2 = sharded(x)
    try checkedEval(y1, y2)

    XCTAssertTrue(
        y1.allClose(y2, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "a predicate sharded model must reproduce the unsharded output")
}

/// Sharding swaps the layers of a model in place, the way quantization does.
///
/// Models declare their projections as `Linear`, and a model's shard method
/// installs the sharded layers with `update(modules:)`.  That only works
/// because the sharded layers are `Linear` subclasses: a sharded layer that
/// was not a `Linear` could not be installed at all.
func moduleSubstitutionBody(world: MLXDistributed.Group) throws {
    MLXRandom.seed(0xF0F0_F0F0)
    let x = MLXRandom.normal([4, 64])

    // the check has to be able to fail: a module that is not a Linear is refused
    let control = ProjectionBlock(64, 256)
    let notLinear: [(String, Module)] = [("up", Identity()), ("down", control.down)]
    XCTAssertThrowsError(
        try control.update(modules: ModuleChildren.unflattened(notLinear), verify: .all)
    ) { error in
        XCTAssertTrue(error is UpdateError, "unexpected error: \(error)")
    }

    // a float model, and one quantized first like a loaded quantized model
    for quantized in [false, true] {
        let block = ProjectionBlock(64, 256)
        if quantized {
            quantize(model: block)
        }
        let expected = block(x)
        try checkedEval(expected)

        // what a model's shard method does
        let up = try shardLinear(block.up, sharding: .allToSharded, group: world)
        let down = try shardLinear(block.down, sharding: .shardedToAll, group: world)
        let sharded: [(String, Module)] = [("up", up), ("down", down)]
        try block.update(modules: ModuleChildren.unflattened(sharded), verify: .all)

        if quantized {
            XCTAssertTrue(block.up is QuantizedAllToShardedLinear, "\(type(of: block.up))")
            XCTAssertTrue(block.down is QuantizedShardedToAllLinear, "\(type(of: block.down))")
        } else {
            XCTAssertTrue(block.up is AllToShardedLinear, "\(type(of: block.up))")
            XCTAssertTrue(block.down is ShardedToAllLinear, "\(type(of: block.down))")
        }

        // every process computes the output of the whole block
        let y = block(x)
        try checkedEval(y)
        XCTAssertTrue(
            expected.allClose(y, rtol: 1e-4, atol: 1e-6).item(Bool.self),
            "the sharded block must reproduce the unsharded output (quantized: \(quantized))")
    }
}

/// Port of `test_donation`.
///
/// A collective donates its result to the operation that consumes it, so
/// summing and then scaling must not cost more memory than summing alone.
///
/// This one has no single process half: in a group of one MLX returns the
/// input unchanged, so there is no buffer to donate and the input is still
/// live.  Python only ever runs it under a launcher for the same reason.
func donationBody(world: MLXDistributed.Group) throws {
    let cpu = Stream.defaultStream(.cpu)

    let x = MLXRandom.normal([1024])
    try checkedEval(x)
    cpu.synchronize()
    GPU.resetPeakMemory()

    let scale = MLXArray(2.0)

    // Python rebinds one name, so the first result is released before the
    // second is evaluated.  Holding on to it here would add its buffer to the
    // peak and make the comparison below fail for the wrong reason.
    let allSumOnly: Int
    do {
        let sum = MLXDistributed.allSum(x, group: world)
        try checkedEval(sum)
        cpu.synchronize()
        allSumOnly = Memory.peakMemory
    }

    let scaled = MLXDistributed.allSum(x, group: world) * scale
    try checkedEval(scaled)
    cpu.synchronize()
    let allSumWithBinary = Memory.peakMemory

    // the instrument has to move at all, or the comparison proves nothing
    XCTAssertGreaterThan(allSumOnly, 0, "peak memory is not tracked for this stream")
    XCTAssertEqual(
        allSumOnly, allSumWithBinary,
        "the multiply must donate the buffer the all sum produced")
}

/// Port of `test_quantized_sharded_linear_construction` from the Python nn
/// tests.
///
/// Every bit width packs a different number of values into each `uint32`, so
/// the sharded layers have to come out with the shapes the layer they were
/// built from would have, divided along the sharded axis.
func quantizedShardedConstructionBody(world: MLXDistributed.Group) throws {
    for bits in [2, 3, 4, 5, 6, 8] {
        let quantized = QuantizedLinear(Linear(1536, 1024), groupSize: 64, bits: bits)
        let allToSharded = try QuantizedAllToShardedLinear(quantized, group: world)
        let shardedToAll = try QuantizedShardedToAllLinear(quantized, group: world)

        // rows are outputs, columns are the packed inputs
        XCTAssertEqual(
            allToSharded.weight.dim(0), quantized.weight.dim(0) / world.size, "bits \(bits)")
        XCTAssertEqual(allToSharded.weight.dim(1), quantized.weight.dim(1), "bits \(bits)")

        XCTAssertEqual(shardedToAll.weight.dim(0), quantized.weight.dim(0), "bits \(bits)")
        XCTAssertEqual(
            shardedToAll.weight.dim(1), quantized.weight.dim(1) / world.size, "bits \(bits)")

        // and the scales have to cover exactly the inputs the weight holds
        for layer in [allToSharded, shardedToAll] {
            XCTAssertEqual(
                layer.weight.dim(1) * 32 / bits, layer.scales.dim(1) * layer.groupSize,
                "bits \(bits)")
        }
    }
}

/// Cases the Python tests do not cover, each one a bug this port had.
func shardingEdgeCasesBody(world: MLXDistributed.Group) throws {
    // A sharded-to-all layer adds its bias after the reduction, so every
    // process has to hold the same bias.  Seed the processes differently:
    // nothing keeps their generators in step -- they may have done different
    // work before this -- and the bias must not depend on that.
    MLXRandom.seed(UInt64(world.rank + 1))
    let layer = try ShardedToAllLinear(8 * world.size, 4, group: world)
    let bias = try XCTUnwrap(layer.bias)
    let gathered = MLXDistributed.allGather(bias, group: world)
    try checkedEval(gathered)

    let width = bias.dim(0)
    for rank in 1 ..< world.size {
        XCTAssertTrue(
            gathered[0 ..< width].allClose(gathered[rank * width ..< (rank + 1) * width])
                .item(Bool.self),
            "rank \(rank) holds a different bias than rank 0")
    }

    // A dimension can divide by the group size while one of its segments does
    // not.  That has to be reported rather than trap inside split(parts:).
    if world.size > 1 {
        XCTAssertThrowsError(
            try shardLinear(
                Linear(8 * world.size, 8), sharding: .shardedToAll,
                segments: .indices([3]), group: world)
        ) { error in
            XCTAssertEqual(
                error as? ShardingError,
                .indivisible(dimension: "segment of weight", of: 3, across: world.size))
        }
    }

    // Segments that cannot split a dimension at all are reported too, in a
    // group of one as well.  split(parts:) would call fatalError.
    XCTAssertThrowsError(
        try shardLinear(
            Linear(8 * world.size, 8), sharding: .shardedToAll, segments: .count(3),
            group: world)
    ) { error in
        XCTAssertEqual(
            error as? ShardingError, .invalidSegments(.count(3), dimension: 8 * world.size))
    }

    // A QuantizedLinear is a Linear, so the float layers accept one as far as
    // the compiler is concerned, but they would keep its packed weight as a
    // float weight and drop its scales.
    let quantized = QuantizedLinear(Linear(64, 64))
    XCTAssertThrowsError(try AllToShardedLinear(quantized, group: world)) { error in
        XCTAssertEqual(error as? ShardingError, .quantizedLayer)
    }
    XCTAssertThrowsError(try ShardedToAllLinear(quantized, group: world)) { error in
        XCTAssertEqual(error as? ShardingError, .quantizedLayer)
    }

    // A quantized layer packs its inputs in the weight and groups them in the
    // scales, so a boundary between segments lands at a different position in
    // each.  Scaled through a Double, 3840 of 5632 inputs came to
    // 479.99999999999994 packed words and 59.99999999999999 groups, which
    // truncate to the wrong positions.  The shard has to reproduce the layer.
    MLXRandom.seed(0xF0F0_F0F0)
    let inputs = 5632
    let boundary = 3840
    let wide = QuantizedLinear(Linear(inputs, 8))
    let segmented = try QuantizedShardedToAllLinear(
        wide, segments: .indices([boundary]), group: world)

    // this rank's part of each segment
    let (rank, size) = (world.rank, world.size)
    let first = (rank * boundary / size) ..< ((rank + 1) * boundary / size)
    let second =
        (boundary + rank * (inputs - boundary) / size)
        ..< (boundary + (rank + 1) * (inputs - boundary) / size)

    let x = MLXRandom.normal([2, inputs])
    let y = wide(x)
    let ySharded = segmented(concatenated([x[first, axis: 1], x[second, axis: 1]], axis: 1))
    try checkedEval(y, ySharded)

    XCTAssertTrue(
        y.allClose(ySharded, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "a segmented quantized shard must reproduce the unsharded output")

    // a boundary inside a quantization group cannot be sharded at all
    XCTAssertThrowsError(
        try QuantizedShardedToAllLinear(wide, segments: .indices([272]), group: world)
    ) { error in
        XCTAssertEqual(
            error as? ShardingError,
            .quantizationGroup(inputDimensions: 272, groupSize: 64, across: world.size))
    }
}

/// Port of `test_average_gradients`.
///
/// Python counts the `all_sum` calls by replacing `mx.distributed.all_sum`,
/// which Swift cannot do, so this counts the batches `averageGradients` forms
/// instead.  Every batching has to produce the same average, and the average
/// has to include every rank.
func averageGradientsBody(world: MLXDistributed.Group) throws {
    // Ten gradients of ten float32 values, 40 bytes each as in Python, so the
    // limits below form different batches.  Each rank contributes its own
    // rank + 1, so the average is (size + 1) / 2 times the gradient's own
    // factor.  The values differ per gradient and half are matrices on
    // purpose: batching concatenates them, reduces once and splits the result
    // again, and identical or flat arrays would hide a bad split.
    let mean = Float(world.size + 1) / 2
    let shapes = (0 ..< 10).map { $0.isMultiple(of: 2) ? [10] : [2, 5] }
    let gradients = ModuleParameters.unflattened(
        (0 ..< 10).map {
            ("g\($0)", MLXArray.ones(shapes[$0]) * Float(($0 + 1) * (world.rank + 1)))
        })
    let expected = Dictionary(
        uniqueKeysWithValues: (0 ..< 10).map {
            ("g\($0)", MLXArray.ones(shapes[$0]) * (Float($0 + 1) * mean))
        })

    // Python makes one all_sum call for 32MiB, two for 4 * 50 bytes and ten
    // without batching.  A limit of one byte batches every gradient alone.
    let bytes = gradients.flattened().map { $0.1.nbytes }
    XCTAssertEqual(groupBySize(bytes, limit: 32 * 1024 * 1024), [Array(0 ..< 10)])
    XCTAssertEqual(groupBySize(bytes, limit: 4 * 50), [Array(0 ..< 5), Array(5 ..< 10)])
    XCTAssertEqual(groupBySize(bytes, limit: 1), (0 ..< 10).map { [$0] })

    for allReduceSize in [32 * 1024 * 1024, 4 * 50, 1, 0] {
        let averaged = try averageGradients(
            gradients, group: world, allReduceSize: allReduceSize)
        let flat = averaged.flattened()
        try checkedEval(flat.map { $0.1 })

        XCTAssertEqual(flat.count, 10, "allReduceSize \(allReduceSize)")
        for (key, value) in flat {
            let want = try XCTUnwrap(expected[key])
            XCTAssertEqual(value.shape, want.shape, "\(key) with allReduceSize \(allReduceSize)")
            XCTAssertTrue(
                value.allClose(want).item(Bool.self),
                "\(key) with allReduceSize \(allReduceSize)")
        }
    }

    // arrays of different types cannot be concatenated, so they fall back to
    // one reduction per array and keep their dtype
    let mixed = try averageGradients(
        ModuleParameters.unflattened([
            ("wide", MLXArray.ones([4]) * Float(world.rank + 1)),
            ("narrow", (MLXArray.ones([4]) * Float(world.rank + 1)).asType(.float16)),
        ]), group: world)
    // evaluate in the flattened order, which is sorted by key: a Dictionary
    // iterates in a per process order, and evaluating the collectives in
    // different orders makes the ranks exchange mismatched buffers
    let flat = mixed.flattened()
    try checkedEval(flat.map { $0.1 })

    let flatMixed = Dictionary(uniqueKeysWithValues: flat)
    XCTAssertEqual(flatMixed["narrow"]?.dtype, .float16, "the dtype must survive")
    for key in ["wide", "narrow"] {
        let averaged = try XCTUnwrap(flatMixed[key])
        XCTAssertTrue(
            averaged.asType(.float32)
                .allClose(MLXArray.ones([4]) * (Float(world.size + 1) / 2), rtol: 1e-2)
                .item(Bool.self), key)
    }
}

/// Port of `test_clip_grad_norm_sharded`.
func clipGradNormShardedBody(world: MLXDistributed.Group) throws {
    let value: Float = 3
    let gradients = ModuleParameters.unflattened([
        ("a", MLXArray.ones([4, 3]) * value),
        ("b", MLXArray.ones([5]) * value),
    ])

    // every rank holds the same number of elements, so the global norm counts
    // all of them
    let localCount = 4 * 3 + 5
    let expectedNorm = (Float(world.size * localCount)).squareRoot() * value

    // a limit far above the norm leaves the shard alone
    let (unclipped, norm) = try clipGradNormSharded(
        gradients: gradients, maxNorm: 1e9, group: world)
    try checkedEval(norm, unclipped.flattened().map { $0.1 })

    XCTAssertEqual(norm.item(Float.self), expectedNorm, accuracy: expectedNorm * 1e-4)
    for (key, clipped) in unclipped.flattened() {
        XCTAssertTrue(
            clipped.allClose(MLXArray.ones(clipped.shape) * value).item(Bool.self), key)
    }

    // below it every gradient is scaled by maxNorm / norm
    let maxNorm: Float = 1
    let (clipped, _) = try clipGradNormSharded(
        gradients: gradients, maxNorm: maxNorm, group: world)
    try checkedEval(clipped.flattened().map { $0.1 })

    let scale = maxNorm / (expectedNorm + 1e-6)
    for (key, scaled) in clipped.flattened() {
        XCTAssertTrue(
            scaled.allClose(MLXArray.ones(scaled.shape) * value * scale, rtol: 1e-4)
                .item(Bool.self), key)
    }
}

/// A block that declares its projections as `Linear`, the way models do.
private class ProjectionBlock: Module, UnaryLayer {

    @ModuleInfo(key: "up") var up: Linear
    @ModuleInfo(key: "down") var down: Linear

    init(_ dimensions: Int, _ hiddenDimensions: Int) {
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(relu(up(x)))
    }
}

/// A convolution that can sum its output across the group, so that a layer fed
/// sharded input channels produces the whole result.
private class AggregatingConv: Module, UnaryLayer {

    @ModuleInfo(key: "conv") var conv: Conv2d

    let aggregate: MLXDistributed.Group?

    init(
        _ inputChannels: Int, _ outputChannels: Int, kernelSize: Int = 3,
        aggregate: MLXDistributed.Group? = nil
    ) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize))
        self.aggregate = aggregate
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        guard let aggregate else { return y }
        return MLXDistributed.allSum(y, group: aggregate)
    }
}

private func assertGradientShard(
    _ full: [String: MLXArray], _ shard: [String: MLXArray], _ key: String,
    _ range: Range<Int>, axis: Int, file: StaticString = #filePath, line: UInt = #line
) {
    guard let expected = full[key], let actual = shard[key] else {
        return XCTFail("missing gradient for \(key)", file: file, line: line)
    }
    XCTAssertTrue(
        expected[range, axis: axis].allClose(actual, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "\(key) gradient shard mismatch", file: file, line: line)
}

private func assertGradientWhole(
    _ full: [String: MLXArray], _ shard: [String: MLXArray], _ key: String,
    file: StaticString = #filePath, line: UInt = #line
) {
    guard let expected = full[key], let actual = shard[key] else {
        return XCTFail("missing gradient for \(key)", file: file, line: line)
    }
    XCTAssertTrue(
        expected.allClose(actual, rtol: 1e-4, atol: 1e-6).item(Bool.self),
        "\(key) gradient mismatch", file: file, line: line)
}

/// The single process half: sharding is the identity in a group of size one,
/// but the layers are still built and run.
///
/// That is enough to catch a layer that mishandles its input -- a
/// ``QuantizedLinear`` sharded as if it were a float layer, for example --
/// which is why this runs in CI without a launcher.
class DistributedNNTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    override func setUpWithError() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["MLX_TEST_DISTRIBUTED"] == "1",
            "The multi process run covers this; see DistributedNNRingTests.")
    }

    func testShardLinear() throws {
        try shardLinearBody(world: try MLXDistributed.initialize())
    }

    func testShardPredicate() throws {
        try shardPredicateBody(world: try MLXDistributed.initialize())
    }

    func testModuleSubstitution() throws {
        try moduleSubstitutionBody(world: try MLXDistributed.initialize())
    }

    func testQuantizedShardedConstruction() throws {
        try quantizedShardedConstructionBody(world: try MLXDistributed.initialize())
    }

    func testShardingEdgeCases() throws {
        try shardingEdgeCasesBody(world: try MLXDistributed.initialize())
    }

    func testAverageGradients() throws {
        try averageGradientsBody(world: try MLXDistributed.initialize())
    }

    func testClipGradNormSharded() throws {
        try clipGradNormShardedBody(world: try MLXDistributed.initialize())
    }
}

/// The multi process half, where every rank holds a different slice.
///
/// Skipped unless `MLX_TEST_DISTRIBUTED=1`:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 xcrun xctest -XCTest DistributedNNRingTests \
///     .../MLXTests.xctest
/// ```
///
/// Like ``DistributedRingTests`` this has a single test method, since a
/// process can only join one ring.  It runs every ported body in turn, the way
/// the Python tests all share the group their launcher formed.
class DistributedNNRingTests: XCTestCase {

    static let testName = "DistributedNNRingTests/testShardedLayers"

    /// Four ranks: the dimensions the Python tests use, 1024 and 128, divide by
    /// four, and a shard of 1024 inputs still holds whole quantization groups.
    static let rankCount = 4

    override class func setUp() {
        setDefaultDevice()
    }

    func testShardedLayers() throws {
        try DistributedHarness.run(ranks: Self.rankCount, testName: Self.testName) { group in
            try shardLinearBody(world: group)
            try shardPredicateBody(world: group)
            try moduleSubstitutionBody(world: group)
            try donationBody(world: group)
            try quantizedShardedConstructionBody(world: group)
            try shardingEdgeCasesBody(world: group)
            try averageGradientsBody(world: group)
            try clipGradNormShardedBody(world: group)
        }
    }
}
