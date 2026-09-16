// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

/// Port of `test_shard_linear` from the Python distributed tests
/// (`python/tests/mlx_distributed_tests.py`).
///
/// Python runs it for every group size under a launcher.  This runs the same
/// body twice: once in a single process, where the sharding degenerates but
/// construction and the forward pass are still exercised, and once across real
/// ranks, where the slices actually differ.
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
        y.allClose(y2, rtol: 1e-2, atol: 1e-2).item(Bool.self),
        "sharded-to-all must reproduce the unsharded output")
    XCTAssertTrue(
        y[lower ..< upper, axis: 1].allClose(y1, rtol: 1e-2, atol: 1e-2).item(Bool.self),
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
        qy.allClose(qy2, rtol: 1e-2, atol: 1e-2).item(Bool.self),
        "quantized sharded-to-all must reproduce the unsharded output")
    XCTAssertTrue(
        qy[lower ..< upper, axis: 1].allClose(qy1, rtol: 1e-2, atol: 1e-2).item(Bool.self),
        "quantized all-to-sharded must reproduce this rank's slice")

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
class DistributedShardLinearTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testShardLinear() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["MLX_TEST_DISTRIBUTED"] == "1",
            "The multi process run covers this; see DistributedShardLinearRingTests.")

        try shardLinearBody(world: try MLXDistributed.initialize())
    }
}

/// The multi process half, where every rank holds a different slice.
///
/// Skipped unless `MLX_TEST_DISTRIBUTED=1`:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 xcrun xctest -XCTest DistributedShardLinearRingTests \
///     .../MLXTests.xctest
/// ```
class DistributedShardLinearRingTests: XCTestCase {

    static let testName = "DistributedShardLinearRingTests/testShardLinear"

    /// Four ranks: the dimensions the Python test uses, 1024 and 128, divide by
    /// four, and a shard of 1024 inputs still holds whole quantization groups.
    static let rankCount = 4

    override class func setUp() {
        setDefaultDevice()
    }

    func testShardLinear() throws {
        try DistributedHarness.run(ranks: Self.rankCount, testName: Self.testName) { group in
            try shardLinearBody(world: group)
        }
    }
}
