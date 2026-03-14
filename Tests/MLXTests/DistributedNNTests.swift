// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

class DistributedNNTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Helper

    /// Get a size-1 distributed group for single-process testing.
    private func singletonGroup() -> DistributedGroup {
        MLXDistributed.`init`()!
    }

    // MARK: - AllToShardedLinear Init Tests

    func testAllToShardedLinearInit() {
        // VAL-NN-001: weight shape [outDims/N, inDims], bias shape [outDims/N]
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true, group: group)

        // N=1, so outDims/N = 64
        XCTAssertEqual(layer.weight.shape, [64, 128])
        XCTAssertNotNil(layer.bias)
        XCTAssertEqual(layer.bias!.shape, [64])
        XCTAssertEqual(layer.weight.dtype, .float32)
    }

    func testAllToShardedLinearInitNoBias() {
        // VAL-NN-016: layers work with bias=false
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false, group: group)

        XCTAssertEqual(layer.weight.shape, [64, 128])
        XCTAssertNil(layer.bias)
    }

    // MARK: - AllToShardedLinear Forward Tests

    func testAllToShardedLinearForwardBatch1() {
        // VAL-NN-002: output shape [batch, outDims/N] for input [batch, inDims]
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)

        let input = MLXRandom.uniform(0 ..< 1, [1, 32])
        let output = layer(input)
        XCTAssertEqual(output.shape, [1, 16])
    }

    func testAllToShardedLinearForwardBatch4() {
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)

        let input = MLXRandom.uniform(0 ..< 1, [4, 32])
        let output = layer(input)
        XCTAssertEqual(output.shape, [4, 16])
    }

    func testAllToShardedLinearForwardNoBias() {
        // VAL-NN-016: forward with bias=false
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: false, group: group)

        let input = MLXRandom.uniform(0 ..< 1, [2, 32])
        let output = layer(input)
        XCTAssertEqual(output.shape, [2, 16])
    }

    // MARK: - ShardedToAllLinear Init Tests

    func testShardedToAllLinearInit() {
        // VAL-NN-003: weight shape [outDims, inDims/N], bias shape [outDims]
        let group = singletonGroup()
        let layer = ShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true, group: group)

        // N=1, so inDims/N = 128
        XCTAssertEqual(layer.weight.shape, [64, 128])
        XCTAssertNotNil(layer.bias)
        XCTAssertEqual(layer.bias!.shape, [64])
        XCTAssertEqual(layer.weight.dtype, .float32)
    }

    func testShardedToAllLinearInitNoBias() {
        let group = singletonGroup()
        let layer = ShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false, group: group)

        XCTAssertEqual(layer.weight.shape, [64, 128])
        XCTAssertNil(layer.bias)
    }

    // MARK: - ShardedToAllLinear Forward Tests

    func testShardedToAllLinearForward() {
        // VAL-NN-004: output matches standard Linear within atol=1e-5
        let group = singletonGroup()

        // Create a standard linear and a sharded version with the same weights
        let linear = Linear(32, 16, bias: true)
        eval(linear)

        let sharded = ShardedToAllLinear.fromLinear(linear, group: group)
        eval(sharded)

        let input = MLXRandom.uniform(0 ..< 1, [4, 32])
        eval(input)

        let linearOutput = linear(input)
        let shardedOutput = sharded(input)

        // On a size-1 group, should match exactly
        assertEqual(shardedOutput, linearOutput, atol: 1e-5)
    }

    func testShardedToAllLinearForwardNoBias() {
        let group = singletonGroup()

        let linear = Linear(32, 16, bias: false)
        eval(linear)

        let sharded = ShardedToAllLinear.fromLinear(linear, group: group)
        eval(sharded)

        let input = MLXRandom.uniform(0 ..< 1, [2, 32])
        eval(input)

        let linearOutput = linear(input)
        let shardedOutput = sharded(input)

        assertEqual(shardedOutput, linearOutput, atol: 1e-5)
    }

    // MARK: - Module Protocol Compliance Tests

    func testAllToShardedLinearModuleProtocol() {
        // VAL-NN-015: parameters() returns weight (not group), children() excludes group
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)

        let params = layer.parameters()
        let flatParams = params.flattened()

        // Should have weight and bias
        let keys = Set(flatParams.map { $0.0 })
        XCTAssertTrue(keys.contains("weight"), "parameters() should contain weight")
        XCTAssertTrue(keys.contains("bias"), "parameters() should contain bias")
        XCTAssertFalse(keys.contains("group"), "parameters() should NOT contain group")

        // children() should be empty (no sub-modules)
        let children = layer.children()
        XCTAssertTrue(children.isEmpty, "children() should be empty (no sub-modules)")
    }

    func testShardedToAllLinearModuleProtocol() {
        let group = singletonGroup()
        let layer = ShardedToAllLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)

        let params = layer.parameters()
        let flatParams = params.flattened()

        let keys = Set(flatParams.map { $0.0 })
        XCTAssertTrue(keys.contains("weight"), "parameters() should contain weight")
        XCTAssertTrue(keys.contains("bias"), "parameters() should contain bias")
        XCTAssertFalse(keys.contains("group"), "parameters() should NOT contain group")

        let children = layer.children()
        XCTAssertTrue(children.isEmpty, "children() should be empty (no sub-modules)")
    }

    func testNoBiasModuleProtocol() {
        // Parameters should only contain weight when bias=false
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: false, group: group)

        let params = layer.parameters()
        let flatParams = params.flattened()

        let keys = Set(flatParams.map { $0.0 })
        XCTAssertTrue(keys.contains("weight"))
        XCTAssertFalse(keys.contains("bias"), "parameters() should not contain bias when bias=false")
        XCTAssertFalse(keys.contains("group"))
    }

    func testFreezeUnfreeze() {
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)

        // Initially all parameters are trainable
        let trainable = layer.trainableParameters().flattened()
        XCTAssertFalse(trainable.isEmpty)

        // Freeze
        layer.freeze()
        let frozenTrainable = layer.trainableParameters().flattened()
        XCTAssertTrue(frozenTrainable.isEmpty, "After freeze, no trainable parameters expected")

        // Unfreeze
        layer.unfreeze()
        let unfrozenTrainable = layer.trainableParameters().flattened()
        XCTAssertFalse(
            unfrozenTrainable.isEmpty, "After unfreeze, trainable parameters expected")
    }

    func testUpdateParameters() {
        // VAL-NN-015: update(parameters:) updates weights used in next forward pass
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)
        eval(layer)

        let input = MLXRandom.uniform(0 ..< 1, [1, 32])
        eval(input)

        let output1 = layer(input)
        eval(output1)

        // Double all parameters
        layer.update(parameters: layer.mapParameters { $0 * 2 })

        let output2 = layer(input)
        eval(output2)

        // Output should be different after update
        let isClose = output1.allClose(output2, atol: 1e-5).item(Bool.self)
        XCTAssertFalse(isClose, "Output should differ after parameter update")
    }

    // MARK: - fromLinear Conversion Tests

    func testAllToShardedFromLinear() {
        // VAL-NN-009: shardLinear -> AllToShardedLinear, weights identical for size-1 group
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = AllToShardedLinear.fromLinear(linear, group: group)
        eval(sharded)

        // For size-1 group, sharded weights should be identical to original
        assertEqual(sharded.weight, linear.weight, atol: 1e-5)
        XCTAssertNotNil(sharded.bias)
        assertEqual(sharded.bias!, linear.bias!, atol: 1e-5)
    }

    func testShardedToAllFromLinear() {
        // VAL-NN-010: shardLinear -> ShardedToAllLinear, weights identical for size-1 group
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = ShardedToAllLinear.fromLinear(linear, group: group)
        eval(sharded)

        // For size-1 group, sharded weights should be identical to original
        assertEqual(sharded.weight, linear.weight, atol: 1e-5)
        XCTAssertNotNil(sharded.bias)
        assertEqual(sharded.bias!, linear.bias!, atol: 1e-5)
    }

    func testFromLinearNoBias() {
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: false)
        eval(linear)

        let sharded = AllToShardedLinear.fromLinear(linear, group: group)
        eval(sharded)

        assertEqual(sharded.weight, linear.weight, atol: 1e-5)
        XCTAssertNil(sharded.bias)
    }

    // MARK: - Rectangular Matrix Tests

    func testRectangularMatrixAllToSharded() {
        // VAL-NN-019: non-square Linear layers
        let group = singletonGroup()

        // Wide: 512 -> 128
        let wide = Linear(512, 128, bias: true)
        eval(wide)
        let shardedWide = AllToShardedLinear.fromLinear(wide, group: group)
        eval(shardedWide)
        XCTAssertEqual(shardedWide.weight.shape, [128, 512])

        // Tall: 128 -> 512
        let tall = Linear(128, 512, bias: true)
        eval(tall)
        let shardedTall = AllToShardedLinear.fromLinear(tall, group: group)
        eval(shardedTall)
        XCTAssertEqual(shardedTall.weight.shape, [512, 128])
    }

    func testRectangularMatrixShardedToAll() {
        let group = singletonGroup()

        let wide = Linear(512, 128, bias: true)
        eval(wide)
        let shardedWide = ShardedToAllLinear.fromLinear(wide, group: group)
        eval(shardedWide)
        XCTAssertEqual(shardedWide.weight.shape, [128, 512])

        let tall = Linear(128, 512, bias: true)
        eval(tall)
        let shardedTall = ShardedToAllLinear.fromLinear(tall, group: group)
        eval(shardedTall)
        XCTAssertEqual(shardedTall.weight.shape, [512, 128])
    }

    // MARK: - sumGradients Tests

    func testSumGradientsForwardIdentity() {
        // VAL-NN-013: sumGradients is identity in forward pass
        let group = singletonGroup()
        let fn = sumGradients(group: group)

        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let output = fn(input)

        assertEqual(output, input)
    }

    // MARK: - Gradient Flow Tests

    func testGradientFlowThroughAllToShardedLinear() {
        // VAL-CROSS-004: grad of a scalar loss through AllToShardedLinear
        // produces non-zero gradients
        let group = singletonGroup()
        let layer = AllToShardedLinear(
            inputDimensions: 8, outputDimensions: 4, bias: true, group: group)
        eval(layer)

        let input = MLXRandom.uniform(0 ..< 1, [1, 8])
        eval(input)

        // Compute gradient of sum(layer(x)) w.r.t. x
        let gradFn = grad { (x: MLXArray) -> MLXArray in
            layer(x).sum()
        }

        let g = gradFn(input)
        eval(g)

        // Gradient should be non-zero
        XCTAssertEqual(g.shape, input.shape)
        let absSum = abs(g).sum().item(Float.self)
        XCTAssertGreaterThan(absSum, 0.0, "Gradient should be non-zero")
    }

    // MARK: - ShardedToAllLinear vs Linear Comparison

    func testShardedToAllMatchesLinear() {
        // VAL-CROSS-002: ShardedToAllLinear produces same result as Linear
        let group = singletonGroup()

        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = ShardedToAllLinear.fromLinear(linear, group: group)
        eval(sharded)

        // Test with multiple batch sizes
        for batchSize in [1, 4, 8] {
            let input = MLXRandom.uniform(0 ..< 1, [batchSize, 64])
            eval(input)

            let linearOutput = linear(input)
            let shardedOutput = sharded(input)

            assertEqual(
                shardedOutput, linearOutput, atol: 1e-5)
        }
    }

    func testAllToShardedMatchesLinear() {
        // On size-1 group, AllToShardedLinear should also match Linear
        let group = singletonGroup()

        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = AllToShardedLinear.fromLinear(linear, group: group)
        eval(sharded)

        let input = MLXRandom.uniform(0 ..< 1, [4, 64])
        eval(input)

        let linearOutput = linear(input)
        let shardedOutput = sharded(input)

        assertEqual(shardedOutput, linearOutput, atol: 1e-5)
    }
}
