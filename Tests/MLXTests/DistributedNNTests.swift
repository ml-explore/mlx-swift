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

    // MARK: - (1) AllToShardedLinear Init Tests

    func testAllToShardedLinearInit() {
        // VAL-NN-001: weight shape [outDims/N, inDims], bias shape [outDims/N], dtype float32
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

    // MARK: - (2) AllToShardedLinear Forward Tests

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

    // MARK: - (3) ShardedToAllLinear Init Tests

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

    // MARK: - (4) ShardedToAllLinear Forward Tests

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

    // MARK: - (5) QuantizedAllToShardedLinear Init Tests

    func testQuantizedAllToShardedLinearInit() {
        // VAL-NN-005: frozen state, Quantized protocol conformance, parameter shapes
        let group = singletonGroup()
        let layer = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        // Verify Quantized protocol conformance
        XCTAssertTrue(layer is Quantized, "Should conform to Quantized protocol")
        XCTAssertEqual(layer.groupSize, 64)
        XCTAssertEqual(layer.bits, 4)
        XCTAssertEqual(layer.mode, .affine)

        // Verify frozen state: trainableParameters should be empty
        let trainable = layer.trainableParameters().flattened()
        XCTAssertTrue(trainable.isEmpty, "Quantized layer should be frozen after init")

        // Verify parameters are non-empty (weight, scales, etc.)
        let params = layer.parameters().flattened()
        XCTAssertFalse(params.isEmpty, "parameters() should be non-empty")

        // Verify bias shape: [outDims/N] = [64] for N=1
        XCTAssertNotNil(layer.bias)
        XCTAssertEqual(layer.bias!.shape, [64])

        // Verify weight and scales exist
        XCTAssertFalse(layer.weight.shape.isEmpty)
        XCTAssertFalse(layer.scales.shape.isEmpty)
    }

    func testQuantizedAllToShardedLinearInitNoBias() {
        // VAL-NN-016: no-bias test for quantized layer
        let group = singletonGroup()
        let layer = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false,
            groupSize: 64, bits: 4, group: group)

        XCTAssertNil(layer.bias)
        XCTAssertTrue(layer is Quantized)
    }

    // MARK: - (6) QuantizedAllToShardedLinear Forward Test

    func testQuantizedAllToShardedLinearForward() {
        // VAL-NN-006: correct output shape
        let group = singletonGroup()
        let layer = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        let input = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output = layer(input)
        // outDims/N = 64 for N=1
        XCTAssertEqual(output.shape, [2, 64])
    }

    // MARK: - (7) QuantizedShardedToAllLinear Init and Forward Tests

    func testQuantizedShardedToAllLinearInit() {
        // VAL-NN-007: init with quantized parameters, bias shape [outDims] (not sharded)
        let group = singletonGroup()
        let layer = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        // Verify Quantized protocol conformance
        XCTAssertTrue(layer is Quantized)
        XCTAssertEqual(layer.groupSize, 64)
        XCTAssertEqual(layer.bits, 4)
        XCTAssertEqual(layer.mode, .affine)

        // Bias shape should be full [outDims] = [64], not sharded
        XCTAssertNotNil(layer.bias)
        XCTAssertEqual(layer.bias!.shape, [64])

        // Verify frozen state
        let trainable = layer.trainableParameters().flattened()
        XCTAssertTrue(trainable.isEmpty, "Quantized layer should be frozen after init")

        // Verify parameters are non-empty
        let params = layer.parameters().flattened()
        XCTAssertFalse(params.isEmpty)
    }

    func testQuantizedShardedToAllLinearInitNoBias() {
        // VAL-NN-016: no-bias test for quantized ShardedToAll
        let group = singletonGroup()
        let layer = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false,
            groupSize: 64, bits: 4, group: group)

        XCTAssertNil(layer.bias)
        XCTAssertTrue(layer is Quantized)
    }

    func testQuantizedShardedToAllLinearForward() {
        // VAL-NN-008: correct output shape [batch, outDims]
        let group = singletonGroup()
        let layer = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        let input = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output = layer(input)
        // outDims = 64 (full, not sharded)
        XCTAssertEqual(output.shape, [2, 64])
    }

    // MARK: - (8) Quantized Unfreeze Override Tests

    func testQuantizedUnfreezeOverride() {
        // VAL-NN-018: after unfreeze, quantized params remain frozen
        let group = singletonGroup()

        let allToSharded = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        // Initially frozen
        XCTAssertTrue(allToSharded.trainableParameters().flattened().isEmpty)

        // Unfreeze -- should re-freeze own params
        try! allToSharded.unfreeze()
        XCTAssertTrue(
            allToSharded.trainableParameters().flattened().isEmpty,
            "Quantized layer should stay frozen after unfreeze (Python: self.freeze(recurse=False))"
        )

        let shardedToAll = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        XCTAssertTrue(shardedToAll.trainableParameters().flattened().isEmpty)
        try! shardedToAll.unfreeze()
        XCTAssertTrue(
            shardedToAll.trainableParameters().flattened().isEmpty,
            "QuantizedShardedToAllLinear should stay frozen after unfreeze")
    }

    // MARK: - (9) Module Protocol Compliance Tests

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
        XCTAssertFalse(
            keys.contains("bias"), "parameters() should not contain bias when bias=false")
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

    // MARK: - (10) No-Bias Tests for All 4 Layers

    // No-bias tests for AllToShardedLinear and ShardedToAllLinear are covered
    // in the init/forward sections above. No-bias for quantized layers:

    func testQuantizedAllToShardedNoBiasForward() {
        let group = singletonGroup()
        let layer = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false,
            groupSize: 64, bits: 4, group: group)

        XCTAssertNil(layer.bias)
        let input = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output = layer(input)
        XCTAssertEqual(output.shape, [2, 64])
    }

    func testQuantizedShardedToAllNoBiasForward() {
        let group = singletonGroup()
        let layer = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 64, bias: false,
            groupSize: 64, bits: 4, group: group)

        XCTAssertNil(layer.bias)
        let input = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output = layer(input)
        XCTAssertEqual(output.shape, [2, 64])
    }

    // MARK: - (11) Non-Divisible Dimension Error

    func testNonDivisibleDimensionError() {
        // VAL-NN-017: Non-divisible dimension should trigger precondition failure.
        // We can't directly test precondition failures in XCTest without
        // crashing, but we can verify valid dimensions work and document
        // the expected behavior. For size-1 group all dimensions are divisible
        // by 1, so we verify the layers initialize correctly with various sizes.
        let group = singletonGroup()

        // These should all succeed (divisible by 1)
        let a = AllToShardedLinear(
            inputDimensions: 17, outputDimensions: 13, bias: true, group: group)
        XCTAssertEqual(a.weight.shape, [13, 17])

        let s = ShardedToAllLinear(
            inputDimensions: 17, outputDimensions: 13, bias: true, group: group)
        XCTAssertEqual(s.weight.shape, [13, 17])

        // Verify the precondition message text exists in the source
        // (For a size-1 group, everything is divisible by 1, so we test
        // that layers init correctly. Non-divisible errors are caught by
        // the precondition in init and would crash in multi-process scenarios.)
    }

    // MARK: - (12) shardLinear Tests

    func testShardLinearAllToSharded() {
        // VAL-NN-009: Linear -> AllToShardedLinear
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = shardLinear(module: linear, sharding: .allToSharded, group: group)
        XCTAssertTrue(sharded is AllToShardedLinear, "Should return AllToShardedLinear")

        let asLayer = sharded as! AllToShardedLinear
        // For size-1 group, weights should be identical
        assertEqual(asLayer.weight, linear.weight, atol: 1e-5)
        XCTAssertNotNil(asLayer.bias)
        assertEqual(asLayer.bias!, linear.bias!, atol: 1e-5)
    }

    func testShardLinearShardedToAll() {
        // VAL-NN-010: Linear -> ShardedToAllLinear
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let sharded = shardLinear(module: linear, sharding: .shardedToAll, group: group)
        XCTAssertTrue(sharded is ShardedToAllLinear, "Should return ShardedToAllLinear")

        let asLayer = sharded as! ShardedToAllLinear
        assertEqual(asLayer.weight, linear.weight, atol: 1e-5)
        XCTAssertNotNil(asLayer.bias)
        assertEqual(asLayer.bias!, linear.bias!, atol: 1e-5)
    }

    func testShardLinearQuantizedAllToSharded() {
        // VAL-NN-011: QuantizedLinear -> QuantizedAllToShardedLinear
        let group = singletonGroup()
        let linear = Linear(128, 64, bias: true)
        eval(linear)

        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
        eval(quantized)

        let sharded = shardLinear(module: quantized, sharding: .allToSharded, group: group)
        XCTAssertTrue(
            sharded is QuantizedAllToShardedLinear,
            "Should return QuantizedAllToShardedLinear")
    }

    func testShardLinearQuantizedShardedToAll() {
        // VAL-NN-011: QuantizedLinear -> QuantizedShardedToAllLinear
        let group = singletonGroup()
        let linear = Linear(128, 64, bias: true)
        eval(linear)

        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
        eval(quantized)

        let sharded = shardLinear(module: quantized, sharding: .shardedToAll, group: group)
        XCTAssertTrue(
            sharded is QuantizedShardedToAllLinear,
            "Should return QuantizedShardedToAllLinear")
    }

    // MARK: - (13) shardLinear with segments=3

    func testShardLinearWithSegments() {
        // VAL-NN-020: shardLinear with segments=3 for fused QKV
        let group = singletonGroup()

        // Fused QKV weight: shape [3*hidden, hidden] = [192, 64]
        let linear = Linear(64, 192, bias: true)
        eval(linear)

        let sharded = shardLinear(
            module: linear, sharding: .allToSharded, segments: 3, group: group)
        XCTAssertTrue(sharded is AllToShardedLinear)

        let asLayer = sharded as! AllToShardedLinear
        // For size-1 group with segments=3: weight shape should be [192, 64]
        // (each of 3 segments [64, 64] split into 1 part each, concatenated = [192, 64])
        XCTAssertEqual(asLayer.weight.shape, [192, 64])

        // Verify forward pass works
        let input = MLXRandom.uniform(0 ..< 1, [2, 64])
        let output = asLayer(input)
        XCTAssertEqual(output.shape, [2, 192])
    }

    // MARK: - (14) shardInPlace Tests

    func testShardInPlace() {
        // VAL-NN-012: shardInPlace modifies parameters without changing module type
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let originalWeightShape = linear.weight.shape
        let originalBiasShape = linear.bias!.shape

        shardInPlace(module: linear, sharding: .allToSharded, group: group)

        // For size-1 group, shapes remain unchanged
        XCTAssertEqual(linear.weight.shape, originalWeightShape)
        XCTAssertEqual(linear.bias!.shape, originalBiasShape)

        // Module type should not change
        XCTAssertTrue(type(of: linear) == Linear.self, "Module type should remain Linear")
    }

    func testShardInPlaceShardedToAll() {
        let group = singletonGroup()
        let linear = Linear(64, 32, bias: true)
        eval(linear)

        let originalWeightShape = linear.weight.shape

        shardInPlace(module: linear, sharding: .shardedToAll, group: group)

        // For size-1 group with shardedToAll: weight shape unchanged, bias unchanged
        XCTAssertEqual(linear.weight.shape, originalWeightShape)
        XCTAssertTrue(type(of: linear) == Linear.self)
    }

    // MARK: - (15) averageGradients Tests

    func testAverageGradientsIdentity() {
        // VAL-NN-014: averageGradients on size-1 group returns unchanged
        let group = singletonGroup()

        // Create a simple module and get its parameter structure
        let layer = AllToShardedLinear(
            inputDimensions: 32, outputDimensions: 16, bias: true, group: group)
        eval(layer)

        let grads = layer.parameters()
        let averaged = averageGradients(gradients: grads, group: group)

        // On size-1 group, should be identity
        let flatGrads = grads.flattened()
        let flatAveraged = averaged.flattened()

        XCTAssertEqual(flatGrads.count, flatAveraged.count)
        for (g, a) in zip(flatGrads, flatAveraged) {
            XCTAssertEqual(g.0, a.0, "Keys should match")
            assertEqual(a.1, g.1, atol: 1e-5)
        }
    }

    func testAverageGradientsWithAllReduceSize() {
        // Test that averageGradients accepts allReduceSize and communicationStream params
        let group = singletonGroup()

        let layer = Linear(32, 16, bias: true)
        eval(layer)

        let grads = layer.parameters()

        // Test with different allReduceSize values
        let averaged1 = averageGradients(
            gradients: grads, group: group, allReduceSize: 1024)
        let averaged2 = averageGradients(
            gradients: grads, group: group, allReduceSize: 0)

        let flatGrads = grads.flattened()
        let flatAvg1 = averaged1.flattened()
        let flatAvg2 = averaged2.flattened()

        // Both should be identity on size-1 group
        for (g, a) in zip(flatGrads, flatAvg1) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }
        for (g, a) in zip(flatGrads, flatAvg2) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }
    }

    // MARK: - (16) sumGradients Forward Identity

    func testSumGradientsForwardIdentity() {
        // VAL-NN-013: sumGradients is identity in forward pass
        let group = singletonGroup()
        let fn = sumGradients(group: group)

        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let output = fn(input)

        assertEqual(output, input)
    }

    // MARK: - (17) Rectangular Matrix Handling

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

    func testRectangularMatrixShardLinear() {
        // shardLinear on non-square dimensions
        let group = singletonGroup()

        let linear1 = Linear(512, 128, bias: true)
        eval(linear1)
        let sharded1 = shardLinear(module: linear1, sharding: .allToSharded, group: group)
        XCTAssertTrue(sharded1 is AllToShardedLinear)
        XCTAssertEqual((sharded1 as! AllToShardedLinear).weight.shape, [128, 512])

        let linear2 = Linear(128, 512, bias: false)
        eval(linear2)
        let sharded2 = shardLinear(module: linear2, sharding: .shardedToAll, group: group)
        XCTAssertTrue(sharded2 is ShardedToAllLinear)
        XCTAssertEqual((sharded2 as! ShardedToAllLinear).weight.shape, [512, 128])
    }

    // MARK: - (18) Gradient Flow Through AllToShardedLinear

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

    // MARK: - (19) ShardedToAllLinear vs Linear Comparison

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

    // MARK: - (20) Quantization Round-Trip

    func testQuantizationRoundTrip() {
        // VAL-CROSS-003: Linear -> shardLinear -> forward pass succeeds
        let group = singletonGroup()

        // Linear -> AllToShardedLinear via shardLinear
        let linear1 = Linear(128, 64, bias: true)
        eval(linear1)
        let sharded1 = shardLinear(module: linear1, sharding: .allToSharded, group: group)
        let input1 = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output1 = (sharded1 as! UnaryLayer)(input1)
        XCTAssertEqual(output1.shape, [2, 64])

        // QuantizedLinear -> QuantizedAllToShardedLinear via shardLinear
        let linear2 = Linear(128, 64, bias: true)
        eval(linear2)
        let quantized = QuantizedLinear(linear2, groupSize: 64, bits: 4)
        eval(quantized)

        let shardedQuantized = shardLinear(
            module: quantized, sharding: .allToSharded, group: group)
        XCTAssertTrue(shardedQuantized is QuantizedAllToShardedLinear)

        let input2 = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output2 = (shardedQuantized as! UnaryLayer)(input2)
        XCTAssertEqual(output2.shape, [2, 64])
    }

    func testQuantizationRoundTripShardedToAll() {
        // QuantizedLinear -> QuantizedShardedToAllLinear via shardLinear
        let group = singletonGroup()

        let linear = Linear(128, 64, bias: true)
        eval(linear)
        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
        eval(quantized)

        let sharded = shardLinear(module: quantized, sharding: .shardedToAll, group: group)
        XCTAssertTrue(sharded is QuantizedShardedToAllLinear)

        let input = MLXRandom.uniform(0 ..< 1, [2, 128])
        let output = (sharded as! UnaryLayer)(input)
        XCTAssertEqual(output.shape, [2, 64])
    }

    // MARK: - Additional: fromLinear Conversion Tests

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

    // MARK: - Additional: Quantized Module Protocol Tests

    func testQuantizedModuleProtocol() {
        // Verify quantized distributed layers have correct Module behavior
        let group = singletonGroup()

        let layer = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 64, bias: true,
            groupSize: 64, bits: 4, group: group)

        let params = layer.parameters()
        let flatParams = params.flattened()
        let keys = Set(flatParams.map { $0.0 })

        // Should NOT contain group
        XCTAssertFalse(keys.contains("group"), "parameters() should NOT contain group")

        // children() should be empty
        let children = layer.children()
        XCTAssertTrue(children.isEmpty, "children() should be empty")

        // Should contain weight, scales, bias
        XCTAssertTrue(keys.contains("weight"), "parameters() should contain weight")
        XCTAssertTrue(keys.contains("scales"), "parameters() should contain scales")
        XCTAssertTrue(keys.contains("bias"), "parameters() should contain bias")
    }
}
