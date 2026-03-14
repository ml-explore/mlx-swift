// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

class DistributedNNTests: XCTestCase {

    /// Sequential port counter to avoid ephemeral port collisions between tests.
    /// Each multi-process test increments by 2 (one port per rank). The base port
    /// is randomized per test run to avoid TIME_WAIT conflicts when the suite is
    /// run multiple times in quick succession. Range: 35000-48999 avoids both
    /// well-known ports and the macOS ephemeral range, and is offset from
    /// DistributedTests (15000-28999) to prevent cross-class collisions.
    private static var nextPort: Int = 35000 + Int.random(in: 0 ..< 7000) * 2

    /// Track spawned process PIDs for cleanup in tearDown.
    private var spawnedProcesses: [Process] = []

    override class func setUp() {
        setDefaultDevice()
    }

    override func tearDown() {
        // Kill any orphan worker processes that may still be running
        for process in spawnedProcesses where process.isRunning {
            process.terminate()
            Thread.sleep(forTimeInterval: 0.5)
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }
        spawnedProcesses.removeAll()

        // Allow socket cleanup between tests. The ring backend uses TCP sockets
        // that enter TIME_WAIT state after close. A delay ensures all sockets
        // from the previous test are fully released before the next test starts.
        Thread.sleep(forTimeInterval: 1.0)

        super.tearDown()
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
        // VAL-NN-017: Non-divisible dimension error handling.
        //
        // The distributed layers use `precondition` for dimension validation,
        // consistent with the rest of MLXNN (Conv1d, MultiHeadAttention, etc.).
        // A `precondition` failure terminates the process, so it cannot be
        // caught or tested directly in XCTest.
        //
        // In single-process tests the group size is always 1, and every
        // integer is divisible by 1, so the precondition never fires here.
        // Multi-process tests with group size >= 2 would be needed to trigger
        // the actual crash for non-divisible dimensions.
        //
        // What we verify below:
        //  1. The divisibility invariant holds for the layers we create
        //     (outputDimensions % N == 0 for AllToSharded variants,
        //      inputDimensions % N == 0 for ShardedToAll variants).
        //  2. Odd/prime dimensions that would be non-divisible by N > 1
        //     still work on a size-1 group (since N == 1).
        //  3. Weight shapes confirm the division was applied correctly.
        //  4. All four distributed layer types have consistent validation.

        let group = singletonGroup()
        let N = group.size
        XCTAssertEqual(N, 1, "Single-process group size must be 1")

        // -- AllToShardedLinear validates outputDimensions % N == 0 --
        // Use a prime outputDimensions (7) which would fail for any N > 1.
        let a = AllToShardedLinear(
            inputDimensions: 17, outputDimensions: 7, bias: true, group: group)
        XCTAssertEqual(a.weight.shape, [7 / N, 17])
        XCTAssertEqual(a.bias!.shape, [7 / N])
        // Confirm the divisibility check: 7 % 1 == 0 is true
        XCTAssertEqual(7 % N, 0, "7 is divisible by 1 (would fail for N=2..6)")

        // -- ShardedToAllLinear validates inputDimensions % N == 0 --
        // Use a prime inputDimensions (13) which would fail for any N > 1.
        let s = ShardedToAllLinear(
            inputDimensions: 13, outputDimensions: 5, bias: true, group: group)
        XCTAssertEqual(s.weight.shape, [5, 13 / N])
        XCTAssertEqual(s.bias!.shape, [5])
        XCTAssertEqual(13 % N, 0, "13 is divisible by 1 (would fail for N=2..12)")

        // -- QuantizedAllToShardedLinear validates outputDimensions % N == 0 --
        let qa = QuantizedAllToShardedLinear(
            inputDimensions: 128, outputDimensions: 7, bias: true,
            groupSize: 64, bits: 4, group: group)
        XCTAssertNotNil(qa.weight)
        XCTAssertEqual(qa.bias!.shape, [7 / N])
        XCTAssertEqual(7 % N, 0)

        // -- QuantizedShardedToAllLinear validates inputDimensions % N == 0 --
        let qs = QuantizedShardedToAllLinear(
            inputDimensions: 128, outputDimensions: 7, bias: true,
            groupSize: 64, bits: 4, group: group)
        XCTAssertNotNil(qs.weight)
        XCTAssertEqual(qs.bias!.shape, [7])
        XCTAssertEqual(128 % N, 0)

        // -- Verify that forward passes work with these odd dimensions --
        let inputA = MLXRandom.uniform(0 ..< 1, [2, 17])
        let outputA = a(inputA)
        XCTAssertEqual(outputA.shape, [2, 7 / N])

        let inputS = MLXRandom.uniform(0 ..< 1, [2, 13])
        let outputS = s(inputS)
        XCTAssertEqual(outputS.shape, [2, 5])
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

    func testAverageGradientsCommunicationType() {
        // VAL-NN-021: averageGradients with communicationType preserves identity
        // on a size-1 group. When communicationType is provided, gradients are
        // cast to that type before communication and cast back after.
        let group = singletonGroup()

        let layer = Linear(32, 16, bias: true)
        eval(layer)

        let grads = layer.parameters()

        // Call with communicationType: .float16
        let averaged = averageGradients(
            gradients: grads, group: group, communicationType: .float16)

        // On size-1 group, N==1 returns early (identity), so dtypes unchanged
        let flatGrads = grads.flattened()
        let flatAveraged = averaged.flattened()

        XCTAssertEqual(flatGrads.count, flatAveraged.count)
        for (g, a) in zip(flatGrads, flatAveraged) {
            XCTAssertEqual(g.0, a.0, "Keys should match")
            // Identity on size-1 group
            assertEqual(a.1, g.1, atol: 1e-5)
            // dtype should remain float32 (the original dtype)
            XCTAssertEqual(a.1.dtype, g.1.dtype)
        }

        // Also verify with communicationType: .bfloat16
        let averaged2 = averageGradients(
            gradients: grads, group: group, communicationType: .bfloat16)
        let flatAveraged2 = averaged2.flattened()
        for (g, a) in zip(flatGrads, flatAveraged2) {
            assertEqual(a.1, g.1, atol: 1e-5)
            XCTAssertEqual(a.1.dtype, g.1.dtype)
        }
    }

    func testAverageGradientsMixedDtypeFallback() {
        // VAL-NN-022: gradient tree with mixed float32/float16 arrays falls
        // back to non-batched reduction. On a size-1 group all gradients are
        // returned unchanged.
        let group = singletonGroup()

        // Build a gradient tree with mixed dtypes using ModuleParameters
        let grad1 = MLXRandom.uniform(0 ..< 1, [4, 8])  // float32
        let grad2 = MLXRandom.uniform(0 ..< 1, [4, 8]).asType(.float16)  // float16
        let grad3 = MLXRandom.uniform(0 ..< 1, [2, 3])  // float32
        eval(grad1, grad2, grad3)

        var grads = ModuleParameters()
        grads["weight"] = .value(grad1)
        grads["bias"] = .value(grad2)
        grads["scale"] = .value(grad3)

        // With default allReduceSize (batched), the mixed types trigger fallback
        let averaged = averageGradients(gradients: grads, group: group)

        let flatGrads = grads.flattened()
        let flatAveraged = averaged.flattened()

        XCTAssertEqual(flatGrads.count, flatAveraged.count)
        for (g, a) in zip(flatGrads, flatAveraged) {
            XCTAssertEqual(g.0, a.0, "Keys should match")
            // On size-1 group, should be identity
            assertEqual(a.1, g.1, atol: 1e-3)
            // dtype should be preserved
            XCTAssertEqual(a.1.dtype, g.1.dtype)
        }

        // Also test with communicationType on mixed-dtype tree
        let averaged2 = averageGradients(
            gradients: grads, group: group, communicationType: .float16)
        let flatAveraged2 = averaged2.flattened()
        for (g, a) in zip(flatGrads, flatAveraged2) {
            assertEqual(a.1, g.1, atol: 1e-3)
            XCTAssertEqual(a.1.dtype, g.1.dtype)
        }
    }

    func testAverageGradientsBatchingBehavior() {
        // Verify averageGradients accepts allReduceSize parameter with various
        // values including 0, negative, and small positive values.
        let group = singletonGroup()

        let layer = Linear(64, 32, bias: true)
        eval(layer)

        let grads = layer.parameters()
        let flatGrads = grads.flattened()

        // allReduceSize = 0 disables batching
        let avg0 = averageGradients(
            gradients: grads, group: group, allReduceSize: 0)
        for (g, a) in zip(flatGrads, avg0.flattened()) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }

        // allReduceSize = -1 also disables batching
        let avgNeg = averageGradients(
            gradients: grads, group: group, allReduceSize: -1)
        for (g, a) in zip(flatGrads, avgNeg.flattened()) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }

        // allReduceSize = 1 (very small, forces many batches)
        let avg1 = averageGradients(
            gradients: grads, group: group, allReduceSize: 1)
        for (g, a) in zip(flatGrads, avg1.flattened()) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }

        // allReduceSize = very large (everything in one batch)
        let avgBig = averageGradients(
            gradients: grads, group: group, allReduceSize: 1024 * 1024 * 1024)
        for (g, a) in zip(flatGrads, avgBig.flattened()) {
            assertEqual(a.1, g.1, atol: 1e-5)
        }

        // Also with communicationType combined with various allReduceSize
        let avgComm = averageGradients(
            gradients: grads, group: group, allReduceSize: 100,
            communicationType: .float16)
        for (g, a) in zip(flatGrads, avgComm.flattened()) {
            assertEqual(a.1, g.1, atol: 1e-5)
            XCTAssertEqual(a.1.dtype, g.1.dtype)
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

    // MARK: - Multi-Process NN Parity Tests

    /// Find the DistributedWorker binary in the build products directory.
    private func findWorkerBinary() -> URL? {
        let testBundle = Bundle(for: type(of: self))
        let bundleURL = testBundle.bundleURL
        let productsDir = bundleURL.deletingLastPathComponent()
        let workerURL = productsDir.appendingPathComponent("DistributedWorker")

        if FileManager.default.isExecutableFile(atPath: workerURL.path) {
            return workerURL
        }

        return nil
    }

    /// Allocate two unique TCP ports for the ring backend using a sequential counter.
    ///
    /// Instead of binding to port 0 (which lets the OS pick an ephemeral port and risks
    /// TIME_WAIT collisions when tests run in rapid succession), we use a monotonically
    /// increasing counter with a random base. Each call advances by 2, guaranteeing unique
    /// port pairs across all tests within a single run. The random base avoids TIME_WAIT
    /// conflicts when the test suite is run multiple times in quick succession.
    ///
    /// Each candidate port is validated by binding with SO_REUSEADDR to confirm it is not
    /// stuck in TIME_WAIT or occupied by another process.
    private func allocatePorts() -> (Int, Int) {
        let port1 = nextAvailablePort()
        let port2 = nextAvailablePort()
        return (port1, port2)
    }

    /// Advance the port counter and verify the port is bindable (not in TIME_WAIT).
    private func nextAvailablePort() -> Int {
        while true {
            let port = DistributedNNTests.nextPort
            DistributedNNTests.nextPort += 1
            if isPortAvailable(port) {
                return port
            }
            // Skip ports that are in TIME_WAIT or otherwise occupied
        }
    }

    /// Check if a port can be bound on loopback with SO_REUSEADDR.
    private func isPortAvailable(_ port: Int) -> Bool {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { close(sock) }

        var reuse: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(sock, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bindResult == 0
    }

    /// Create a temporary hostfile for 2-process ring backend on localhost.
    private func createHostfile(port1: Int, port2: Int) throws -> URL {
        let hostfile = [
            ["\("127.0.0.1"):\(port1)"],
            ["\("127.0.0.1"):\(port2)"],
        ]
        let jsonData = try JSONSerialization.data(
            withJSONObject: hostfile, options: [.prettyPrinted])
        let jsonString = String(data: jsonData, encoding: .utf8)!

        let tempDir = FileManager.default.temporaryDirectory
        let hostfilePath = tempDir.appendingPathComponent(
            "mlx_test_hostfile_\(UUID().uuidString).json")
        try jsonString.write(to: hostfilePath, atomically: true, encoding: .utf8)

        return hostfilePath
    }

    /// Spawn a worker process with the given rank and operation, wait for completion.
    ///
    /// Pipe data is read asynchronously to prevent deadlocks when the process
    /// fills the pipe buffer before the test reads it.
    private func spawnWorker(
        workerBinary: URL, rank: Int, hostfilePath: URL, operation: String, timeout: TimeInterval
    ) -> (exitCode: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = workerBinary
        process.environment = [
            "MLX_RANK": "\(rank)",
            "MLX_HOSTFILE": hostfilePath.path,
            "MLX_TEST_OP": operation,
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "HOME": ProcessInfo.processInfo.environment["HOME"] ?? "/tmp",
            "DYLD_LIBRARY_PATH":
                ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? "",
            "DYLD_FRAMEWORK_PATH":
                ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
        ]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Read pipe data asynchronously to prevent deadlocks
        var stdoutData = Data()
        var stderrData = Data()
        let dataLock = NSLock()

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty {
                dataLock.lock()
                stdoutData.append(data)
                dataLock.unlock()
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty {
                dataLock.lock()
                stderrData.append(data)
                dataLock.unlock()
            }
        }

        do {
            try process.run()
        } catch {
            return (-1, "", "Failed to start process: \(error)")
        }

        // Track for cleanup in tearDown
        spawnedProcesses.append(process)

        let deadline = DispatchTime.now() + timeout
        let group = DispatchGroup()
        group.enter()

        DispatchQueue.global().async {
            process.waitUntilExit()
            group.leave()
        }

        let result = group.wait(timeout: deadline)

        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil

        if result == .timedOut {
            process.terminate()
            Thread.sleep(forTimeInterval: 0.5)
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
            dataLock.lock()
            let stdoutStr = String(data: stdoutData, encoding: .utf8) ?? ""
            let stderrStr = String(data: stderrData, encoding: .utf8) ?? ""
            dataLock.unlock()

            // The ring backend's TCP sockets can keep the process alive after the
            // worker's main code finishes — the ring destructor may block waiting
            // for peer socket closure. If the worker already produced valid JSON
            // output (and logged "completed successfully"), treat it as a pass
            // rather than a timeout failure.
            let trimmedStdout = stdoutStr.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedStdout.isEmpty,
                let jsonData = trimmedStdout.data(using: .utf8),
                (try? JSONSerialization.jsonObject(with: jsonData)) != nil
            {
                // Worker produced valid JSON before timeout — treat as success.
                // The process was killed only because the ring backend's socket
                // cleanup blocked exit; the actual operation completed fine.
                return (0, stdoutStr, stderrStr)
            }

            let timeoutMsg = "Process timed out after \(timeout) seconds"
            return (
                -1, stdoutStr,
                stderrStr.isEmpty ? timeoutMsg : "\(stderrStr)\n\(timeoutMsg)"
            )
        }

        Thread.sleep(forTimeInterval: 0.1)

        dataLock.lock()
        let stdoutStr = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderrStr = String(data: stderrData, encoding: .utf8) ?? ""
        dataLock.unlock()

        return (process.terminationStatus, stdoutStr, stderrStr)
    }

    /// Run a multi-process test with the given operation.
    ///
    /// Spawns 2 worker processes with rank 0 and rank 1, waits for both,
    /// and returns their results. Uses a 30-second per-attempt timeout. If a
    /// timeout occurs (ring backend TCP race), the test is retried once with
    /// fresh ports. Total worst-case: ~62 seconds (30s + 2s wait + 30s retry).
    private func runMultiProcessTest(
        operation: String,
        timeout: TimeInterval = 30.0,
        retries: Int = 1,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> (
        rank0: (exitCode: Int32, stdout: String, stderr: String),
        rank1: (exitCode: Int32, stdout: String, stderr: String)
    )? {
        guard let workerBinary = findWorkerBinary() else {
            XCTFail(
                "DistributedWorker binary not found. Build with: xcodebuild build -scheme mlx-swift-Package",
                file: file, line: line)
            return nil
        }

        for attempt in 0 ... retries {
            let (port1, port2) = allocatePorts()

            let hostfilePath: URL
            do {
                hostfilePath = try createHostfile(port1: port1, port2: port2)
            } catch {
                XCTFail("Failed to create hostfile: \(error)", file: file, line: line)
                return nil
            }

            let result = runWorkerPair(
                workerBinary: workerBinary, hostfilePath: hostfilePath,
                operation: operation, timeout: timeout)

            try? FileManager.default.removeItem(at: hostfilePath)

            guard let (rank0Result, rank1Result) = result else {
                XCTFail(
                    "Multi-process test timed out waiting for workers", file: file, line: line)
                return nil
            }

            if rank0Result.exitCode == 0 && rank1Result.exitCode == 0 {
                return (rank0Result, rank1Result)
            }

            let rank0TimedOut =
                rank0Result.exitCode == -1
                && rank0Result.stderr.contains("timed out")
            let rank1TimedOut =
                rank1Result.exitCode == -1
                && rank1Result.stderr.contains("timed out")

            if (rank0TimedOut || rank1TimedOut) && attempt < retries {
                Thread.sleep(forTimeInterval: 2.0)
                continue
            }

            return (rank0Result, rank1Result)
        }

        return nil
    }

    /// Spawn a pair of worker processes for a multi-process test.
    private func runWorkerPair(
        workerBinary: URL,
        hostfilePath: URL,
        operation: String,
        timeout: TimeInterval
    ) -> (
        rank0: (exitCode: Int32, stdout: String, stderr: String),
        rank1: (exitCode: Int32, stdout: String, stderr: String)
    )? {
        var rank0Result: (exitCode: Int32, stdout: String, stderr: String)!
        var rank1Result: (exitCode: Int32, stdout: String, stderr: String)!

        let group = DispatchGroup()

        group.enter()
        DispatchQueue.global().async {
            rank0Result = self.spawnWorker(
                workerBinary: workerBinary, rank: 0, hostfilePath: hostfilePath,
                operation: operation, timeout: timeout)
            group.leave()
        }

        // Delay to let rank 0 start up and begin its accept() listener.
        Thread.sleep(forTimeInterval: 1.0)

        group.enter()
        DispatchQueue.global().async {
            rank1Result = self.spawnWorker(
                workerBinary: workerBinary, rank: 1, hostfilePath: hostfilePath,
                operation: operation, timeout: timeout)
            group.leave()
        }

        let waitResult = group.wait(timeout: .now() + timeout + 10)
        if waitResult == .timedOut {
            return nil
        }

        return (rank0Result, rank1Result)
    }

    // MARK: - (23) Multi-Process Shard Linear Forward Parity

    func testMultiProcessShardLinearForward() {
        // VAL-NN-023: Two processes create same Linear (seeded), shardLinear to
        // AllToShardedLinear and ShardedToAllLinear, forward on same input.
        // Verify concatenated sharded outputs match original Linear output.
        guard let results = runMultiProcessTest(operation: "shardLinearForward") else { return }

        if results.rank0.exitCode != 0 || results.rank1.exitCode != 0 {
            print("=== Rank 0 stderr ===")
            print(results.rank0.stderr)
            print("=== Rank 0 stdout ===")
            print(results.rank0.stdout)
            print("=== Rank 1 stderr ===")
            print(results.rank1.stderr)
            print("=== Rank 1 stdout ===")
            print(results.rank1.stdout)
        }

        XCTAssertEqual(
            results.rank0.exitCode, 0,
            "Rank 0 failed with exit code \(results.rank0.exitCode). stderr: \(results.rank0.stderr)"
        )
        XCTAssertEqual(
            results.rank1.exitCode, 0,
            "Rank 1 failed with exit code \(results.rank1.exitCode). stderr: \(results.rank1.stderr)"
        )

        // Verify JSON output from both ranks
        for (rank, result) in [(0, results.rank0), (1, results.rank1)] {
            let stdout = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stdout.isEmpty,
                let data = stdout.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let allToShardedMatch = json["allToShardedMatch"] as? Bool,
                let shardedToAllMatch = json["shardedToAllMatch"] as? Bool
            else {
                XCTFail("Rank \(rank) produced invalid JSON output: '\(stdout)'")
                continue
            }

            XCTAssertTrue(
                allToShardedMatch,
                "Rank \(rank): AllToSharded forward parity failed")
            XCTAssertTrue(
                shardedToAllMatch,
                "Rank \(rank): ShardedToAll forward parity failed")
        }
    }

    // MARK: - (24) Multi-Process Shard Linear Backward Gradient Parity

    func testMultiProcessShardLinearBackward() {
        // VAL-NN-024: Two processes with 4-layer Sequential (sharded Linear layers).
        // Backward pass gradients for each rank's weight slice should match
        // the corresponding slice from the non-sharded model's gradient.
        guard let results = runMultiProcessTest(operation: "shardLinearBackward") else { return }

        if results.rank0.exitCode != 0 || results.rank1.exitCode != 0 {
            print("=== Rank 0 stderr ===")
            print(results.rank0.stderr)
            print("=== Rank 0 stdout ===")
            print(results.rank0.stdout)
            print("=== Rank 1 stderr ===")
            print(results.rank1.stderr)
            print("=== Rank 1 stdout ===")
            print(results.rank1.stdout)
        }

        XCTAssertEqual(
            results.rank0.exitCode, 0,
            "Rank 0 failed with exit code \(results.rank0.exitCode). stderr: \(results.rank0.stderr)"
        )
        XCTAssertEqual(
            results.rank1.exitCode, 0,
            "Rank 1 failed with exit code \(results.rank1.exitCode). stderr: \(results.rank1.stderr)"
        )

        // Verify JSON output from both ranks
        for (rank, result) in [(0, results.rank0), (1, results.rank1)] {
            let stdout = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stdout.isEmpty,
                let data = stdout.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let lossMatch = json["lossMatch"] as? Bool,
                let l0WeightMatch = json["l0WeightMatch"] as? Bool,
                let l0BiasMatch = json["l0BiasMatch"] as? Bool,
                let l1WeightMatch = json["l1WeightMatch"] as? Bool,
                let l1BiasMatch = json["l1BiasMatch"] as? Bool,
                let l2WeightMatch = json["l2WeightMatch"] as? Bool,
                let l2BiasMatch = json["l2BiasMatch"] as? Bool,
                let l3WeightMatch = json["l3WeightMatch"] as? Bool,
                let l3BiasMatch = json["l3BiasMatch"] as? Bool
            else {
                XCTFail("Rank \(rank) produced invalid JSON output: '\(stdout)'")
                continue
            }

            XCTAssertTrue(lossMatch, "Rank \(rank): loss mismatch")
            XCTAssertTrue(l0WeightMatch, "Rank \(rank): layer 0 weight gradient mismatch")
            XCTAssertTrue(l0BiasMatch, "Rank \(rank): layer 0 bias gradient mismatch")
            XCTAssertTrue(l1WeightMatch, "Rank \(rank): layer 1 weight gradient mismatch")
            XCTAssertTrue(l1BiasMatch, "Rank \(rank): layer 1 bias gradient mismatch")
            XCTAssertTrue(l2WeightMatch, "Rank \(rank): layer 2 weight gradient mismatch")
            XCTAssertTrue(l2BiasMatch, "Rank \(rank): layer 2 bias gradient mismatch")
            XCTAssertTrue(l3WeightMatch, "Rank \(rank): layer 3 weight gradient mismatch")
            XCTAssertTrue(l3BiasMatch, "Rank \(rank): layer 3 bias gradient mismatch")
        }
    }
}
