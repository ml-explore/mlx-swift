// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Single process tests for the distributed layers.
///
/// Without a backend the global group has size one, so sharding is the
/// identity and the sharded layers must behave exactly like ``Linear``.  That
/// covers construction, parameter shapes, conversion and the forward pass.
class DistributedNNTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// The assertions here describe a group of size one, so they are not valid
    /// in a multi process run -- see ``DistributedRingTests``.
    override func setUpWithError() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["MLX_TEST_DISTRIBUTED"] == "1",
            "Single process assertions; skipped during a multi process run.")
    }

    // MARK: - Segments

    func testSegmentsCount() {
        let w = MLXArray(0 ..< 12, [6, 2]).asType(.float32)

        XCTAssertEqual(Segments.count(1).split(w, axis: 0).count, 1)
        XCTAssertEqual(Segments.count(3).split(w, axis: 0).map { $0.dim(0) }, [2, 2, 2])
    }

    func testSegmentsIndicesAndFractions() {
        let w = MLXArray(0 ..< 20, [10, 2]).asType(.float32)

        XCTAssertEqual(Segments.indices([4, 7]).split(w, axis: 0).map { $0.dim(0) }, [4, 3, 3])
        XCTAssertEqual(
            Segments.fractions([0.5]).split(w, axis: 0).map { $0.dim(0) }, [5, 5])
    }

    // MARK: - sumGradients

    func testSumGradientsIsIdentityInSingletonGroup() {
        let f = sumGradients()
        let x = MLXArray([1, 2, 3]).asType(.float32)

        assertEqual(f(x), x)
    }

    // MARK: - AllToShardedLinear

    func testAllToShardedLinearShapes() {
        let layer = AllToShardedLinear(4, 8)

        XCTAssertEqual(layer.weight.shape, [8, 4])
        XCTAssertEqual(layer.bias?.shape, [8])
    }

    func testAllToShardedLinearWithoutBias() {
        let layer = AllToShardedLinear(4, 8, bias: false)

        XCTAssertNil(layer.bias)
        XCTAssertEqual(layer(MLXRandom.normal([2, 4])).shape, [2, 8])
    }

    /// In a group of size one the sharded layer must match the original.
    func testAllToShardedLinearMatchesLinear() {
        let linear = Linear(4, 8)
        let sharded = AllToShardedLinear(linear)
        let x = MLXRandom.normal([3, 4])

        assertEqual(sharded.weight, linear.weight)
        assertEqual(sharded(x), linear(x))
    }

    // MARK: - ShardedToAllLinear

    func testShardedToAllLinearShapes() {
        let layer = ShardedToAllLinear(4, 8)

        XCTAssertEqual(layer.weight.shape, [8, 4])
        XCTAssertEqual(layer.bias?.shape, [8])
    }

    func testShardedToAllLinearMatchesLinear() {
        let linear = Linear(4, 8)
        let sharded = ShardedToAllLinear(linear)
        let x = MLXRandom.normal([3, 4])

        assertEqual(sharded.weight, linear.weight)
        assertEqual(sharded(x), linear(x))
    }

    // MARK: - shardLinear / shardInPlace

    func testShardLinearReturnsRequestedType() {
        let linear = Linear(4, 8)

        XCTAssertTrue(shardLinear(linear, sharding: .allToSharded) is AllToShardedLinear)
        XCTAssertTrue(shardLinear(linear, sharding: .shardedToAll) is ShardedToAllLinear)
    }

    func testShardInPlaceIsIdentityInSingletonGroup() {
        let linear = Linear(4, 8)
        let weight = linear.weight
        let bias = linear.bias

        shardInPlace(linear, sharding: .allToSharded)

        assertEqual(linear.weight, weight)
        if let bias, let sharded = linear.bias {
            assertEqual(sharded, bias)
        } else {
            XCTFail("bias went missing")
        }
    }

    // MARK: - averageGradients

    func testAverageGradientsIsIdentityInSingletonGroup() {
        let gradients = ModuleParameters.unflattened([
            ("a", MLXArray([1, 2, 3]).asType(.float32)),
            ("b", MLXArray([4, 5]).asType(.float32)),
        ])

        let averaged = averageGradients(gradients)

        assertEqual(
            averaged.flattened().map { $0.1 },
            gradients.flattened().map { $0.1 })
    }

    /// Grouping is disabled with `allReduceSize: 0` -- the result is the same.
    func testAverageGradientsUngrouped() {
        let gradients = ModuleParameters.unflattened([
            ("a", MLXArray([1, 2, 3]).asType(.float32))
        ])

        let averaged = averageGradients(gradients, allReduceSize: 0)

        assertEqual(averaged.flattened().map { $0.1 }, gradients.flattened().map { $0.1 })
    }

    func testAverageGradientsEmpty() {
        let gradients = ModuleParameters()

        XCTAssertTrue(averageGradients(gradients).flattened().isEmpty)
    }
}
