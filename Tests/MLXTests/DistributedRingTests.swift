// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Multi process tests for the ring backend, ported from the Python
/// distributed tests: `test_groups`, `test_all_reduce`, `test_all_gather`,
/// `test_send_recv` and `test_all_gather_vjp`, together with the ring specific
/// `test_all_reduce_extra` and `test_all_gather_extra`.
///
/// ``DistributedHarness`` runs the body in several processes.  They are
/// skipped unless `MLX_TEST_DISTRIBUTED=1` is set:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 xcrun xctest -XCTest DistributedRingTests \
///     .../MLXTests.xctest
/// ```
///
/// There is deliberately a single test method: MLX caches the group per
/// process, and a single method also guarantees every rank issues the same
/// operations in the same order, which the ring backend requires.
class DistributedRingTests: XCTestCase {

    static let testName = "DistributedRingTests/testRingCollectives"

    /// Three ranks, so that the left and right neighbors of a rank are
    /// different processes.
    static let rankCount = 3

    /// The dtypes of the Python tests, including the ring specific ones.
    static let dtypes: [DType] = [
        .int8, .uint8, .int16, .uint16, .int32, .uint32, .float32, .float16, .bfloat16,
        .complex64,
    ]

    /// Sizes from the Python tests.  The large ones exercise the ring's chunked
    /// transfers, which a handful of elements never reach.
    static let shapes = [[7], [10], [1024], [1024, 1024]]

    func testRingCollectives() throws {
        try DistributedHarness.run(ranks: Self.rankCount, testName: Self.testName) { group in
            try runRankBody(group)
        }
    }

    private func runRankBody(_ group: MLXDistributed.Group) throws {
        try groups(group)
        try reductions(group)
        try allGather(group)
        try sendRecv(group)
        try allGatherVJP(group)
    }

    /// Port of `test_groups`.
    private func groups(_ group: MLXDistributed.Group) throws {
        XCTAssertEqual(group.size, Self.rankCount)
        XCTAssertTrue(group.rank >= 0 && group.rank < group.size)

        // initializing again yields the same group
        let again = try MLXDistributed.initialize()
        XCTAssertEqual(again.size, group.size)
        XCTAssertEqual(again.rank, group.rank)

        // the ring backend does not support splitting
        XCTAssertThrowsError(try group.split(color: group.rank % 2))
    }

    /// Port of `test_all_reduce` and `test_all_reduce_extra`.
    ///
    /// Every rank builds the same array and contributes its own row, so the
    /// reduction over the rows is the expected result in every process.
    private func reductions(_ group: MLXDistributed.Group) throws {
        let tolerances: [DType: Float] = [
            .float32: 1e-6, .float16: 5e-3, .bfloat16: 1e-1, .complex64: 1e-6,
        ]
        let key = MLXRandom.key(0)
        var combinations = 0
        defer {
            if group.rank == 0 {
                print("reductions: \(combinations) dtype and shape combinations")
            }
        }

        for dtype in Self.dtypes {
            for shape in Self.shapes {
                let rtol = tolerances[dtype] ?? 0
                let x = (MLXRandom.uniform(0 ..< 1, [group.size] + shape, key: key) * 10)
                    .asType(dtype)
                let name = "\(dtype) \(shape)"
                combinations += 1

                let sum = MLXDistributed.allSum(x[group.rank], group: group)
                let expected = x.sum(axis: 0)
                var error = abs(sum - expected)
                if rtol > 0 {
                    error = error / abs(expected)
                }
                try checkedEval(error)
                XCTAssertLessThanOrEqual(
                    error.max().asType(.float32).item(Float.self), rtol, "allSum \(name)")

                let maximum = MLXDistributed.allMax(x[group.rank], group: group)
                try checkedEval(maximum)
                XCTAssertTrue(
                    (maximum .== x.max(axis: 0)).all().item(Bool.self), "allMax \(name)")

                let minimum = MLXDistributed.allMin(x[group.rank], group: group)
                try checkedEval(minimum)
                XCTAssertTrue(
                    (minimum .== x.min(axis: 0)).all().item(Bool.self), "allMin \(name)")
            }
        }
    }

    /// Port of `test_all_gather` and `test_all_gather_extra`.
    private func allGather(_ group: MLXDistributed.Group) throws {
        for dtype in Self.dtypes {
            let x = MLXArray.ones([2, 2, 4]).asType(dtype)
            let gathered = MLXDistributed.allGather(x, group: group)
            try checkedEval(gathered)

            XCTAssertEqual(gathered.shape, [group.size * 2, 2, 4], "allGather \(dtype)")
            XCTAssertTrue(
                (gathered .== MLXArray(1).asType(dtype)).all().item(Bool.self),
                "allGather \(dtype)")
        }

        // the shards are concatenated in rank order rather than reduced
        let base = MLXArray([1, 2, 3]).asType(.float32)
        let gathered = MLXDistributed.allGather(base * Float(group.rank + 1), group: group)
        try checkedEval(gathered)
        assertEqual(gathered, concatenated((0 ..< group.size).map { base * Float($0 + 1) }))
    }

    /// Port of `test_send_recv`.
    ///
    /// Every rank sends to its right neighbor and receives from its left one.
    /// Even ranks send first and odd ranks receive first, so that neighbors do
    /// not wait on each other.
    private func sendRecv(_ group: MLXDistributed.Group) throws {
        let rank = group.rank
        let size = group.size
        let right = (rank + 1) % size
        let left = (rank + size - 1) % size
        let key = MLXRandom.key(0)
        var transfers = 0
        defer {
            if rank == 0 {
                print("send/recv: \(transfers) dtype and shape combinations")
            }
        }

        for dtype in Self.dtypes {
            for shape in Self.shapes {
                let x = (MLXRandom.uniform(0 ..< 1, [size] + shape, key: key) * 10)
                    .asType(dtype)
                let name = "\(dtype) \(shape)"
                transfers += 1

                let sent: MLXArray
                let received: MLXArray
                if rank % 2 == 0 {
                    sent = MLXDistributed.send(x[rank], to: right, group: group)
                    received = MLXDistributed.recvLike(sent, from: left, group: group)
                    try checkedEval(sent, received)
                } else {
                    received = MLXDistributed.recvLike(x[rank], from: left, group: group)
                    sent = MLXDistributed.send(x[rank], to: right, group: group)
                    try checkedEval(received, sent)
                }

                XCTAssertTrue((sent .== x[rank]).all().item(Bool.self), "send \(name)")
                XCTAssertTrue((received .== x[left]).all().item(Bool.self), "recv \(name)")
            }
        }
    }

    /// Port of `test_all_gather_vjp`.
    ///
    /// The gathered array starts with rank 0's contribution, so only rank 0
    /// sees a gradient.
    private func allGatherVJP(_ group: MLXDistributed.Group) throws {
        let gradient = grad { x in
            MLXDistributed.allGather(x, group: group)[0]
        }(MLXArray(1.0))
        try checkedEval(gradient)

        XCTAssertEqual(gradient.item(Float.self), group.rank == 0 ? 1.0 : 0.0)
    }
}
