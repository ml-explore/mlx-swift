// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Multi process tests for the ring backend, ported from the Python
/// distributed tests.
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

    func testRingCollectives() throws {
        try DistributedHarness.run(ranks: Self.rankCount, testName: Self.testName) { group in
            try runRankBody(group)
        }
    }

    private func runRankBody(_ group: MLXDistributed.Group) throws {
        let rank = group.rank
        let size = group.size
        print("[rank \(rank)] joined a group of size \(size)")
        XCTAssertEqual(size, Self.rankCount)
        XCTAssertTrue(rank >= 0 && rank < size)

        try reductions(group)

        // every rank contributes base * (rank + 1)
        let base = MLXArray([1, 2, 3]).asType(.float32)
        let x = base * Float(rank + 1)

        // allGather concatenates along the first axis in rank order
        assertEqual(
            MLXDistributed.allGather(x, group: group),
            concatenated((0 ..< size).map { base * Float($0 + 1) }))

        // send to both neighbors, one pair at a time so that every rank issues
        // the same operations in the same order.  The ring backend connects
        // only neighbors, and with three ranks the left and right ones differ.
        for sender in 0 ..< size {
            for receiver in [(sender + 1) % size, (sender + size - 1) % size] {
                if rank == sender {
                    try checkedEval(MLXDistributed.send(x, to: receiver, group: group))
                } else if rank == receiver {
                    let received = MLXDistributed.recvLike(x, from: sender, group: group)
                    try checkedEval(received)
                    assertEqual(received, base * Float(sender + 1))
                }
            }
        }
    }

    /// allSum, allMax and allMin across types and sizes, as in the Python
    /// distributed tests.  The large sizes exercise the ring's chunked
    /// transfers, which a handful of elements never reach.
    private func reductions(_ group: MLXDistributed.Group) throws {
        let rank = group.rank
        let size = group.size

        let dtypes: [(DType, Float)] = [
            (.int8, 0), (.uint8, 0), (.int32, 0), (.uint32, 0),
            (.float32, 1e-6), (.float16, 5e-3), (.bfloat16, 1e-1),
        ]
        let shapes = [[7], [10], [1024], [1024, 1024]]
        let key = MLXRandom.key(0)

        for (dtype, rtol) in dtypes {
            for shape in shapes {
                // every rank generates the same array and contributes its row
                let x = (MLXRandom.uniform(0 ..< 1, [size] + shape, key: key) * 10)
                    .asType(dtype)
                let name = "\(dtype) \(shape)"

                let sum = MLXDistributed.allSum(x[rank], group: group)
                let expected = x.sum(axis: 0)
                var error = abs(sum - expected)
                if rtol > 0 {
                    error = error / abs(expected)
                }
                try checkedEval(error)
                XCTAssertLessThanOrEqual(
                    error.max().asType(.float32).item(Float.self), rtol, "allSum \(name)")

                let maximum = MLXDistributed.allMax(x[rank], group: group)
                try checkedEval(maximum)
                XCTAssertTrue(
                    (maximum .== x.max(axis: 0)).all().item(Bool.self), "allMax \(name)")

                let minimum = MLXDistributed.allMin(x[rank], group: group)
                try checkedEval(minimum)
                XCTAssertTrue(
                    (minimum .== x.min(axis: 0)).all().item(Bool.self), "allMin \(name)")
            }
        }
    }
}
