// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Single process tests for ``MLXDistributed``.
///
/// Without `MLX_HOSTFILE` / `MLX_RANK` no backend can form a communication
/// group, so `initialize()` yields an empty group of size one.  At that size
/// MLX short circuits the collectives to the identity and rejects the point to
/// point operations, which is enough to cover the whole API surface in CI.
///
/// Multi process behaviour is verified by launching several copies of a
/// program with a hostfile, matching how the Python tests are run.
class DistributedTests: XCTestCase {

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

    // MARK: - Availability

    func testIsAvailableLinks() {
        for backend in MLXDistributed.Backend.allCases {
            _ = MLXDistributed.isAvailable(backend)
        }
    }

    /// The ring backend is built on all platforms -- it needs only TCP sockets.
    func testRingIsAvailable() {
        XCTAssertTrue(MLXDistributed.isAvailable(.ring))

        // `any` is true when at least one backend is available
        XCTAssertTrue(MLXDistributed.isAvailable(.any))
    }

    /// Backends that are not compiled in report `false` rather than trapping.
    func testUnavailableBackendsReportFalse() {
        XCTAssertFalse(MLXDistributed.isAvailable(.mpi))
        XCTAssertFalse(MLXDistributed.isAvailable(.jaccl))

        // nccl is CUDA-only and is never built on Apple platforms
        #if !os(Linux)
            XCTAssertFalse(MLXDistributed.isAvailable(.nccl))
        #endif
    }

    // MARK: - Group

    func testDefaultGroupIsSingleton() throws {
        let group = try XCTUnwrap(MLXDistributed.initialize())
        XCTAssertEqual(group.rank, 0)
        XCTAssertEqual(group.size, 1)
    }

    /// A strict init returns nil rather than a group with a null handle whose
    /// rank and size would silently read as zero.
    func testStrictInitializeReturnsNil() {
        var group: MLXDistributed.Group?
        XCTAssertThrowsError(
            try withError {
                group = MLXDistributed.initialize(backend: .any, strict: true)
            })
        XCTAssertNil(group)
    }

    /// A strict init reports an error when no backend can form a group.
    ///
    /// Note this uses `.any`: MLX caches successfully registered groups per
    /// backend name, but never caches under "any" when initialization fails,
    /// so this is independent of the order the tests run in.
    func testStrictInitializeReportsError() {
        XCTAssertThrowsError(
            try withError {
                MLXDistributed.initialize(backend: .any, strict: true)
            })
    }

    func testSplitSingletonGroupReportsError() throws {
        let group = try XCTUnwrap(MLXDistributed.initialize())
        XCTAssertThrowsError(
            try withError {
                group.split(color: 0)
            })
    }

    // MARK: - Collectives

    /// In a group of size one the collectives return their input unchanged.
    func testCollectivesAreIdentityInSingletonGroup() {
        let x = MLXArray([1, 2, 3, 4], [2, 2])

        assertEqual(MLXDistributed.allSum(x), x)
        assertEqual(MLXDistributed.allMax(x), x)
        assertEqual(MLXDistributed.allMin(x), x)
        assertEqual(MLXDistributed.allGather(x), x)
        assertEqual(MLXDistributed.sumScatter(x), x)
    }

    // MARK: - Point to point

    func testSendInSingletonGroupReportsError() {
        let x = MLXArray([1, 2, 3])
        XCTAssertThrowsError(
            try withError {
                MLXDistributed.send(x, to: 0)
            })
    }

    func testRecvInSingletonGroupReportsError() {
        XCTAssertThrowsError(
            try withError {
                MLXDistributed.recv([3], dtype: .int32, from: 0)
            })
    }

    func testRecvLikeInSingletonGroupReportsError() {
        let x = MLXArray([1, 2, 3])
        XCTAssertThrowsError(
            try withError {
                MLXDistributed.recvLike(x, from: 0)
            })
    }
}
