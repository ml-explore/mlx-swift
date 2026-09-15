// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class DistributedTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Smoke test: proves the mlx-c distributed wrappers are compiled into
    /// Cmlx and reachable from Swift.
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
}
