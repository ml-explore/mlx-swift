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

    /// Backends that are not compiled in report `false` rather than trapping.
    func testUnavailableBackendsReportFalse() {
        // nccl is CUDA-only and is never built on Apple platforms
        #if !os(Linux)
            XCTAssertFalse(MLXDistributed.isAvailable(.nccl))
        #endif
    }
}
