// Copyright © 2024 Apple Inc.

import XCTest

@testable import MLX

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    final class SwiftPMMetallibResourceTests: XCTestCase {
        func testFindsGeneratedSwiftPMMetallib() throws {
            let url = try XCTUnwrap(SwiftPMMetallibResource.findMetallibURL())
            XCTAssertEqual(url.lastPathComponent, "default.metallib")
            XCTAssertEqual(
                url.deletingLastPathComponent().lastPathComponent, "mlx-swift_Cmlx.bundle")
        }

        func testGPUStreamLoadsGeneratedSwiftPMMetallib() {
            Stream.gpu.synchronize()
        }
    }
#endif
