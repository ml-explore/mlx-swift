// Copyright © 2024 Apple Inc.

import XCTest

@testable import MLX

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    final class SwiftPMMetallibResourceTests: XCTestCase {
        func testLookupRequiresMLXBundleAndSupportsBothLayouts() throws {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            try Data().write(to: root.appendingPathComponent("default.metallib"))
            XCTAssertNil(SwiftPMMetallibResource.metallibURL(near: root))

            let bundle = root.appendingPathComponent("mlx-swift_Cmlx.bundle")
            let resources = bundle.appendingPathComponent("Contents/Resources")
            try FileManager.default.createDirectory(
                at: resources, withIntermediateDirectories: true)
            let xcodeLibrary = resources.appendingPathComponent("default.metallib")
            try Data().write(to: xcodeLibrary)
            XCTAssertEqual(SwiftPMMetallibResource.metallibURL(near: root), xcodeLibrary)
            XCTAssertEqual(SwiftPMMetallibResource.metallibURL(near: resources), xcodeLibrary)

            let swiftPMLibrary = bundle.appendingPathComponent("default.metallib")
            try Data().write(to: swiftPMLibrary)
            XCTAssertEqual(SwiftPMMetallibResource.metallibURL(near: root), swiftPMLibrary)
            XCTAssertEqual(SwiftPMMetallibResource.metallibURL(near: bundle), swiftPMLibrary)
        }

        func testFindsGeneratedSwiftPMMetallib() throws {
            let url = try XCTUnwrap(SwiftPMMetallibResource.findMetallibURL())
            XCTAssertEqual(url.lastPathComponent, "default.metallib")
            let bundleLayouts = [
                "/mlx-swift_Cmlx.bundle/default.metallib",
                "/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
            ]
            XCTAssertTrue(bundleLayouts.contains { url.path.hasSuffix($0) }, url.path)
        }

        func testGPUStreamLoadsGeneratedSwiftPMMetallib() {
            Stream.gpu.synchronize()
        }
    }
#endif
