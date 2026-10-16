// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class MemoryTests: XCTestCase {

    func testWiredMemory() {
        Memory.withWiredLimit(1024 * 1024 * 256) {
            let x = MLXArray(10)
            print(x * x)
        }
    }

    func testDeviceInfo() {
        let info = GPU.deviceInfo()
        XCTAssertGreaterThan(info.memorySize, 0)

        guard Device.defaultDevice().deviceType == .gpu else { return }
        XCTAssertNotEqual(info.architecture, "Unknown")
        XCTAssertGreaterThan(info.maxBufferSize, 0)
        XCTAssertGreaterThan(info.maxRecommendedWorkingSetSize, 0)

        let again = GPU.deviceInfo()
        XCTAssertEqual(again.architecture, info.architecture)
        XCTAssertEqual(again.maxBufferSize, info.maxBufferSize)
        XCTAssertEqual(again.maxRecommendedWorkingSetSize, info.maxRecommendedWorkingSetSize)
    }
}
