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

    func testCacheLimitRoundTrip() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }

        Memory.cacheLimit = 4096
        XCTAssertEqual(Memory.cacheLimit, 4096)
    }

    func testMemoryLimitRoundTrip() {
        let original = Memory.memoryLimit
        defer { Memory.memoryLimit = original }

        Memory.memoryLimit = original + 1024
        XCTAssertEqual(Memory.memoryLimit, original + 1024)
    }
}
