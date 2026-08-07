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

    func testCacheLimitConcurrentAccess() async {
        // Regression test for the Memory.limits Mutex refactor (see
        // Source/MLX/Memory.swift): hammer cacheLimit get/set from many
        // concurrent tasks.
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 100 {
                group.addTask {
                    Memory.cacheLimit = 1024 * (i + 1)
                    _ = Memory.cacheLimit
                }
            }
        }
    }
}
