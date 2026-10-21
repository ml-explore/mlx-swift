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

    // `Memory.bufferSize` reports the allocator length of each buffer, and the
    // allocator may hand out a cached buffer up to ~2x the requested size, so the
    // exact value depends on the allocation history of the process (and other
    // tests running in parallel).  These tests only check the properties that hold
    // regardless of the cache.

    func testBufferSizeAtLeastLogicalSize() {
        let a = MLXRandom.normal([64, 32])
        let b = MLXArray.zeros([10, 10], type: UInt8.self)
        eval(a, b)

        XCTAssertGreaterThanOrEqual(Memory.bufferSize([a]), a.nbytes)
        XCTAssertGreaterThanOrEqual(Memory.bufferSize([b]), b.nbytes)
    }

    func testBufferSizeEmpty() {
        XCTAssertEqual(Memory.bufferSize([MLXArray]()), 0)
    }

    func testBufferSizeCountsDuplicatesOnce() {
        let a = MLXRandom.normal([64, 32])
        eval(a)

        XCTAssertEqual(Memory.bufferSize([a, a, a]), Memory.bufferSize([a]))
    }

    func testBufferSizeCountsViewsOnce() {
        let a = MLXRandom.normal([64, 32])
        let slice = a[0 ..< 8]
        let transposed = a.T
        eval(a, slice, transposed)

        // views share the buffer of `a`
        let size = Memory.bufferSize([a])
        XCTAssertEqual(Memory.bufferSize([a, slice]), size)
        XCTAssertEqual(Memory.bufferSize([a, transposed]), size)
        XCTAssertEqual(Memory.bufferSize([slice, transposed]), size)
    }

    func testBufferSizeSumsDistinctBuffers() {
        let a = MLXRandom.normal([64, 32])
        let b = MLXRandom.normal([5])
        let c = MLXRandom.randInt(low: 0, high: 4, [100], type: UInt32.self)
        eval(a, b, c)

        XCTAssertEqual(
            Memory.bufferSize([a, b, c]),
            Memory.bufferSize([a]) + Memory.bufferSize([b]) + Memory.bufferSize([c]))
    }
}
