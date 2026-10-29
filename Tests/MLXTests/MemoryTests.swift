// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct MemoryTests {

    @Test func wiredMemory() {
        Memory.withWiredLimit(1024 * 1024 * 256) {
            let x = MLXArray(10)
            print(x * x)
        }
    }

    @Test func deviceInfo() {
        let info = GPU.deviceInfo()
        #expect(info.memorySize > 0)

        guard Device.defaultDevice().deviceType == .gpu else { return }
        #expect(info.architecture != "Unknown")
        #expect(info.maxBufferSize > 0)
        #expect(info.maxRecommendedWorkingSetSize > 0)

        let again = GPU.deviceInfo()
        #expect(again.architecture == info.architecture)
        #expect(again.maxBufferSize == info.maxBufferSize)
        #expect(again.maxRecommendedWorkingSetSize == info.maxRecommendedWorkingSetSize)
    }

    // `Memory.bufferSize` reports the allocator length of each buffer, and the
    // allocator may hand out a cached buffer up to ~2x the requested size, so the
    // exact value depends on the allocation history of the process (and other
    // tests running in parallel).  These tests only check the properties that hold
    // regardless of the cache.

    @Test func bufferSizeAtLeastLogicalSize() {
        let a = MLXRandom.normal([64, 32])
        let b = MLXArray.zeros([10, 10], type: UInt8.self)
        eval(a, b)

        #expect(Memory.bufferSize([a]) >= a.nbytes)
        #expect(Memory.bufferSize([b]) >= b.nbytes)
    }

    @Test func bufferSizeEmpty() {
        #expect(Memory.bufferSize([MLXArray]()) == 0)
    }

    @Test func bufferSizeCountsDuplicatesOnce() {
        let a = MLXRandom.normal([64, 32])
        eval(a)

        #expect(Memory.bufferSize([a, a, a]) == Memory.bufferSize([a]))
    }

    @Test func bufferSizeCountsViewsOnce() {
        let a = MLXRandom.normal([64, 32])
        let slice = a[0 ..< 8]
        let transposed = a.T
        eval(a, slice, transposed)

        // views share the buffer of `a`
        let size = Memory.bufferSize([a])
        #expect(Memory.bufferSize([a, slice]) == size)
        #expect(Memory.bufferSize([a, transposed]) == size)
        #expect(Memory.bufferSize([slice, transposed]) == size)
    }

    @Test func bufferSizeSumsDistinctBuffers() {
        let a = MLXRandom.normal([64, 32])
        let b = MLXRandom.normal([5])
        let c = MLXRandom.randInt(low: 0, high: 4, [100], type: UInt32.self)
        eval(a, b, c)

        #expect(
            Memory.bufferSize([a, b, c]) == Memory.bufferSize([a]) + Memory.bufferSize([b])
                + Memory.bufferSize([c]))
    }
}
