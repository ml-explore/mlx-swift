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

    func testBufferSize() {
        let a = MLXArray([1, 2, 3, 4] as [Float])
        let view = a[..<1]
        eval(a, view)

        let size = Memory.bufferSize(of: a)

        // the allocator size is at least the logical size
        XCTAssertGreaterThanOrEqual(size, a.nbytes)

        // no arrays, no memory
        XCTAssertEqual(Memory.bufferSize(of: [MLXArray]()), 0)

        // each unique buffer is counted once -- the view shares a's buffer
        XCTAssertEqual(Memory.bufferSize(of: view), size)
        XCTAssertEqual(Memory.bufferSize(of: a, a), size)
        XCTAssertEqual(Memory.bufferSize(of: a, view), size)

        let b = MLXArray([5, 6] as [Float])
        eval(b)
        XCTAssertEqual(Memory.bufferSize(of: a, b), size + Memory.bufferSize(of: b))
    }

    func testBufferSizeUnevaluated() throws {
        let a = MLXArray([1, 2, 3, 4] as [Float])
        eval(a)

        // an unevaluated array has no buffer to measure
        let lazy = a + 1
        XCTAssertThrowsError(try withError { _ in Memory.bufferSize(of: lazy) })
    }
}
