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
}
