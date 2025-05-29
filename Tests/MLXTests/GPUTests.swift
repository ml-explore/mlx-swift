// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class GPUTests: XCTestCase {

    func testWiredMemory() {
        GPU.withWiredLimit(1024 * 1024 * 256) {
            let x = MLXArray(10)
            print(x * x)
        }
    }
}
