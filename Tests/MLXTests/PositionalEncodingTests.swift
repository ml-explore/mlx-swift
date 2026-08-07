// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class PositionalEncodingTests: XCTestCase {

    /// Exercises `ALiBi`, which is the only consumer of the internal `Cache` type
    /// (`Source/MLXNN/Cache.swift`). Calling it twice with identical shape/dtype/offset
    /// hits the cached path on the second call; this validates the Mutex-backed cache
    /// still returns a correct, consistent result.
    func testALiBiCaching() {
        let alibi = ALiBi()
        let scores = MLXArray.zeros([1, 4, 5, 5])

        let first = alibi(attentionScores: scores)
        let second = alibi(attentionScores: scores)

        XCTAssertEqual(first.shape, [1, 4, 5, 5])
        XCTAssertTrue(allClose(first, second).all().item())
    }
}
