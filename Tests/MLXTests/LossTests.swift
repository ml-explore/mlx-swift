// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class LossTests: XCTestCase {
    override class func setUp() {
        setDefaultDevice()
    }

    func testCrossEntropy() {
        // This is just testing that crossEntropy supports both class indices and class
        // probabilities as targets.

        // test class indices
        let logits = zeros([2, 2])
        logits[0, 1] = MLXArray(-Float.infinity)
        logits[1, 0] = MLXArray(-Float.infinity)
        let indices = MLXArray([0, 1])
        let expected = MLXArray(converting: [0.0, 0.0])
        let loss = crossEntropy(logits: logits, targets: indices, reduction: .none)
        XCTAssertTrue(allClose(loss, expected).item())

        // test class probabilities
        let probs = zeros([2, 2])
        probs[0, 0] = MLXArray(1)
        probs[1, 1] = MLXArray(1)
        let loss2 = crossEntropy(logits: logits, targets: probs, reduction: .none)
        XCTAssertTrue(all(isNaN(loss2)).item())
    }
}
