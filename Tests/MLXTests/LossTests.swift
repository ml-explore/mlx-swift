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

    func testTripletLoss() {
        // This mainly tests that tripletLoss now works for arbitrary values of p

        let anchors = MLXArray([1, 2, 3, 1, 2, 3], [2, 3])
        let positives = MLXArray([4, 5, 6, 0, -1, 2], [2, 3])
        let negatives = MLXArray([7, 8, 9, 3, 2, 3], [2, 3])

        // Test with reduction 'none'
        let lossesNone = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .none)
        let expectedNone = MLXArray(converting: [0.0, 2.31662])
        XCTAssertTrue(allClose(lossesNone, expectedNone).item())

        // Test with reduction 'mean'
        let lossesMean = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .mean)
        let expectedMean = mean(expectedNone)
        XCTAssertTrue(allClose(lossesMean, expectedMean).item())

        // Test with reduction 'sum'
        let lossesSum = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .sum)
        let expectedSum = sum(expectedNone)
        XCTAssertTrue(allClose(lossesSum, expectedSum).item())

        // Test with non-default 'p' norm degrees
        let normAnchors = MLXArray(converting: [1.0, 2.0]).reshaped([1, 2])
        let normPositives = MLXArray(converting: [4.0, 6.0]).reshaped([1, 2])
        let normNegatives = MLXArray(converting: [2.0, 3.0]).reshaped([1, 2])

        let lossesP1 = tripletLoss(
            anchors: normAnchors, positives: normPositives, negatives: normNegatives,
            p: 1, reduction: .none)
        let expectedP1 = MLXArray(converting: [6.0])
        XCTAssertTrue(allClose(lossesP1, expectedP1).item())

        let lossesP3 = tripletLoss(
            anchors: normAnchors, positives: normPositives, negatives: normNegatives,
            p: 3, reduction: .none)
        let expectedP3 = MLXArray(converting: [4.23802])
        XCTAssertTrue(allClose(lossesP3, expectedP3).item())
    }
}
