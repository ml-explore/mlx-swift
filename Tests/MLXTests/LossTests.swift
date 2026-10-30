// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@Suite
struct LossTests {

    @Test func crossEntropyIndicesAndProbabilities() {
        // This is just testing that crossEntropy supports both class indices and class
        // probabilities as targets.

        // test class indices
        let logits = zeros([2, 2])
        logits[0, 1] = MLXArray(-Float.infinity)
        logits[1, 0] = MLXArray(-Float.infinity)
        let indices = MLXArray([0, 1])
        let expected = MLXArray(converting: [0.0, 0.0])
        let loss = crossEntropy(logits: logits, targets: indices, reduction: .none)
        expectEqual(loss, expected)

        // test class probabilities
        let probs = zeros([2, 2])
        probs[0, 0] = MLXArray(1)
        probs[1, 1] = MLXArray(1)
        let loss2 = crossEntropy(logits: logits, targets: probs, reduction: .none)
        #expect(all(isNaN(loss2)).item(Bool.self))
    }

    @Test func crossEntropyAxis() {
        // The class axis need not be the last one. Expected values from Python mlx:
        // nn.losses.cross_entropy(logits, targets, axis=0) -> [0.169846, 0.169846].
        let logits = MLXArray(converting: [2.0, -1.0, -1.0, 2.0, 0.0, 0.0], [3, 2])
        let targets = MLXArray([0, 1])
        let expected = MLXArray(converting: [0.169846, 0.169846])
        let loss = crossEntropy(logits: logits, targets: targets, axis: 0, reduction: .none)
        #expect(loss.shape == [2])
        expectEqual(loss, expected, atol: 1e-5)
    }

    @Test func nllLossAxis() {
        // nn.losses.nll_loss(inputs, targets, axis=0) -> [-2, -2]
        let inputs = MLXArray(converting: [2.0, -1.0, -1.0, 2.0, 0.0, 0.0], [3, 2])
        let targets = MLXArray([0, 1])
        let expected = MLXArray(converting: [-2.0, -2.0])
        let loss = nllLoss(inputs: inputs, targets: targets, axis: 0, reduction: .none)
        #expect(loss.shape == [2])
        expectEqual(loss, expected)
    }

    @Test func tripletLossReductionsAndNorms() {
        // This mainly tests that tripletLoss now works for arbitrary values of p

        let anchors = MLXArray([1, 2, 3, 1, 2, 3], [2, 3])
        let positives = MLXArray([4, 5, 6, 0, -1, 2], [2, 3])
        let negatives = MLXArray([7, 8, 9, 3, 2, 3], [2, 3])

        // Test with reduction 'none'
        let lossesNone = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .none)
        let expectedNone = MLXArray(converting: [0.0, 2.31662])
        expectEqual(lossesNone, expectedNone)

        // Test with reduction 'mean'
        let lossesMean = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .mean)
        let expectedMean = mean(expectedNone)
        expectEqual(lossesMean, expectedMean)

        // Test with reduction 'sum'
        let lossesSum = tripletLoss(
            anchors: anchors, positives: positives, negatives: negatives, reduction: .sum)
        let expectedSum = sum(expectedNone)
        expectEqual(lossesSum, expectedSum)

        // Test with non-default 'p' norm degrees
        let normAnchors = MLXArray(converting: [1.0, 2.0]).reshaped([1, 2])
        let normPositives = MLXArray(converting: [4.0, 6.0]).reshaped([1, 2])
        let normNegatives = MLXArray(converting: [2.0, 3.0]).reshaped([1, 2])

        let lossesP1 = tripletLoss(
            anchors: normAnchors, positives: normPositives, negatives: normNegatives,
            p: 1, reduction: .none)
        let expectedP1 = MLXArray(converting: [6.0])
        expectEqual(lossesP1, expectedP1)

        let lossesP3 = tripletLoss(
            anchors: normAnchors, positives: normPositives, negatives: normNegatives,
            p: 3, reduction: .none)
        let expectedP3 = MLXArray(converting: [4.23802])
        expectEqual(lossesP3, expectedP3)
    }
}
