// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

/// Tests for layers whose formulas are easy to get subtly wrong -- these pin down
/// the parts that a shape check would not catch.
class PositionalEncodingTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - ALiBi

    func testALiBiSlopesPowerOfTwo() {
        // matches python `ALiBi.create_alibi_slope`: 2^(-8i/n) for i in 1...n
        let slopes = ALiBi.alibiSlope(numHeads: 4).reshaped([-1])
        assertEqual(
            slopes, MLXArray(converting: [0.25, 0.0625, 0.015625, 0.00390625]), atol: 1e-7)
    }

    func testALiBiSlopesNotPowerOfTwo() {
        // python does *not* extend the geometric series: it takes the slopes of the
        // power of two below and pads with every other slope of the one above
        assertEqual(
            ALiBi.alibiSlope(numHeads: 3).reshaped([-1]),
            MLXArray(converting: [0.0625, 0.00390625, 0.25]), atol: 1e-7)
        assertEqual(
            ALiBi.alibiSlope(numHeads: 6).reshaped([-1]),
            MLXArray(converting: [0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125]),
            atol: 1e-7)
    }

    func testALiBiMaskIsADistanceMatrix() {
        // the mask must be -|i - j| * slope: a (q, k) matrix, not a broadcast of a
        // single column (which collapsed to all zeros when q == k)
        let scores = MLXArray.zeros([1, 2, 4, 4])
        let mask = ALiBi()(attentionScores: scores)

        XCTAssertEqual(mask.shape, [1, 2, 4, 4])

        let slopes: [Float] = [0.0625, 0.00390625]  // numHeads == 2
        for head in 0 ..< 2 {
            for q in 0 ..< 4 {
                for k in 0 ..< 4 {
                    let expected = -Float(abs(q - k)) * slopes[head]
                    XCTAssertEqual(
                        mask[0, head, q, k].item(Float.self), expected, accuracy: 1e-6,
                        "head \(head), q \(q), k \(k)")
                }
            }
        }
    }

    func testALiBiOffsetShiftsTheQueryPositions() {
        // q positions start at `offset`, so with offset 1 and 3 queries the
        // distances are |[1, 2, 3] - [0, 1, 2, 3]|
        let scores = MLXArray.zeros([1, 1, 3, 4])
        let mask = ALiBi()(attentionScores: scores, offset: 1)

        XCTAssertEqual(mask.shape, [1, 1, 3, 4])

        let slope: Float = 0.00390625  // numHeads == 1
        for q in 0 ..< 3 {
            for k in 0 ..< 4 {
                let expected = -Float(abs((q + 1) - k)) * slope
                XCTAssertEqual(
                    mask[0, 0, q, k].item(Float.self), expected, accuracy: 1e-6,
                    "q \(q), k \(k)")
            }
        }
    }

    func testALiBiAddsToTheScores() {
        let scores = MLXArray.ones([1, 1, 3, 3])
        let mask = ALiBi()(attentionScores: MLXArray.zeros([1, 1, 3, 3]))
        assertEqual(ALiBi()(attentionScores: scores), scores + mask)
    }
}

/// Recurrent layers: the formulas have branches that only run for some argument
/// combinations, which is where they drift from python.
class RecurrentTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func parameters(_ module: Module) -> ModuleParameters {
        // deterministic values so the tests do not depend on the random init
        module.mapParameters { parameter in
            let size = parameter.size
            return MLX.arange(size, dtype: .float32).reshaped(parameter.shape) / Float(size)
                - 0.5
        }
    }

    func testGRUAppliesHiddenBiasWithoutAnIncomingHiddenState() {
        // python applies `r * bhn` even when no hidden state is passed in; when
        // this was missing, `bhn` had no effect on the first call at all
        let gru = GRU(inputSize: 4, hiddenSize: 3)
        gru.update(parameters: parameters(gru))

        let x = MLX.arange(2 * 5 * 4, dtype: .float32).reshaped([2, 5, 4]) / 40

        let before = gru(x)

        gru.update(parameters: ModuleParameters.unflattened([("bhn", MLXArray.ones([3]))]))
        let after = gru(x)

        assertNotEqual(before, after)
    }

    func testGRUMatchesAReferenceImplementation() {
        let hiddenSize = 3
        let gru = GRU(inputSize: 4, hiddenSize: hiddenSize)
        gru.update(parameters: parameters(gru))

        let x = MLX.arange(1 * 2 * 4, dtype: .float32).reshaped([1, 2, 4]) / 8
        let result = gru(x)

        // the same formula written out, with no incoming hidden state
        let wx = gru.wx
        let wh = gru.wh
        let b = gru.b!
        let bhn = gru.bhn!

        let projected = addMM(b, x, wx.T)
        var hidden: MLXArray? = nil
        var steps = [MLXArray]()

        for step in 0 ..< x.dim(-2) {
            var rz = projected[.ellipsis, step, ..<(2 * hiddenSize)]
            var hiddenN: MLXArray? = nil

            if let hidden {
                let projectedHidden = matmul(hidden, wh.T)
                rz = rz + projectedHidden[.ellipsis, ..<(2 * hiddenSize)]
                hiddenN = projectedHidden[.ellipsis, (2 * hiddenSize)...] + bhn
            }

            rz = sigmoid(rz)
            let r = rz[.ellipsis, ..<hiddenSize]
            let z = rz[.ellipsis, hiddenSize...]

            var n = projected[.ellipsis, step, (2 * hiddenSize)...]
            n = n + r * (hiddenN ?? bhn)
            n = tanh(n)

            if let previous = hidden {
                hidden = (1 - z) * n + z * previous
            } else {
                hidden = (1 - z) * n
            }
            steps.append(hidden!)
        }

        assertEqual(result, stacked(steps, axis: -2), atol: 1e-6)
    }

    func testRNNAndGRUShapes() {
        let x = MLXArray.zeros([2, 5, 4])

        XCTAssertEqual(RNN(inputSize: 4, hiddenSize: 3)(x).shape, [2, 5, 3])
        XCTAssertEqual(GRU(inputSize: 4, hiddenSize: 3)(x).shape, [2, 5, 3])

        let (hidden, cell) = LSTM(inputSize: 4, hiddenSize: 3)(x)
        XCTAssertEqual(hidden.shape, [2, 5, 3])
        XCTAssertEqual(cell.shape, [2, 5, 3])
    }
}
