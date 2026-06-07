// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class MLXNNPositionalEncodingTests: XCTestCase {
    func testALiBiMatrixIsRelativeDistance() {
        // With a single head the slope is 256**-1, so the bias added to the
        // attention scores is the relative-distance matrix -(|i - j|) / 256.
        Stream.withNewDefaultStream(device: .cpu) {
            let q = 4
            let k = 4
            let attentionScores = MLXArray.zeros([1, 1, q, k])
            let output = ALiBi().callAsFunction(attentionScores: attentionScores)

            let slope = 1.0 / 256.0
            var expectedValues = [Double]()
            for i in 0 ..< q {
                for j in 0 ..< k {
                    expectedValues.append(-Double(abs(i - j)) * slope)
                }
            }
            let expected = MLXArray(converting: expectedValues, [1, 1, q, k])

            assertEqual(output, expected)
        }
    }

    func testALiBiSupportsDifferentQueryAndKeyLengths() {
        // The query and key sequence lengths differ, so the distance matrix must
        // be a proper (q, k) outer difference rather than an elementwise one.
        Stream.withNewDefaultStream(device: .cpu) {
            let attentionScores = MLXArray.zeros([1, 1, 2, 3])
            let output = ALiBi().callAsFunction(attentionScores: attentionScores)

            XCTAssertEqual(output.shape, [1, 1, 2, 3])
        }
    }
}
