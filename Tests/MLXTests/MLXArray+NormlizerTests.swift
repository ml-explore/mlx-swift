// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX

public final class MLXArrayL2NormalizationTests: XCTestCase {

    /// Tests standard L2 normalization for a 1D vector.
    /// Magnitude is exactly 5.0, result should be unit length (1.0).
    func testL2NormalizationStandard() {
        let rawArray: [Float] = [3.0, 4.0]
        let array = MLXArray(rawArray, [2])
        let normalized = array.l2Normalized()

        let rawExpected: [Float] = [0.6, 0.8]
        let expected = MLXArray(rawExpected, [2])

        // Use allClose for floating point comparison in MLX
        XCTAssertTrue(allClose(normalized, expected).item(Bool.self))

        // Verify Magnitude: Must be 1.0
        let magnitude = MLXLinalg.norm(normalized, ord: 2).item(Float.self)
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-6)
    }

    /// Tests normalization along a specific axis in a 2D matrix.
    func testL2NormalizationAlongAxis() {
        // 2x2 Matrix: [[3, 4], [0, 1]]
        let rawArray: [Float] = [3.0, 4.0, 0.0, 1.0]
        let array = MLXArray(rawArray, [2, 2])

        // Normalize along the last axis (rows)
        let normalized = array.l2Normalized(axis: -1)

        // Row 1: [0.6, 0.8], Row 2: [0.0, 1.0]
        let rawExpected: [Float] = [0.6, 0.8, 0.0, 1.0]
        let expected = MLXArray(rawExpected, [2, 2])

        XCTAssertTrue(allClose(normalized, expected).item(Bool.self))
    }

    /// CRITICAL: Tests behavior with a zero vector to ensure numerical stability via epsilon.
    func testL2NormalizationZeroVector() {
        let eps: Float = 1e-8
        let rawArray: [Float] = [0.0, 0.0]
        let array = MLXArray(rawArray, [2])
        let normalized = array.l2Normalized(eps: eps)

        // Since Norm (0) < eps, we divide by eps.
        // 0 / eps remains 0, preventing NaN.
        let rawExpected: [Float] = [0.0, 0.0]
        let expected = MLXArray(rawExpected, [2])

        XCTAssertTrue(allClose(normalized, expected).item(Bool.self))

        // Magnitude should be 0.0, not NaN!
        let magnitude = MLXLinalg.norm(normalized, ord: 2).item(Float.self)
        XCTAssertFalse(magnitude.isNaN, "Resulting magnitude should not be NaN")
        XCTAssertEqual(magnitude, 0.0)
    }

    /// Tests values that are smaller than the provided epsilon.
    func testL2NormalizationUnderEpsilon() {
        let eps: Float = 1e-3
        let rawArray: [Float] = [1e-5, 1e-5]
        let array = MLXArray(rawArray, [2])  // Norm is approx 1.41 * 1e-5
        let normalized = array.l2Normalized(eps: eps)

        // The norm is smaller than eps, so the divisor is clamped to eps (0.001)
        let expectedValue = Float(1e-5) / eps
        let expected = MLXArray([expectedValue, expectedValue], [2])

        XCTAssertTrue(allClose(normalized, expected).item(Bool.self))
    }

}
