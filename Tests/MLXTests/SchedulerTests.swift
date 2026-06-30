// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLXOptimizers

class SchedulerTests: XCTestCase {

    func testExponentialDecay() {
        let s = exponentialDecay(0.1, decayRate: 0.9)
        XCTAssertEqual(s(0), 0.1, accuracy: 1e-6)
        XCTAssertEqual(s(1), 0.09, accuracy: 1e-6)
        XCTAssertEqual(s(2), 0.081, accuracy: 1e-6)
        XCTAssertEqual(s(5), 0.1 * 0.9 * 0.9 * 0.9 * 0.9 * 0.9, accuracy: 1e-6)
    }

    func testStepDecay() {
        let s = stepDecay(0.1, decayRate: 0.9, stepSize: 10)
        XCTAssertEqual(s(0), 0.1, accuracy: 1e-6)
        XCTAssertEqual(s(9), 0.1, accuracy: 1e-6)  // constant within a step window
        XCTAssertEqual(s(10), 0.09, accuracy: 1e-6)
        XCTAssertEqual(s(21), 0.081, accuracy: 1e-6)  // 0.1 * 0.9^2
    }

    func testCosineDecay() {
        let s = cosineDecay(0.1, decaySteps: 1000)
        XCTAssertEqual(s(0), 0.1, accuracy: 1e-6)
        XCTAssertEqual(s(500), 0.05, accuracy: 1e-6)  // half-way: 0.5*(1+cos(pi/2))*0.1
        XCTAssertEqual(s(1000), 0.0, accuracy: 1e-6)
        XCTAssertEqual(s(5000), 0.0, accuracy: 1e-6)  // clamped beyond decaySteps

        // non-zero end value
        let s2 = cosineDecay(0.1, decaySteps: 100, end: 0.01)
        XCTAssertEqual(s2(0), 0.1, accuracy: 1e-6)
        XCTAssertEqual(s2(100), 0.01, accuracy: 1e-6)
    }

    func testLinearSchedule() {
        let s = linearSchedule(0.0, end: 0.1, steps: 100)
        XCTAssertEqual(s(0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(s(50), 0.05, accuracy: 1e-6)
        XCTAssertEqual(s(100), 0.1, accuracy: 1e-6)
        XCTAssertEqual(s(200), 0.1, accuracy: 1e-6)  // clamped beyond steps
    }

    func testJoinSchedules() {
        // warm-up linearly for 10 steps, then cosine-decay.
        let linear = linearSchedule(0.0, end: 0.1, steps: 10)
        let cosine = cosineDecay(0.1, decaySteps: 200)
        let s = joinSchedules([linear, cosine], boundaries: [10])

        XCTAssertEqual(s(0), 0.0, accuracy: 1e-6)  // linear leg
        XCTAssertEqual(s(5), 0.05, accuracy: 1e-6)  // linear leg
        XCTAssertEqual(s(10), 0.1, accuracy: 1e-6)  // cosine leg at its step 0
        // at step 11 we are 1 step into the cosine schedule
        XCTAssertEqual(s(11), cosine(1), accuracy: 1e-6)
    }
}
