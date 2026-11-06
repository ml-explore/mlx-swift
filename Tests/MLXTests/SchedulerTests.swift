// Copyright © 2024 Apple Inc.

import Foundation
import Testing

@testable import MLXOptimizers

@Suite
struct SchedulerTests {

    @Test func exponentialDecaySchedule() {
        let s = exponentialDecay(0.1, decayRate: 0.9)
        expectEqual(s(0), 0.1, accuracy: 1e-6)
        expectEqual(s(1), 0.09, accuracy: 1e-6)
        expectEqual(s(2), 0.081, accuracy: 1e-6)
        expectEqual(s(5), 0.1 * 0.9 * 0.9 * 0.9 * 0.9 * 0.9, accuracy: 1e-6)
    }

    @Test func stepDecaySchedule() {
        let s = stepDecay(0.1, decayRate: 0.9, stepSize: 10)
        expectEqual(s(0), 0.1, accuracy: 1e-6)
        expectEqual(s(9), 0.1, accuracy: 1e-6)  // constant within a step window
        expectEqual(s(10), 0.09, accuracy: 1e-6)
        expectEqual(s(21), 0.081, accuracy: 1e-6)  // 0.1 * 0.9^2
    }

    @Test func cosineDecaySchedule() {
        let s = cosineDecay(0.1, decaySteps: 1000)
        expectEqual(s(0), 0.1, accuracy: 1e-6)
        expectEqual(s(500), 0.05, accuracy: 1e-6)  // half-way: 0.5*(1+cos(pi/2))*0.1
        expectEqual(s(1000), 0.0, accuracy: 1e-6)
        expectEqual(s(5000), 0.0, accuracy: 1e-6)  // clamped beyond decaySteps

        // non-zero end value
        let s2 = cosineDecay(0.1, decaySteps: 100, end: 0.01)
        expectEqual(s2(0), 0.1, accuracy: 1e-6)
        expectEqual(s2(100), 0.01, accuracy: 1e-6)
    }

    @Test func linearRampSchedule() {
        let s = linearSchedule(0.0, end: 0.1, steps: 100)
        expectEqual(s(0), 0.0, accuracy: 1e-6)
        expectEqual(s(50), 0.05, accuracy: 1e-6)
        expectEqual(s(100), 0.1, accuracy: 1e-6)
        expectEqual(s(200), 0.1, accuracy: 1e-6)  // clamped beyond steps
    }

    @Test func joinedSchedules() {
        // warm-up linearly for 10 steps, then cosine-decay.
        let linear = linearSchedule(0.0, end: 0.1, steps: 10)
        let cosine = cosineDecay(0.1, decaySteps: 200)
        let s = joinSchedules([linear, cosine], boundaries: [10])

        expectEqual(s(0), 0.0, accuracy: 1e-6)  // linear leg
        expectEqual(s(5), 0.05, accuracy: 1e-6)  // linear leg
        expectEqual(s(10), 0.1, accuracy: 1e-6)  // cosine leg at its step 0
        // at step 11 we are 1 step into the cosine schedule
        expectEqual(s(11), cosine(1), accuracy: 1e-6)
    }
}
