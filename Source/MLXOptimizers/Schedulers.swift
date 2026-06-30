// Copyright © 2024 Apple Inc.

import Foundation

// Learning-rate schedules. Ports of `mlx.optimizers.schedulers`.
//
// Each function returns a closure mapping a step count to a learning rate.
// Use it by assigning the optimizer's `learningRate` each step, e.g.
//
// ```swift
// let schedule = cosineDecay(1e-1, decaySteps: 1000)
// let optimizer = SGD(learningRate: schedule(0))
// for step in 0 ..< 1000 {
//     optimizer.learningRate = schedule(step)
//     optimizer.update(model: model, gradients: grads)
// }
// ```

/// Make an exponential-decay schedule: `initial * decayRate ** step`.
///
/// - Parameters:
///   - initial: initial value
///   - decayRate: multiplicative factor to decay by each step
public func exponentialDecay(_ initial: Float, decayRate: Float) -> (Int) -> Float {
    { step in initial * pow(decayRate, Float(step)) }
}

/// Make a step-decay schedule: `initial * decayRate ** (step / stepSize)`.
///
/// - Parameters:
///   - initial: initial value
///   - decayRate: multiplicative factor to decay by
///   - stepSize: decay every `stepSize` steps
public func stepDecay(_ initial: Float, decayRate: Float, stepSize: Int) -> (Int) -> Float {
    { step in initial * pow(decayRate, Float(step / stepSize)) }
}

/// Make a cosine-decay schedule. The value is constant (`end`) for steps beyond
/// `decaySteps`.
///
/// - Parameters:
///   - initial: initial value
///   - decaySteps: number of steps to decay over
///   - end: final value to decay to (default 0)
public func cosineDecay(_ initial: Float, decaySteps: Int, end: Float = 0.0) -> (Int) -> Float {
    { step in
        let s = Float(min(step, decaySteps))
        let decay = 0.5 * (1.0 + cos((Float.pi / Float(decaySteps)) * s))
        return end + decay * (initial - end)
    }
}

/// Make a linear schedule from `initial` to `end` over `steps`. The value is
/// `end` for any steps beyond `steps`.
///
/// - Parameters:
///   - initial: initial value
///   - end: final value
///   - steps: number of steps to apply the schedule over (must be >= 1)
public func linearSchedule(_ initial: Float, end: Float, steps: Int) -> (Int) -> Float {
    precondition(steps >= 1, "steps must be greater than 0, but got \(steps).")
    return { step in
        let s = Float(min(step, steps))
        return s * ((end - initial) / Float(steps)) + initial
    }
}

/// Join multiple schedules into one. Schedule `i + 1` receives a step count
/// indicating the number of steps since the `i`-th boundary.
///
/// - Parameters:
///   - schedules: the schedules to join
///   - boundaries: `schedules.count - 1` step counts marking transitions
public func joinSchedules(_ schedules: [(Int) -> Float], boundaries: [Int]) -> (Int) -> Float {
    precondition(!schedules.isEmpty, "Must provide at least 1 schedule to join.")
    precondition(
        schedules.count == boundaries.count + 1,
        "Received \(boundaries.count) boundaries but expected \(schedules.count - 1).")
    return { step in
        var output = schedules[0](step)
        for (boundary, schedule) in zip(boundaries, schedules.dropFirst()) where step >= boundary {
            output = schedule(step - boundary)
        }
        return output
    }
}
