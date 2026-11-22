// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

/// Tests for ``checkpoint(_:)-(([MLXArray])->[MLXArray])``.
///
/// Mirrors `test checkpointing` in `mlx/tests/custom_vjp_tests.cpp`: the
/// checkpointed function must produce the same values and gradients as the
/// original, and the forward function must run a second time during the
/// backward pass (that recomputation is the whole point of checkpointing).
@Suite(.serialized)
struct CheckpointTests {

    /// Counts calls to the wrapped function.  `checkpoint` may invoke the
    /// closure from inside MLX's transform machinery, so the counter is lock
    /// protected rather than a captured `var`.
    private final class CallCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        func increment() { lock.withLock { value += 1 } }
        var count: Int { lock.withLock { value } }
    }

    /// f(x, y) = (x * y + x + y)^2 -- at x = 2, y = 3: value 121, df/dx 88, df/dy 66
    private static func function(counter: CallCounter) -> ([MLXArray]) -> [MLXArray] {
        { inputs in
            counter.increment()
            let x = inputs[0] * inputs[1]
            let y = inputs[0] + inputs[1]
            return [square(x + y)]
        }
    }

    @Test func forwardMatchesOriginal() {
        let counter = CallCounter()
        let f = Self.function(counter: counter)
        let checkpointed = checkpoint(f)

        let x = MLXArray(2.0)
        let y = MLXArray(3.0)

        let expected = f([x, y])
        let actual = checkpointed([x, y])

        #expect(actual.count == 1)
        #expect(actual[0].item(Float.self) == 121)
        #expect(actual[0].item(Float.self) == expected[0].item(Float.self))
    }

    @Test func vjpRecomputesForward() {
        let counter = CallCounter()
        let checkpointed = checkpoint(Self.function(counter: counter))

        let x = MLXArray(2.0)
        let y = MLXArray(3.0)
        let one = MLXArray(1.0)

        let (value, gradients) = vjp(checkpointed, primals: [x, y], cotangents: [one])
        eval(value, gradients)

        #expect(value[0].item(Float.self) == 121)
        #expect(gradients.count == 2)
        #expect(gradients[0].item(Float.self) == 88)
        #expect(gradients[1].item(Float.self) == 66)

        // once for the forward pass, once more to recompute during the backward pass
        #expect(counter.count == 2)
    }

    @Test func gradMatchesUncheckpointed() {
        let counter = CallCounter()
        let f = Self.function(counter: counter)

        let x = MLXArray(2.0)
        let y = MLXArray(3.0)

        let g = grad(f, argumentNumbers: [0, 1])
        let gCheckpointed = grad(checkpoint(f), argumentNumbers: [0, 1])

        let expected = g([x, y])
        let actual = gCheckpointed([x, y])

        #expect(actual.count == expected.count)
        for (a, e) in zip(actual, expected) {
            #expect(a.item(Float.self) == e.item(Float.self))
        }
    }

    @Test func unaryOverload() {
        let counter = CallCounter()
        let checkpointed = checkpoint { (x: MLXArray) -> MLXArray in
            counter.increment()
            return (x * x).sum()
        }

        // d/dx sum(x^2) = 2x
        let x = MLXArray([1, 2, 3] as [Float])
        let value = checkpointed(x)
        let gradient = grad(checkpointed)(x)
        eval(value, gradient)

        #expect(value.item(Float.self) == 14)
        #expect(gradient.asArray(Float.self) == [2, 4, 6])

        // forward call, then forward + recompute inside grad
        #expect(counter.count == 3)
    }

    @Test func returnedFunctionIsReusable() {
        let counter = CallCounter()
        let checkpointed = checkpoint(Self.function(counter: counter))

        for i in 1 ... 3 {
            let v = Float(i)
            let result = checkpointed([MLXArray(v), MLXArray(v)])
            // (v*v + 2v)^2
            let expected = (v * v + 2 * v) * (v * v + 2 * v)
            #expect(result[0].item(Float.self) == expected)
        }
        #expect(counter.count == 3)
    }
}
