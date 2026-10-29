// Copyright © 2026 Apple Inc.

import MLX
import Testing

// Swift Testing counterparts to the XCTest helpers in `Utils.swift`.
//
// `XCTAssert*` is not a reliable way to fail a Swift Testing `@Test`, so
// tests converted to Swift Testing should use these instead.

/// Prefix `detail` with the caller supplied `comment`, if any.
private func message(_ comment: Comment?, _ detail: String) -> Comment {
    if let comment {
        Comment(rawValue: "\(comment.rawValue)\n\(detail)")
    } else {
        Comment(rawValue: detail)
    }
}

/// Expect two arrays to have the same shape and contents.
///
/// Shape is checked explicitly because `allClose` broadcasts -- e.g. `[2]`
/// would otherwise compare equal to `[[2, 2], [2, 2]]`.
///
/// - Returns: `true` if the expectation passed (an issue is recorded otherwise)
@discardableResult
func expectEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation
) -> Bool {
    guard array1.shape == array2.shape else {
        Issue.record(
            message(
                comment, "shapes differ: \(array1.shape) != \(array2.shape)\n\(array1)\n\(array2)"),
            sourceLocation: sourceLocation)
        return false
    }
    if !array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self) {
        Issue.record(
            message(comment, "contents differ:\n\(array1)\n\(array2)"),
            sourceLocation: sourceLocation)
        return false
    }
    return true
}

/// Expect two lists of arrays to be pairwise equal (see ``expectEqual(_:_:rtol:atol:_:sourceLocation:)``).
@discardableResult
func expectEqual(
    _ array1: [MLXArray], _ array2: [MLXArray], rtol: Double = 1e-5, atol: Double = 1e-8,
    _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation
) -> Bool {
    guard array1.count == array2.count else {
        Issue.record(
            message(comment, "counts differ: \(array1.count) != \(array2.count)"),
            sourceLocation: sourceLocation)
        return false
    }
    var result = true
    for (e1, e2) in zip(array1, array2) {
        result =
            expectEqual(e1, e2, rtol: rtol, atol: atol, comment, sourceLocation: sourceLocation)
            && result
    }
    return result
}

/// Expect two arrays to have the same shape but different contents.
@discardableResult
func expectNotEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation
) -> Bool {
    guard array1.shape == array2.shape else {
        Issue.record(
            message(comment, "shapes differ: \(array1.shape) != \(array2.shape)"),
            sourceLocation: sourceLocation)
        return false
    }
    if array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self) {
        Issue.record(
            message(comment, "contents same:\n\(array1)\n\(array2)"),
            sourceLocation: sourceLocation)
        return false
    }
    return true
}

/// Expect two floating point scalars to be within `accuracy` of each other.
///
/// Swift Testing counterpart to `XCTAssertEqual(_:_:accuracy:)`.
///
/// - Returns: `true` if the expectation passed (an issue is recorded otherwise)
@discardableResult
func expectEqual<T: FloatingPoint>(
    _ value1: T, _ value2: T, accuracy: T,
    _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation
) -> Bool {
    // written so that NaN fails rather than passes
    guard abs(value1 - value2) <= accuracy else {
        Issue.record(
            message(comment, "\(value1) is not within \(accuracy) of \(value2)"),
            sourceLocation: sourceLocation)
        return false
    }
    return true
}

// MARK: - Random state

/// Run a test with a task-local random state seeded with `seed`, e.g.:
///
/// ```swift
/// @Test(.randomSeed(7)) func somethingRandom() {
///     // draws from the task-local state, as if MLXRandom.seed(7) had been called
///     let a = MLXRandom.normal([2, 2])
/// }
/// ```
///
/// Use this instead of `MLXRandom.seed(_:)`.  The global random state is shared by
/// every test in the process and Swift Testing runs tests in parallel, so another
/// test can draw from it between the `seed` and the values a test depends on.
///
/// `MLXRandom.RandomState(seed: n)` starts from the same key as `MLXRandom.seed(n)`,
/// so expected values computed with a global seed stay valid.  The state is a
/// task local (see `withRandomState`) so it covers the whole test body, including
/// child tasks and anything that draws implicitly, e.g. `Linear` initialization.
struct RandomSeedTrait: TestTrait, TestScoping {
    let seed: UInt64

    func provideScope(
        for test: Test, testCase: Test.Case?,
        performing function: @Sendable () async throws -> Void
    ) async throws {
        try await withRandomState(MLXRandom.RandomState(seed: seed)) {
            try await function()
        }
    }
}

extension Trait where Self == RandomSeedTrait {
    /// See ``RandomSeedTrait``.
    static func randomSeed(_ seed: UInt64) -> Self {
        RandomSeedTrait(seed: seed)
    }
}
