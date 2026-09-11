// Copyright © 2025 Apple Inc.

// Hand written support for the generated integration tests -- NOT generated.
//
// The generated tests (see tools/integration_tests) declare their inputs with a
// seeded random state, evaluate one expression, and compare the result against a
// summary of the value python produced.  All of the comparison policy lives here
// so it can be tuned without regenerating.

import Foundation
import MLX
import MLXNN
import Testing

/// Statistics of an `MLXArray`, computed by python `mlx` in
/// `tools/integration_tests/core.py` (`summarize`) and recomputed here.
///
/// The definitions must match the generator exactly:
///
/// ```
/// values = array.astype(float32).reshape(-1)
/// absoluteSum = sum(|values|)
/// positionChecksum = sum(|values| * arange(1, n + 1)) / n
/// samples = values[evenly spaced indices]
/// ```
///
/// `positionChecksum` and `samples` are what make this order sensitive: `mean`,
/// `min`, `max` and `absoluteSum` are all invariant under a permutation of the
/// elements, so a wrong axis or a transposed result would otherwise pass.
struct ArraySummary: Sendable {
    let shape: [Int]
    let dtype: DType
    let mean: Double
    let minimum: Double
    let maximum: Double
    let absoluteSum: Double
    let positionChecksum: Double
    let sampleIndices: [Int]
    let samples: [Double]
}

/// Comparison tolerance: `|actual - expected| <= absolute + relative * |expected|`.
struct Tolerance: Sendable {
    let relative: Double
    let absolute: Double

    /// integer and bool results must match exactly
    static let exact = Tolerance(relative: 0, absolute: 0)

    /// float32 computed on GPU vs python: relative, with a floor for values near zero
    static let float32 = Tolerance(relative: 1e-4, absolute: 1e-6)

    /// float16 / bfloat16
    static let float16 = Tolerance(relative: 5e-3, absolute: 1e-3)

    /// for cases where reduction order legitimately differs
    static let loose = Tolerance(relative: 1e-2, absolute: 1e-4)
}

/// Run a generated case.
///
/// - the default device is scoped to the block rather than set globally, so the
///   generated tests can run in parallel with tests that want another device
/// - MLX errors are converted into Swift errors instead of ending the process,
///   so a bad case fails its own test with the mlx message
/// - the random state is task-local and equivalent to python's
///   `mx.random.seed(seed)`
func withIntegrationState<R>(seed: UInt64, _ body: () throws -> R) throws -> R {
    try Device.withDefaultDevice(.gpu) {
        try withError {
            try withRandomState(MLXRandom.RandomState(seed: seed), body: body)
        }
    }
}

/// Assert that `array` matches the summary python produced.
func expectSummary(
    _ array: MLXArray, _ expected: ArraySummary, tolerance: Tolerance = .float32,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(array.shape == expected.shape, "shape", sourceLocation: sourceLocation)
    #expect(array.dtype == expected.dtype, "dtype", sourceLocation: sourceLocation)

    guard array.shape == expected.shape else {
        // the remaining comparisons would be meaningless (and may trap)
        return
    }

    // `reshaped([-1])` rather than `flattened()`: matches python's `reshape(-1)`
    // and also works for a 0-d (scalar) result
    let values = array.asType(.float32).reshaped([-1])
    let count = values.size
    let absolute = MLX.abs(values)
    let weights = MLX.arange(1, count + 1, dtype: .float32)

    expectClose(
        Double(values.mean().item(Float.self)), expected.mean, tolerance, "mean",
        sourceLocation: sourceLocation)
    expectClose(
        Double(values.min().item(Float.self)), expected.minimum, tolerance, "minimum",
        sourceLocation: sourceLocation)
    expectClose(
        Double(values.max().item(Float.self)), expected.maximum, tolerance, "maximum",
        sourceLocation: sourceLocation)
    expectClose(
        Double(absolute.sum().item(Float.self)), expected.absoluteSum, tolerance, "absoluteSum",
        sourceLocation: sourceLocation)
    expectClose(
        Double((absolute * weights).sum().item(Float.self)) / Double(count),
        expected.positionChecksum, tolerance, "positionChecksum",
        sourceLocation: sourceLocation)

    for (index, expectedValue) in zip(expected.sampleIndices, expected.samples) {
        expectClose(
            Double(values[index].item(Float.self)), expectedValue, tolerance,
            "element[\(index)]", sourceLocation: sourceLocation)
    }
}

private func expectClose(
    _ actual: Double, _ expected: Double, _ tolerance: Tolerance, _ label: String,
    sourceLocation: SourceLocation
) {
    if expected.isNaN {
        #expect(
            actual.isNaN, "\(label): expected nan, got \(actual)", sourceLocation: sourceLocation)
        return
    }
    if expected.isInfinite {
        #expect(
            actual == expected, "\(label): expected \(expected), got \(actual)",
            sourceLocation: sourceLocation)
        return
    }

    let limit = tolerance.absolute + tolerance.relative * Swift.abs(expected)
    let delta = Swift.abs(actual - expected)
    #expect(
        delta <= limit,
        "\(label): expected \(expected), got \(actual) (delta \(delta) > tolerance \(limit))",
        sourceLocation: sourceLocation)
}

// MARK: - Modules

/// The value a generated module test writes into every parameter.
///
/// Module tests do **not** rely on python and Swift drawing the same random
/// initialization (they do not: the two implementations draw different numbers of
/// keys in different orders).  Instead every parameter is replaced with a
/// deterministic function of its own shape, which both sides can produce exactly.
///
/// This must match `PARAMETER_PY` in `tools/integration_tests/core.py`:
///
/// ```
/// (mx.arange(v.size, dtype=mx.float32).reshape(v.shape) / v.size - 0.5).astype(v.dtype)
/// ```
func deterministicParameter(_ parameter: MLXArray) -> MLXArray {
    let size = parameter.size
    let values = MLX.arange(size, dtype: .float32).reshaped(parameter.shape) / Float(size) - 0.5
    return values.asType(parameter.dtype)
}

/// Assert that a module's parameters are named and shaped exactly as python's.
///
/// The keys matter beyond this test: mlx-swift uses them to load python
/// checkpoints, so a rename here is a compatibility break (this is why
/// `BatchNorm` spells its keys `running_mean` / `running_var`).  The shapes catch
/// a transposed weight, which would otherwise show up as a confusing value
/// mismatch (or a broadcast error) further down.
func expectParameters(
    _ module: Module, _ expected: [(String, [Int])],
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let actual = module.parameters().flattened()
        .map { ($0.0, $0.1.shape) }
        .sorted { $0.0 < $1.0 }

    #expect(
        actual.map { $0.0 } == expected.map { $0.0 }, "parameter keys",
        sourceLocation: sourceLocation)

    for (actualParameter, expectedParameter) in zip(actual, expected)
    where actualParameter.0 == expectedParameter.0 {
        #expect(
            actualParameter.1 == expectedParameter.1,
            "parameter \(actualParameter.0) shape",
            sourceLocation: sourceLocation)
    }
}
