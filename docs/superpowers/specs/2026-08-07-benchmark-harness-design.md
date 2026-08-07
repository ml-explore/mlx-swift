# Benchmark Harness — Design

Date: 2026-08-07

## Context

This is the first of two planned steps toward "general performance
optimization" for mlx-swift: build real measurement capability, then (as a
separate, later project) optimize based on what the data actually shows.

An audit of the repo found **no existing benchmark infrastructure**: no
`Benchmarks/` directory, no benchmarking package dependency, no XCTest
`measure()` usage anywhere. The only prior art is a disabled/half-alive
hand-rolled timing test (`Tests/MLXTests/TransformTests.swift:242`,
`testCompilePerformance()`) that `print()`s raw numbers with no statistics
and isn't wired into CI. No performance issue is documented anywhere in
`CONTRIBUTING.md`, `MAINTENANCE.md`, or recent commit history — this project
exists to create the ability to measure, not to fix a known problem.

A structural read of the code surfaced several candidate overhead points
(global `evalLock` contention, per-op `MLXArray` allocation/ARC cost since
it's a `final class`, vector/map marshalling in `Cmlx+Util.swift` used on
every `eval()` and closure-based transform call, `@TaskLocal` lookups on the
default-stream path) — but none of these are *known* to matter at any
measurable scale. This project's job is to produce real numbers for a small,
well-chosen set of cases, not to chase every theoretical hotspot at once.

## Non-goals

- **Fixing any performance issue.** This project only measures. Whatever the
  numbers show becomes the input to a separate, later optimization project.
- **CI regression gating.** No baseline-comparison logic, no JSON output, no
  "fail the build if N% slower than last run." That is real, separate
  infrastructure work worth its own scoping once the harness has proven
  useful locally.
- **Concurrent-contention benchmarking** (multiple tasks calling `eval()`
  simultaneously, to measure `evalLock` scaling under real concurrency).
  Valuable, but adds real complexity around making contention measurements
  non-flaky — deferred to a follow-up once the single-threaded harness
  exists and proves out.
- **A statistically rigorous benchmarking library** (e.g. `swift-benchmark`).
  Explicitly decided against: this repo has only two dependencies today
  (`swift-numerics`, `swift-argument-parser`), and a hand-rolled harness
  with warm-up + min/median/mean is enough to answer "does this candidate
  hotspot actually cost anything measurable," which is this project's whole
  job. Cross-platform verification (Linux+CUDA CI) is also simpler without
  a new external dependency to validate there too.

## Design

### File structure

A new executable target, following the exact pattern already established
by `Example1`/`Tutorial` (`Source/Examples/*.swift`, wired into
`Package.swift` as `.executableTarget(...)`):

- `Source/Benchmarks/Benchmarks.swift` — `@main struct Benchmarks`, with the
  same `--device cpu|gpu` command-line flag pattern as `Example1.swift`
  (`Source/Examples/Example1.swift:9-41`), so results can be compared across
  backends.
- `Package.swift` gains one new target entry:
  ```swift
  .executableTarget(
      name: "Benchmarks",
      dependencies: ["MLX", "MLXNN"],
      path: "Source/Benchmarks",
      sources: ["Benchmarks.swift"]
  ),
  ```
  No changes needed to `xcode/MLX.xcodeproj` — the existing example targets
  aren't represented there either (that project only tracks
  `MLX`/`MLXNN`/`MLXOptimizers`/`MLXTests`), so this follows the same
  "SwiftPM-only, run via `swift run Benchmarks`" convention and doesn't
  reintroduce the cross-build-system divergence risk found during the
  concurrency-modernization merge.

### Timing utility

```swift
struct BenchmarkResult {
    let name: String
    let iterations: Int
    let minMs: Double
    let medianMs: Double
    let meanMs: Double
}

func measure(
    name: String, warmup: Int = 5, iterations: Int = 100, _ body: () -> Void
) -> BenchmarkResult {
    for _ in 0 ..< warmup { body() }

    var samplesMs: [Double] = []
    samplesMs.reserveCapacity(iterations)
    for _ in 0 ..< iterations {
        let start = DispatchTime.now()
        body()
        let end = DispatchTime.now()
        samplesMs.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
    }

    samplesMs.sort()
    return BenchmarkResult(
        name: name, iterations: iterations,
        minMs: samplesMs.first!,
        medianMs: samplesMs[samplesMs.count / 2],
        meanMs: samplesMs.reduce(0, +) / Double(iterations))
}
```

`DispatchTime.now().uptimeNanoseconds` is the codebase's existing precedent
for monotonic timing (`Source/MLX/State.swift:39`, used for `RandomState`
seeding) — backed by `libdispatch`, confirmed cross-platform for this
repo's macOS/Linux/CUDA matrix. No `Date`/`ProcessInfo` alternative is used
anywhere else in `Source/`.

Results print as a plain aligned table (manual string padding, not
`String(format:)` — `%@`-style format specifiers aren't reliably portable
across Linux Foundation, and this repo explicitly supports Linux).

### The three benchmark cases

1. **Graph construction only.** Build a chain of ~200 sequential elementwise
   ops (e.g. repeated `x = x + 1`) on a small starting array, *without*
   calling `eval()`, once per measured iteration. Isolates per-op
   `MLXArray` allocation/ARC/dtype-cast overhead from any real compute,
   since building an unevaluated graph node does no C++-side work.

2. **`eval()` in isolation.** Create and `eval()` one array *once*, outside
   the timed loop, to fully realize it. Then repeatedly call `eval()` again
   on that same already-realized array for each measured iteration.
   Re-evaluating a realized array does effectively no compute at the C++
   level, so this isolates the `evalLock` acquire/release and
   `Cmlx+Util.swift` vector-marshalling cost (`new_mlx_vector_array`,
   `mlx_vector_array_values`) that runs on every `eval()` call regardless of
   whether there's real work to do.

3. **Composite: MLP forward pass, compiled vs. uncompiled.** A small
   3-layer stack (`MLXNN.Linear(128, 256)` → `relu` → `Linear(256, 256)` →
   `relu` → `Linear(256, 10)`) run over a `[32, 128]` batch, `eval()`'d each
   iteration. Measured twice: once as a raw closure, once wrapped in
   `compile()` (`Source/MLX/Transforms+Compile.swift:175-190`). This is the
   one case producing genuinely actionable output — it quantifies what
   `compile()` actually saves on a realistic-shaped workload, rather than
   reporting an abstract Swift-overhead number with no compute alongside it
   to contextualize whether that overhead is significant.

### Output

Running `swift run Benchmarks` (optionally `--device gpu`/`--device cpu`)
prints one aligned table row per case: name, iteration count, min/median/
mean milliseconds. No comparison logic, no historical tracking — a
developer reads the numbers and decides by eye whether something looks
worth investigating further.

## Testing

This is a benchmark tool, not library code with correctness requirements in
the usual sense — but each case's *setup* should be sanity-checked so a
bug in the benchmark itself doesn't get mistaken for a real finding:
- Confirm the graph-construction case's final array has the expected shape
  (proves the op chain actually ran end-to-end, not silently short-circuited).
- Confirm the eval-isolation case's array is actually evaluated before
  timing starts (check it doesn't re-trigger compute on every timed call).
- Confirm the compiled and uncompiled MLP paths produce numerically close
  outputs (`allClose`) — if `compile()` changes the result, the comparison
  is meaningless.

These are one-time assertions run at startup (printed as pass/fail, or a
`fatalError` if violated), not a persistent test suite — this target isn't
part of `Tests/MLXTests` and won't run under `xcodebuild test`.
