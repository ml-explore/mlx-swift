# Span Adoption for MLXArray Byte Copying — Design

Date: 2026-08-07

## Context

mlx-swift's raw-pointer usage was audited as a candidate for adopting
Swift's `Span`/`RawSpan`/`MutableRawSpan` types (bounds-checked memory
views, available since Swift 6.2, back-deployed to macOS 10.14.4+ /
iOS 12.2+ — well below this repo's current macOS 15+ floor, so there is no
deployment-target concern here unlike the concurrency-modernization work).

A full-repo sweep (`Source/MLX/**`, `Source/MLXNN`, `Source/MLXRandom`,
`Source/MLXFast`, `Source/MLXFFT`, `Source/MLXLinalg`, `Source/MLXOptimizers`,
`Source/Encuda`) found that the large majority of `Unsafe*Pointer` usage in
this codebase is unavoidable C interop: `@convention(c)` callback types,
out-parameters populated by `mlx_*` C calls, and `withUnsafeBufferPointer`/
`withUnsafeBytes` closures whose sole purpose is producing a raw pointer for
the very next line's C-ABI call. `Span`/`RawSpan` have no C-ABI
representation, so none of that surface is a real candidate.

The one genuine candidate is `Source/MLX/MLXArray+Bytes.swift`'s
`copy(from:toContiguous:)` — a hand-rolled, pure-Swift strided-copy loop
with manual pointer arithmetic and zero bounds checking, used by the public
`asArray(_:)` and `asData(access:)` APIs. This is internal implementation
detail (not part of any public signature), so this project is a pure safety
improvement with zero public API impact.

**Non-goal:** other public-API pointer-typed initializers
(`MLXArray+Init.swift`'s `init(rawPointer:)`, `init(_ ptr: UnsafeBufferPointer<T>, ...)`,
`init(_ ptr: UnsafeRawBufferPointer, ..., type:)`) and the deprecated
`ErrorHandler` C-callback API are explicitly out of scope for this project —
changing them is a breaking public API change and a separate decision.

## A note on what "Span adoption" actually buys here

`RawSpan`/`MutableRawSpan` do not eliminate `unsafe` from this code path
entirely. Bootstrapping a `RawSpan` from an `UnsafeRawBufferPointer` is
itself an `unsafe` call (the boundary crossing from C-owned memory has to
be asserted once), and the actual byte-copy still needs a scoped
`withUnsafeBytes`/`withUnsafeMutableBytes` closure to call into a
`memcpy`-equivalent. The honest value proposition is: **confine `unsafe` to
two narrow, clearly-marked points (the bootstrap, and the innermost copy
call) and make every offset computation in between bounds-checked**,
replacing manual `baseAddress! + sourceIndex * itemSize` pointer arithmetic
(currently checked by nothing) with `RawSpan.extracting(range:)` (checked by
the stdlib, traps on out-of-bounds).

## A real correctness finding that shapes the design

The non-contiguous branch of `copy(from:toContiguous:)` has an existing
comment acknowledging that with negative strides (e.g.
`asStrided(a, [3,3], strides: [-3,-1], offset: 8)`), the computed
`sourceIndex` can be negative, meaning the accessed address is *before*
`from.baseAddress!`. Tracing through that exact example: accessing index
`(2,2)` computes `sourceIndex = 2*(-3) + 2*(-1) = -8`, landing 8 elements
*backward* from the passed-in base pointer. The current code works only
because raw pointer arithmetic (`UnsafeRawBufferPointer.baseAddress! +`)
has zero bounds enforcement — it relies entirely on an invisible guarantee
from MLX's C++ layer that the address stays inside the real allocation.

This case has **zero test coverage** today (confirmed by a full sweep of
`Tests/MLXTests/` for negative-stride/negative-offset arrays piped through
`asArray`/`asData`). A naive "swap `UnsafeRawBufferPointer` for `RawSpan`,
same base pointer, same count" port would **newly trap** on exactly this
case, since `RawSpan.extracting(range:)` enforces `0 ≤ lowerBound`. That
would be a silent, untested regression.

**Decision:** do the full, correct fix rather than a shallow port or a
partial (contiguous-path-only) adoption that leaves the more interesting
half of the function unsafe. Add real test coverage for the negative-stride
case as part of this work, since the current absence of that coverage is
itself the reason this risk went unnoticed.

## Design

### The min/max reachable offset algorithm

For a given dimension's `(shape, stride)` pair, the per-dimension index
runs `0...(shape-1)`, so its contribution to `sourceIndex` ranges from
`min(0, (shape-1)*stride)` to `max(0, (shape-1)*stride)` — this correctly
handles both positive and negative strides. Summing these per-dimension
bounds across every iterated dimension gives the *true* minimum and maximum
reachable `sourceIndex` for the whole loop:

```swift
var minSourceIndex = 0
var maxSourceIndex = 0
for dimension in 0..<ndim {
    let contribution = (shape[dimension] - 1) * strides[dimension]
    minSourceIndex += Swift.min(0, contribution)
    maxSourceIndex += Swift.max(0, contribution)
}
```

**Verified against the existing positive-stride test coverage**: for
`testAsArrayNonContiguous1`'s slice (`a[0..<2, 1..<3]` on a `[3,3]` array,
iterated dimension has `stride = 3 ≥ 0`), the formula computes
`minSourceIndex = 0` — matching the current code's implicit assumption
that iteration starts at `from.baseAddress!` with no backward extension.
This is a strict generalization: every already-tested, all-positive-stride
case is unaffected; only the previously-unsafe negative case changes
behavior (from "silently trusts C++" to "explicitly bounds-checked").

### The rewrite

`copy(from:toContiguous:)` changes to take a raw base pointer (not a
pre-sized `UnsafeRawBufferPointer`) for the source, since the function must
compute its own correct bounds rather than trust a caller-supplied size
that (for the non-contiguous branch) was never actually anchored correctly
in the first place:

- **Contiguous fast path** (`contiguousDimension == 0`): bootstrap a
  `RawSpan`/`MutableRawSpan` over `[0, byteCount)` — no negative-offset
  concern here at all, this path is a straight whole-buffer copy.
- **Non-contiguous branch**: compute `minSourceIndex`/`maxSourceIndex` per
  the algorithm above, then do **one** `unsafe` bootstrap of a `RawSpan`
  covering exactly `[minSourceIndex * itemSize, maxSourceIndex * itemSize +
  destItemSize)` relative to the base pointer — this correctly extends
  *backward* from the base pointer when strides are negative. Every loop
  iteration then computes its offset relative to `minSourceIndex` (so it's
  always ≥ 0) and navigates via `.extracting(range:)` (bounds-checked)
  instead of raw pointer addition. The actual byte copy happens inside
  matched `withUnsafeBytes`/`withUnsafeMutableBytes` closures, which are
  `@safe` APIs that hand back a raw pointer scoped to exactly the extracted
  sub-range.

The destination side (`output`/`toContiguous`) is always written
sequentially from offset 0 with no negative-offset concern, so it only
needs the straightforward `MutableRawSpan` bootstrap over its full byte
range.

### Call site changes

The three call sites in `MLXArray+Bytes.swift` (`asArray<T>()` line ~131,
`asDataCopy()` line ~181) currently construct
`UnsafeRawBufferPointer(start: mlx_array_data_uint8(self.ctx), count:
physicalSize * itemSize)` before calling `copy`. These change to pass the
raw `mlx_array_data_uint8(self.ctx)` pointer directly (as
`UnsafeRawPointer`), since `copy` now computes its own bounds internally.
`physicalSize` remains used elsewhere (it's a public-ish internal property
with its own doc comment and test coverage) but is no longer used to
pre-size the `from` buffer for this function.

## Testing

Add a new test to `Tests/MLXTests/MLXArrayTests.swift`, alongside the
existing `testAsArrayNonContiguous*` tests, using the exact example already
documented in `asStrided`'s own doc comment (`Source/MLX/Ops.swift:351`):

```swift
func testAsArrayNegativeStride() {
    // negative strides + a nonzero offset: this the case that was
    // previously untested and relied on raw pointer arithmetic outside
    // any range the Swift-side code verified
    let a = MLXArray(0 ..< 16, [4, 4])
    let s = asStrided(a, [4, 4], strides: [-4, -1], offset: 15)

    let expected = Array((0 ..< 16).reversed())
    assertEqual(s, MLXArray(expected.map { Int32($0) }, [4, 4]))

    let s_arr = s.asArray(Int32.self)
    XCTAssertEqual(s_arr, expected.map { Int32($0) })
}
```

This exercises the negative-stride path through `asArray`, which is
exactly the code path being rewritten.

**Full existing coverage to re-run unchanged**: `testArrayRead`,
`testAsArrayContiguous`, `testAsArrayNonContiguous1/2/4`,
`testAsDataContiguous`, `testAsDataContiguousNoCopy`, `testAsDataRoundTrip`,
`testAsDataNonContiguous`, `testAsDataNonContiguousNoCopy` — all must
continue passing unchanged, since the rewrite is designed to be behavior-
preserving for every currently-tested case.

## Risks

- **The min/max formula must be re-verified by direct compilation and test
  run, not just hand-tracing.** The hand-verification above covers one
  positive-stride case and one negative-stride case; the implementation
  step must run the full existing suite plus the new test before this is
  trusted.
- **`RawSpan`/`MutableRawSpan` bootstrap calls require the `unsafe` keyword
  at the call site** (confirmed by direct compilation against this
  project's exact toolchain/settings) — this is expected and is the
  intended "confine unsafe to two narrow points" pattern, not a design
  flaw to work around.
