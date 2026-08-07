# Concurrency Modernization — Design

Date: 2026-08-06
Branch: `modernize/concurrency`

## Context

mlx-swift is a Swift API over the MLX C++ array library (accessed through the
`mlx-c` handle layer, imported as `Cmlx`). This is the first of several
planned modernization efforts (the others — `MLXArray` value semantics, Span
adoption for raw memory, and general perf optimization — are separate,
larger, riskier sub-projects and are explicitly deferred; see "Non-goals"
below).

An audit of the current Swift sources found the synchronization story is a
mix of ad hoc primitives with inconsistent documentation:

- `NSLock` / `NSRecursiveLock` used in ~8 places to guard mutable state.
- `DispatchQueue(label:).sync` used in two places as a de facto lock.
- ~10 types marked `@unchecked Sendable`, only one of which (`CompiledFunction`)
  has a comment explaining why the compiler can't verify it.
- Per-target `.enableExperimentalFeature("StrictConcurrency")` in
  `Package.swift`, rather than full Swift 6 language mode.
- One actor (`WiredMemoryManager`) already exists and is a good example of
  modern structured concurrency — it is left alone except for one internal
  lock swap, for consistency.

None of this is broken today. The goal is to replace the ad hoc primitives
with Swift's standard, cross-platform `Mutex` (from the `Synchronization`
module, part of the toolchain since Swift 6.0 — works identically on macOS,
Linux glibc/musl, and Linux+CUDA, which matches this repo's full CI matrix),
tighten the `Sendable` story so `@unchecked` is only used where it's actually
necessary and always explained, and turn on full Swift 6 language mode to
catch what the experimental flag doesn't.

## Goals

1. Replace `NSLock`/`DispatchQueue.sync`-based mutual exclusion with
   `Synchronization.Mutex` everywhere reentrancy is not required.
2. Leave `evalLock` (`Transforms+Eval.swift`) as `NSRecursiveLock` — it is
   genuinely reentrant (a compiled function's tracer closure can call back
   into `eval()` while the outer lock is held) and `Mutex` does not support
   recursion. Add a comment documenting this so a future contributor doesn't
   "modernize" it into a deadlock.
3. Audit every `@unchecked Sendable`:
   - Where the only mutable state is now `Mutex`-protected and all other
     stored properties are themselves `Sendable`, drop to compiler-checked
     `Sendable` (no `@unchecked`).
   - Where the type holds a non-`Sendable` value (e.g. a plain closure, or an
     `MLXArray`, which is a reference type with no `Sendable` conformance),
     keep `@unchecked Sendable` but add a one-line rationale comment.
   - Where the type only wraps an opaque C handle (`mlx_stream`,
     `mlx_device`, `mlx_fast_metal_kernel`) that the Swift importer can't
     prove `Sendable`, keep `@unchecked Sendable` with a comment explaining
     the handle is immutable after init and correctly freed exactly once.
4. Move `Package.swift` from `.enableExperimentalFeature("StrictConcurrency")`
   to `.swiftLanguageMode(.v6)` per Swift target, and fix whatever new
   diagnostics that surfaces.
5. Fix an incidental gap found during the audit: `MLXFastKernel.callAsFunction`
   calls `mlx_fast_metal_kernel_apply` without holding `evalLock`, unlike
   every other call that touches the backend graph. Wrap it to match.

## Non-goals

- **`MLXArray` value semantics.** `MLXArray` is a reference type wrapping a
  handle to a lazy computation-graph node; this is very likely intentional
  given the graph-sharing design of the underlying C++ library, not an
  oversight. Redesigning it as a COW value type is a large, breaking,
  separate effort and is not part of this design.
- **Span adoption** for the raw-pointer call sites in `MLXArray+Init`,
  `MLXArray+Bytes`, `MLXArray+Metal`, `IO`, and `Transforms+Vmap`.
- **General performance optimization** — needs its own profiling-driven scope.
- **Public API changes.** `compile()`, `CustomFunction`, `RandomState`, etc.
  keep their synchronous, closure-based signatures. In particular, public
  closure parameters are **not** changed to require `@Sendable` — doing so
  would be source-breaking for downstream consumers (mlx-swift-lm,
  mlx-swift-examples) that pass ordinary closures. Internal serialization via
  `Mutex` plus a documented `@unchecked Sendable` on the wrapping state
  object is the accepted trade-off; see "Risks" below.
- **Converting `WiredMemoryManager`'s design or its actor-based API.** It's
  already a good example of the target style; only its internal
  `EndOnceGuard` lock is swapped for consistency.

## Design

### Replacing locks with `Mutex`

`Synchronization.Mutex<State>` is a value type: it owns its protected state
directly (`Mutex<Device?>` rather than "a lock plus a var next to it"), which
is both safer (impossible to touch the state without acquiring the lock —
there's no separate `var` to forget to guard) and idiomatic modern Swift.

Concrete replacements:

- `Device._lock` + `nonisolated(unsafe) static var _defaultDevice: Device?`
  → `static let _defaultDevice = Mutex<Device?>(nil)`. Also removes the
  `#if swift(>=5.10)` branch, since the project's minimum toolchain
  (swift-tools-version 6.3) is well past that.
- `Memory.queue` + two `nonisolated(unsafe) static var` (`_cacheLimit`,
  `_memoryLimit`) → a single `Mutex<(cacheLimit: Int?, memoryLimit: Int?)>`,
  consolidating what were two independently-guarded-by-convention globals
  into one value the lock actually owns.
- `ErrorBox.lock`, `ErrorHandler.lock` → `Mutex`.
- `CompiledFunction.lock`, `_CustomFunctionState.lock`, `RandomState.lock`
  → `Mutex`.
- `MLXNN.Cache.queue` (`DispatchQueue(label: "Cache")` + `.sync`) → `Mutex`.
  This is called on every forward pass that uses `RoPE`, so this also drops
  GCD dispatch overhead from a hot path, not just a style change.
- `WiredMemoryTicket.EndOnceGuard` → `Mutex<Bool>`.

`evalLock` is the one exception, kept as `NSRecursiveLock` (see Goal 2).

### `Sendable` audit outcomes

| Type | Before | After |
|---|---|---|
| `ErrorBox` | `@unchecked Sendable` | checked `Sendable` (only state is `Mutex<Error?>`) |
| `ErrorHandler` (private) | `@unchecked Sendable` | checked `Sendable` |
| `EndOnceGuard` | `@unchecked Sendable` | checked `Sendable` |
| `CompiledFunction` | `@unchecked Sendable`, has comment | stays `@unchecked` — holds a plain (non-`@Sendable`) closure and `[any Updatable]`; comment kept/refreshed |
| `_CustomFunctionState` | `@unchecked Sendable` | stays `@unchecked` — same reason |
| `RandomState` | `@unchecked Sendable` | stays `@unchecked` — wraps a non-`Sendable` `MLXArray`; existing doc comment about cross-thread evaluation already covers this |
| `Cache` | `@unchecked Sendable` | stays `@unchecked` — `Element` is an unconstrained generic, may be non-`Sendable` (e.g. `MLXArray`) |
| `Stream`, `Device`, `MLXFastKernel` | `@unchecked Sendable`, no comment | stays `@unchecked`, comment added explaining the wrapped C handle is immutable after init and freed exactly once in `deinit` |

### Swift 6 language mode

Replace, per first-party target (`MLX`, `MLXRandom`, `MLXFast`, `MLXNN`,
`MLXOptimizers`, `MLXFFT`, `MLXLinalg`):

```swift
swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
```

with:

```swift
swiftSettings: [.swiftLanguageMode(.v6)]
```

This is expected to surface additional diagnostics beyond what the
experimental flag caught. Per the non-goals above, these are fixed without
changing public signatures — e.g. via `nonisolated(unsafe)` only where
provably safe and already the established pattern, or additional
`@unchecked Sendable` with rationale, never by forcing `@Sendable` onto
public closure parameters.

### `MLXFastKernel` evalLock gap

`callAsFunction` builds `mlx_fast_metal_kernel_config`, populates it, and
calls `mlx_fast_metal_kernel_apply` — all without `evalLock`. Every other
call that mutates or reads the backend graph (stream creation, compile,
custom function apply, eval) goes through `evalLock`. Wrap the
config-build-and-apply sequence in `evalLock.withLock` to match.

## Addendum (discovered during implementation, 2026-08-06)

`Synchronization.Mutex` carries `@available(macOS 15.0, iOS 18.0, watchOS
11.0, tvOS 18.0, visionOS 2.0, *)` in Apple's SDK (confirmed directly against
the SDK's `Synchronization.swiftinterface`) — an OS-runtime-ABI gate,
independent of the Swift language mode / toolchain version this design
originally checked (which was correct as far as it went). `Package.swift`
declared `.macOS("14.0"), .iOS(.v17), .tvOS(.v17), .visionOS(.v1)`, which is
below that floor.

Decision: raise the package's minimum deployment targets to `.macOS("15.0"),
.iOS(.v18), .tvOS(.v18), .visionOS(.v2)` to unblock `Mutex` adoption. This is
a real breaking change for consumers still targeting the older OS versions;
approved explicitly by the human partner over the alternatives (dual-path
`@available` gating per type, or dropping the `Mutex`-adoption goal
entirely). Linux/CUDA targets are unaffected — Linux Swift has no
OS-version-gated availability model for this API.

This changes Goal 1 (implicitly) and the Testing section's toolchain-only
availability claim in the original design above; both should be read
together with this addendum. The plan document's Task 0 carries the actual
`Package.swift` change.

## Addendum 2 (discovered during implementation, 2026-08-06): `MLXNN.Cache`

The original design's `Sendable` audit table said `Cache` should drop
`@unchecked Sendable` to checked `Sendable` once its state moves inside a
`Mutex`, on the reasoning that `Element` is "an unconstrained generic, may
be non-`Sendable`" -- but then contradicted itself by listing this as a
reason to *stay* `@unchecked`. That contradiction was not caught before
implementation. It turns out the "stay `@unchecked`" outcome is the correct
one, for a sharper reason than the audit gave: `Mutex.withLock`'s closure
takes `inout sending Value` and returns a `sending Result`. Swift's region
isolation checker rejects returning or extracting a value of an
*unconstrained generic* type (`Cache`'s `Entry`, which embeds `Element`)
through that boundary -- confirmed by direct compilation
(`error: 'inout sending' parameter cannot be task-isolated at end of
function` / `returning a task-isolated value as a 'sending' result risks
causing data races`). This is unrelated to whether `Element` is actually
`Sendable` at any given instantiation (e.g. `ALiBi`'s `Cache<Key, MLXArray>`
-- `MLXArray` is not `Sendable`, so a `Sendable`-constrained `Cache` isn't
usable for its one real caller anyway). `RandomState` (Task 6) didn't hit
this because its `Mutex<MLXArray>` has a *concrete* type argument, which
region isolation can reason about directly.

Decision: `Cache` keeps `@unchecked Sendable` (with a rationale comment),
same posture as `CompiledFunction`/`_CustomFunctionState`/`RandomState`. It
still becomes `final` (no subclassers, `internal`, no reason not to) and
still gets the `Mutex` swap -- only the `Sendable` conformance line in the
original design's audit table was wrong.

**Correction (same day, verified by direct compilation):** marking `Cache`
itself `@unchecked Sendable` was necessary but not sufficient. The region
checker's `sending` boundary in `Mutex.withLock` independently inspects
*every type nested inside* `Value` (here, `Cache.State`, which embeds
`Cache.Entry`, which embeds `Element`) for `Sendable`-conformance --
regardless of the enclosing class's own conformance annotation. Both
`Cache.State` and `Cache.Entry` also need `@unchecked Sendable` for the
build to actually pass; `Cache` alone left two `error: 'inout sending'
parameter ... cannot be task-isolated` / `error: returning a task-isolated
... value as a 'sending' result` diagnostics unresolved. Confirmed by
iterative compilation: `Cache: @unchecked Sendable` alone still failed on
the getter's `Entry`-derived return; adding `Entry: @unchecked Sendable`
resolved that but the setter's `inout State` parameter still failed until
`State: @unchecked Sendable` was added too. All three annotations together
build clean and the new `PositionalEncodingTests.testALiBiCaching` passes.

## Addendum 3 (discovered during implementation, 2026-08-06): Task 10 was a no-op

The "Swift 6 language mode fallout is not fully predictable in advance" risk
called out below turned out not to apply. Verified empirically with a
throwaway package: `swift-tools-version: 6.3` (which this repo already
declared, unrelated to this plan) causes SwiftPM to pass `-swift-version 6`
to every target *regardless* of whether `.enableExperimentalFeature
("StrictConcurrency")` or `.swiftLanguageMode(.v6)` is present in that
target's `swiftSettings` -- the tools-version alone already selects full
Swift 6 language mode. So every first-party target was already compiling
under full Swift 6 checking before Task 10 ran; swapping the feature flag
for the explicit `.swiftLanguageMode(.v6)` setting changed nothing
observable and surfaced zero new diagnostics (confirmed: clean build, full
538-test suite passes, identical to the pre-Task-10 baseline). The edit is
still worth keeping -- it pins the language mode explicitly so a future
tools-version downgrade can't silently relax checking -- but it should not
be read as validating Tasks 1-9's Sendable/concurrency work under a
*newly*-stricter mode; that validation already happened implicitly, task by
task, throughout Tasks 1-9's own build/test runs.

## Risks

- **Swift 6 language mode fallout is not fully predictable in advance.** If
  fixing the new diagnostics would require changing a public signature
  (e.g. forcing `@Sendable` on a closure parameter), that's a stop-and-report
  point, not something to push through silently — it would be a breaking
  change for downstream consumers and needs a separate decision.
- **`Mutex` availability**: confirm the CI's oldest declared Swift toolchain
  (Swift 6.2.3, used by the Linux+CUDA runner) has `Synchronization.Mutex`
  available on Linux. (It shipped as part of the Swift 6.0 standard library,
  cross-platform, so this should be a non-issue, but worth confirming during
  implementation since it's a hard blocker if wrong.)

## Testing

- No new external dependencies — `Synchronization` ships in the toolchain.
- Run the existing `Tests/MLXTests` suite on macOS (SwiftPM + Xcode project)
  after each file's lock swap.
- Add a small concurrency stress test exercising `Device.defaultDevice()`
  and `Memory.cacheLimit` from multiple concurrent `Task`s, to validate the
  new `Mutex` paths under contention (this is new coverage — today there is
  no test exercising these locks concurrently).
- Verify the Linux CMake builds (glibc x86_64/aarch64, CUDA) still succeed
  once Swift 6 language mode is enabled.
