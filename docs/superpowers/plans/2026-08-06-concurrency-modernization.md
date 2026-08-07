# Concurrency Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ad hoc `NSLock`/`DispatchQueue.sync`-based synchronization in mlx-swift's Swift layer with `Synchronization.Mutex`, tighten the `@unchecked Sendable` story so it's only used where actually necessary (and always explained), and move from experimental `StrictConcurrency` to full Swift 6 language mode — with zero public API changes.

**Architecture:** File-by-file swap of each lock primitive for `Synchronization.Mutex<State>`, moving the protected state to live *inside* the mutex so the compiler can verify safety structurally wherever possible (upgrading `@unchecked Sendable` to checked `Sendable`). One exception (`evalLock`) stays as `NSRecursiveLock` because it is genuinely reentrant and `Mutex` doesn't support recursion.

**Tech Stack:** Swift 6.3 (local), Swift 6.2.3+ (CI floor), `Synchronization` module (stdlib since Swift 6.0, cross-platform), XCTest.

**Spec:** `docs/superpowers/specs/2026-08-06-concurrency-modernization-design.md`

## Global Constraints

- No public API signature changes. The one exception explicitly permitted by the
  spec: `Source/MLXNN/Cache.swift`'s `Cache<Key, Element>` is `internal` (not
  public, no subclassers anywhere in the repo — confirmed), so it may become
  `final` and its conformance may change from `@unchecked Sendable` to `Sendable`.
- Every replaced lock uses `Synchronization.Mutex`. Add `import Synchronization`
  to any file that gains a `Mutex`. Never introduce a new `NSLock`,
  `NSRecursiveLock`, or `DispatchQueue` in a touched file.
- Exception: `evalLock` in `Source/MLX/Transforms+Eval.swift` stays
  `NSRecursiveLock` — it is reentrant (a compiled function's tracer closure can
  call back into `eval()` while the outer lock is held) and `Mutex` does not
  support recursion. Only its doc comment changes (Task 9).
- Do not remove existing `import Foundation` statements — every touched file
  uses Foundation for something else (e.g. `String(cString:encoding:)`,
  `DispatchTime`), confirmed per-file during planning.
- Toolchain: swift-tools-version is already `6.3`. `Synchronization.Mutex`
  needs Swift 6.0+ stdlib for the language/toolchain side (confirmed present
  locally and on CI's oldest pinned toolchain, Swift 6.2.3 on the Linux+CUDA
  runner) -- **but on Apple platforms it additionally carries
  `@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0,
  *)`** (confirmed against the SDK's `Synchronization.swiftinterface`; this
  is an OS-runtime-ABI gate, not a toolchain thing). `Package.swift`'s
  deployment targets were below that floor and had to be raised -- see
  Task 0, which must run and be committed before any other task. Linux/CUDA
  builds are unaffected; this gate is Apple-platform-only.
- All tests use `XCTest` (`class Foo: XCTestCase`), matching every existing
  file under `Tests/MLXTests/`. **Do not use `swift test`** -- it fails at
  runtime with "Failed to load the default metallib" because SwiftPM's CLI
  doesn't bundle the Metal resource the GPU backend needs (confirmed during
  setup; this is also why CI uses `xcodebuild`, never `swift test`, for
  macOS). Scoped runs: `xcodebuild test -scheme mlx-swift-Package
  -destination 'platform=macOS' -only-testing:MLXTests/<ClassName>`. Full
  suite: `xcodebuild test -scheme mlx-swift-Package -destination
  'platform=macOS'`. Plain `swift build` (not `swift test`) is fine for
  compile-only checks.
- Branch: `modernize/concurrency` (already checked out). Commit after every task.

---

### Task 0: Raise minimum deployment targets for `Synchronization.Mutex`

This task is a hard prerequisite for Tasks 1-8 (every task that adopts
`Mutex`). It must be complete and committed before any of them start.
`Synchronization.Mutex` is `@available(macOS 15.0, iOS 18.0, watchOS 11.0,
tvOS 18.0, visionOS 2.0, *)` on Apple platforms (SDK-verified) --
`Package.swift`'s prior targets (`macOS 14.0`, `iOS 17`, `tvOS 17`,
`visionOS 1`) are below that floor. Raising them was an explicit decision by
the human partner (a real breaking change for consumers on older OS
versions, accepted in exchange for the simpler, unconditional `Mutex`
adoption the rest of this plan assumes) -- do not revisit that decision in
this task; just carry it out.

**Files:**
- Modify: `Package.swift`

**Interfaces:**
- Produces: raises `mlx-swift`'s minimum supported OS versions. Every
  downstream consumer (mlx-swift-lm, mlx-swift-examples, and any app
  depending on this package) now requires macOS 15 / iOS 18 / tvOS 18 /
  visionOS 2 or newer. This is intentional and approved -- not a bug to
  work around in a later task.

- [ ] **Step 1: Raise the platform floors**

In `Package.swift`, change:

```swift
    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
```

to:

```swift
    platforms: [
        .macOS("15.0"),
        .iOS(.v18),
        .tvOS(.v18),
        .visionOS(.v2),
    ],
```

- [ ] **Step 2: Build**

Run: `swift build`

Expected: succeeds (this alone doesn't touch any Mutex-using code yet --
it just confirms the package manifest and existing code still build against
the raised targets).

- [ ] **Step 3: Run the full test suite**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'`

Expected: PASS, same 533 tests as the pre-existing baseline (this task
doesn't change any runtime behavior, only the declared minimum OS).

- [ ] **Step 4: Commit**

```bash
git add Package.swift
git commit -m "Raise minimum deployment targets to unblock Synchronization.Mutex adoption"
```

---

### Task 1: `Device` — Mutex-backed default device storage

**Files:**
- Modify: `Source/MLX/Device.swift`
- Test: `Tests/MLXTests/StreamTests.swift` (existing — covers `Device` despite the file name)

**Interfaces:**
- Produces: no change to `Device.defaultDevice()`, `Device.withDefaultDevice(_:_:)`, `Device.setDefault(device:)` signatures or behavior.

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLX/Device.swift`, change:

```swift
import Cmlx
import Foundation

///Type of device.
```

to:

```swift
import Cmlx
import Foundation
import Synchronization

///Type of device.
```

- [ ] **Step 2: Add a `Sendable` rationale comment to the `Device` class**

Change:

```swift
public final class Device: @unchecked Sendable, Equatable {
```

to:

```swift
// `@unchecked Sendable`: `ctx` (`mlx_device`) and `defaultStream` are `let` and never
// mutated after `init` -- the imported C struct isn't recognized as `Sendable` by the
// compiler, but sharing an immutable, already-constructed `Device` across threads is safe.
public final class Device: @unchecked Sendable, Equatable {
```

- [ ] **Step 3: Replace the lock + optional static with a `Mutex`**

Change:

```swift
    // support for global default device
    static let _lock = NSLock()
    #if swift(>=5.10)
        nonisolated(unsafe) static var _defaultDevice: Device?
    #else
        static var _defaultDevice: Device?
    #endif

    @TaskLocal static var _tlDefaultDevice = _resolveGlobalDefaultDevice()

    private static func _resolveGlobalDefaultDevice() -> Device {
        _lock.withLock {
            if let device = _defaultDevice {
                return device
            }
            // Ask the underlying MLX C++ core for its default device rather
            // than hard-coding `.gpu`. On Apple platforms with Metal this
            // still resolves to GPU; on a CPU-only host (Linux without
            // CUDA / no Metal) it correctly resolves to CPU. Hard-coding GPU
            // here meant `defaultDevice()` / `StreamOrDevice.default`
            // returned an unavailable device on those hosts.
            var ctx = mlx_device_new()
            mlx_get_default_device(&ctx)
            return Device(ctx)
        }
    }
```

to:

```swift
    // support for global default device
    static let _defaultDevice = Mutex<Device?>(nil)

    @TaskLocal static var _tlDefaultDevice = _resolveGlobalDefaultDevice()

    private static func _resolveGlobalDefaultDevice() -> Device {
        _defaultDevice.withLock { stored in
            if let stored {
                return stored
            }
            // Ask the underlying MLX C++ core for its default device rather
            // than hard-coding `.gpu`. On Apple platforms with Metal this
            // still resolves to GPU; on a CPU-only host (Linux without
            // CUDA / no Metal) it correctly resolves to CPU. Hard-coding GPU
            // here meant `defaultDevice()` / `StreamOrDevice.default`
            // returned an unavailable device on those hosts.
            var ctx = mlx_device_new()
            mlx_get_default_device(&ctx)
            let resolved = Device(ctx)
            stored = resolved
            return resolved
        }
    }
```

- [ ] **Step 4: Update `setDefault` to use the mutex**

Change:

```swift
    static public func setDefault(device: Device?) {
        _lock.withLock {
            if let device {
                // sets the mlx core default device -- only used
                // by the deprecated init().  this isn't thread
                // safe or really usable across tasks/threads
                // but is kept for backward compatibility
                mlx_set_default_device(device.ctx)
            }
            _defaultDevice = device
        }
    }
```

to:

```swift
    static public func setDefault(device: Device?) {
        _defaultDevice.withLock { stored in
            if let device {
                // sets the mlx core default device -- only used
                // by the deprecated init().  this isn't thread
                // safe or really usable across tasks/threads
                // but is kept for backward compatibility
                mlx_set_default_device(device.ctx)
            }
            stored = device
        }
    }
```

- [ ] **Step 5: Build**

Run: `swift build`

Expected: succeeds. **If this fails with an error that `Synchronization` or `Mutex` can't be found**, STOP — this means the toolchain doesn't actually support it on this platform, which contradicts what was confirmed during planning. Report back before substituting any other primitive; don't silently fall back to `NSLock`.

- [ ] **Step 6: Run the existing Device tests**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/StreamTests`

Expected: PASS (covers `testEquatableDevice`, `testDeviceType`, `testUsingDevice`, `testSetUnsetDefaultDevice`, `testWithDefaultDevice`).

- [ ] **Step 7: Commit**

```bash
git add Source/MLX/Device.swift
git commit -m "Use Synchronization.Mutex for Device's default-device storage"
```

---

### Task 2: `Memory` — Mutex-backed cache/memory limit storage

**Files:**
- Modify: `Source/MLX/Memory.swift`
- Test: `Tests/MLXTests/MemoryTests.swift` (existing, gets two new test functions)

**Interfaces:**
- Produces: no change to `Memory.cacheLimit`, `Memory.memoryLimit` signatures or behavior.

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLX/Memory.swift`, change:

```swift
import Cmlx
import Foundation
```

to:

```swift
import Cmlx
import Foundation
import Synchronization
```

- [ ] **Step 2: Replace the queue + two loose statics with one `Mutex`**

Change:

```swift
    static let queue = DispatchQueue(label: "GPUEnum")

    // note: these are guarded by the queue above
    #if swift(>=5.10)
        nonisolated(unsafe) static var _cacheLimit: Int?
        nonisolated(unsafe) static var _memoryLimit: Int?
    #else
        static var _cacheLimit: Int?
        static var _memoryLimit: Int?
    #endif
```

to:

```swift
    /// Cached values for ``cacheLimit`` and ``memoryLimit``, owned by a `Mutex`
    /// rather than a dispatch queue plus loosely-guarded statics.
    static let limits = Mutex<(cacheLimit: Int?, memoryLimit: Int?)>((nil, nil))
```

- [ ] **Step 3: Update the `cacheLimit` property**

Change:

```swift
    public static var cacheLimit: Int {
        get {
            queue.sync {
                if let cacheLimit = _cacheLimit {
                    return cacheLimit
                }

                // set it to a reasonable value in order to read it, then set it back
                // to current
                var current: size_t = 0
                var discard: size_t = 0
                mlx_set_cache_limit(&current, cacheMemory)
                mlx_set_cache_limit(&discard, current)

                _cacheLimit = current
                return current
            }
        }
        set {
            queue.sync {
                _cacheLimit = newValue
                var current: size_t = 0
                mlx_set_cache_limit(&current, newValue)
            }
        }
    }
```

to:

```swift
    public static var cacheLimit: Int {
        get {
            limits.withLock { state in
                if let cacheLimit = state.cacheLimit {
                    return cacheLimit
                }

                // set it to a reasonable value in order to read it, then set it back
                // to current
                var current: size_t = 0
                var discard: size_t = 0
                mlx_set_cache_limit(&current, cacheMemory)
                mlx_set_cache_limit(&discard, current)

                state.cacheLimit = current
                return current
            }
        }
        set {
            limits.withLock { state in
                state.cacheLimit = newValue
                var current: size_t = 0
                mlx_set_cache_limit(&current, newValue)
            }
        }
    }
```

- [ ] **Step 4: Update the `memoryLimit` property**

Change:

```swift
    public static var memoryLimit: Int {
        get {
            queue.sync {
                var current: size_t = 0
                mlx_get_memory_limit(&current)
                return Int(current)
            }
        }
        set {
            queue.sync {
                _memoryLimit = newValue
                var current: size_t = 0
                mlx_set_memory_limit(&current, newValue)
            }
        }
    }
```

to:

```swift
    public static var memoryLimit: Int {
        get {
            limits.withLock { _ in
                var current: size_t = 0
                mlx_get_memory_limit(&current)
                return Int(current)
            }
        }
        set {
            limits.withLock { state in
                state.memoryLimit = newValue
                var current: size_t = 0
                mlx_set_memory_limit(&current, newValue)
            }
        }
    }
```

- [ ] **Step 5: Write the new round-trip tests**

In `Tests/MLXTests/MemoryTests.swift`, add these two test functions inside `class MemoryTests: XCTestCase { ... }`, after `testWiredMemory`:

```swift
    func testCacheLimitRoundTrip() {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }

        Memory.cacheLimit = 4096
        XCTAssertEqual(Memory.cacheLimit, 4096)
    }

    func testMemoryLimitRoundTrip() {
        let original = Memory.memoryLimit
        defer { Memory.memoryLimit = original }

        Memory.memoryLimit = original + 1024
        XCTAssertEqual(Memory.memoryLimit, original + 1024)
    }
```

(These didn't exist before — the only prior coverage was `testWiredMemory`, which doesn't touch `cacheLimit`/`memoryLimit` directly.)

- [ ] **Step 6: Run the tests**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/MemoryTests`

Expected: PASS, including the two new tests.

- [ ] **Step 7: Commit**

```bash
git add Source/MLX/Memory.swift Tests/MLXTests/MemoryTests.swift
git commit -m "Use Synchronization.Mutex for Memory's cache/memory limit storage"
```

---

### Task 3: `ErrorBox` / `ErrorHandler` — Mutex swap, drop to checked `Sendable`

**Files:**
- Modify: `Source/MLX/ErrorHandler.swift`
- Test: `Tests/MLXTests/ErrorTests.swift` (existing)

**Interfaces:**
- Produces: `ErrorBox` is now `Sendable` (was `@unchecked Sendable`) — this is a
  *strictly weaker* constraint for callers (checked Sendable implies unchecked
  Sendable from the outside), so nothing downstream can break.

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLX/ErrorHandler.swift`, change:

```swift
import Cmlx
import Foundation
```

to:

```swift
import Cmlx
import Foundation
import Synchronization
```

- [ ] **Step 2: Rewrite `ErrorBox` to use `Mutex`**

Change:

```swift
public final class ErrorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _firstError: Error?

    /// The first error encountered, if any.
    public var firstError: Error? {
        get {
            lock.withLock { _firstError }
        }
        set {
            lock.withLock {
                if _firstError == nil {
                    _firstError = newValue
                }
            }
        }
    }

    /// Throw the ``firstError`` if set, otherwise do nothing.
    public func check() throws {
        if let _firstError {
            throw _firstError
        }
    }
}
```

to:

```swift
public final class ErrorBox: Sendable {
    private let _firstError = Mutex<Error?>(nil)

    /// The first error encountered, if any.
    public var firstError: Error? {
        get {
            _firstError.withLock { $0 }
        }
        set {
            _firstError.withLock { stored in
                if stored == nil {
                    stored = newValue
                }
            }
        }
    }

    /// Throw the ``firstError`` if set, otherwise do nothing.
    public func check() throws {
        if let firstError {
            throw firstError
        }
    }
}
```

- [ ] **Step 3: Rewrite the private `ErrorHandler` class to use `Mutex`**

Change:

```swift
private final class ErrorHandler: @unchecked Sendable {

    /// task local error handler stack, if any
    @TaskLocal static var errorHandler: [@Sendable (String) -> Void] = []

    /// the global handler, if any -- this is called if there is no task local error handler
    let lock = NSLock()
    var globalHandler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)? =
        nil
    var globalData: UnsafeMutableRawPointer? = nil
    var globalDtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil

    init() {
    }

    deinit {
        if let globalData = self.globalData, let globalDtor = self.globalDtor {
            globalDtor(globalData)
        }
    }

    func setGlobalHandler(
        _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
        data: UnsafeMutableRawPointer? = nil,
        dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
    ) {
        lock.withLock {
            if let globalData = self.globalData, let globalDtor = self.globalDtor {
                globalDtor(globalData)
            }
            globalHandler = handler
            globalData = data
            globalDtor = dtor
        }
    }

    /// entry point when an error is encountered in the C++ MLX layer
    func dispatch(_ message: String) {
        if let handler = Self.errorHandler.last {
            handler(message)
        } else {
            lock.withLock {
                if let globalHandler {
                    globalHandler(message, globalData)
                } else {
                    fatalError(message)
                }
            }
        }
    }
```

to:

```swift
private final class ErrorHandler: Sendable {

    /// task local error handler stack, if any
    @TaskLocal static var errorHandler: [@Sendable (String) -> Void] = []

    /// the global handler state, if any -- this is used if there is no task local error handler
    private struct GlobalHandler {
        var handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?
        var data: UnsafeMutableRawPointer?
        var dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)?
    }

    private let global = Mutex(GlobalHandler(handler: nil, data: nil, dtor: nil))

    init() {
    }

    deinit {
        global.withLock { state in
            if let data = state.data, let dtor = state.dtor {
                dtor(data)
            }
        }
    }

    func setGlobalHandler(
        _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
        data: UnsafeMutableRawPointer? = nil,
        dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
    ) {
        global.withLock { state in
            if let oldData = state.data, let oldDtor = state.dtor {
                oldDtor(oldData)
            }
            state = GlobalHandler(handler: handler, data: data, dtor: dtor)
        }
    }

    /// entry point when an error is encountered in the C++ MLX layer
    func dispatch(_ message: String) {
        if let handler = Self.errorHandler.last {
            handler(message)
        } else {
            global.withLock { state in
                if let handler = state.handler {
                    handler(message, state.data)
                } else {
                    fatalError(message)
                }
            }
        }
    }
```

Leave the rest of the file (`withErrorHandler`, `withError` methods below this point) untouched — they don't reference `lock` directly.

- [ ] **Step 4: Build and run the error-handling tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/ErrorTests`

Expected: PASS (`testErrorHandler`, `testWithErrorCheck`, `testWithErrorThrow`, `testWithErrorThrowAsync`, `testWithErrorThrowNested`).

- [ ] **Step 5: Commit**

```bash
git add Source/MLX/ErrorHandler.swift
git commit -m "Use Synchronization.Mutex for ErrorBox/ErrorHandler; drop to checked Sendable"
```

---

### Task 4: `CompiledFunction` — Mutex swap (stays `@unchecked Sendable`)

**Files:**
- Modify: `Source/MLX/Transforms+Compile.swift`
- Test: `Tests/MLXTests/TransformTests.swift`, `Tests/MLXTests/OptimizerTests.swift` (existing)

**Interfaces:**
- Produces: no change to `compile(inputs:outputs:shapeless:_:)` overloads.

- [ ] **Step 1: Add the rationale comment and swap the lock declaration**

Change:

```swift
// Note: this is all immutable state -- the `id` property is only set at init time
final class CompiledFunction: @unchecked (Sendable) {

    /// unique (for the lifetime of the object) identifier for the compiled function
    private var id: UInt!

    let lock = NSLock()
```

to:

```swift
// `@unchecked Sendable`: `f`, `inputs`, and `outputs` are plain (non-`@Sendable`) stored
// values used directly outside of `lock` (during `init`), so the compiler can't verify
// this structurally even though `call(_:)` fully serializes access via `lock`.
final class CompiledFunction: @unchecked (Sendable) {

    /// unique (for the lifetime of the object) identifier for the compiled function
    private var id: UInt!

    let lock = Mutex(())
```

- [ ] **Step 2: Add the `Synchronization` import**

Change:

```swift
import Cmlx
import Foundation
```

to:

```swift
import Cmlx
import Foundation
import Synchronization
```

- [ ] **Step 3: Update the `call` method to use `Mutex`'s closure shape**

Change:

```swift
    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        lock.withLock {
            innerCall(arguments)
        }
    }
```

to:

```swift
    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        lock.withLock { _ in
            innerCall(arguments)
        }
    }
```

- [ ] **Step 4: Build and run the compile tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/TransformTests -only-testing:MLXTests/OptimizerTests`

Expected: PASS (`TransformTests.testCompile`, `testCompileWithCapturedState`, and the `compile()` usage inside `OptimizerTests`).

- [ ] **Step 5: Commit**

```bash
git add Source/MLX/Transforms+Compile.swift
git commit -m "Use Synchronization.Mutex for CompiledFunction's call serialization"
```

---

### Task 5: `_CustomFunctionState` — Mutex swap + new test coverage

`CustomFunction` currently has **zero** test coverage anywhere in the repo. Add
real coverage as part of validating this refactor.

**Files:**
- Modify: `Source/MLX/MLXCustomFunction.swift`
- Create: `Tests/MLXTests/CustomFunctionTests.swift`

**Interfaces:**
- Consumes: `grad(_ f: @escaping (MLXArray) -> MLXArray) -> (MLXArray) -> MLXArray` (`Source/MLX/Transforms+Grad.swift:51-56`).
- Produces: no change to `CustomFunction { ... }` / `Forward { ... }` / `VJP { ... }` signatures.

- [ ] **Step 1: Add the rationale comment and swap the lock declaration**

In `Source/MLX/MLXCustomFunction.swift`, change:

```swift
final class _CustomFunctionState: @unchecked Sendable {

    private let lock = NSLock()
```

to:

```swift
// `@unchecked Sendable`: `forwardFn` and `vjpFn` are plain (non-`@Sendable`) stored
// closures used directly during `buildClosures()` at init, so the compiler can't verify
// this structurally even though `call(_:)` fully serializes access via `lock`.
final class _CustomFunctionState: @unchecked Sendable {

    private let lock = Mutex(())
```

- [ ] **Step 2: Add the `Synchronization` import**

Change:

```swift
import Cmlx
import Foundation
```

to:

```swift
import Cmlx
import Foundation
import Synchronization
```

- [ ] **Step 3: Update the `call` method**

Change:

```swift
    func call(_ inputs: [MLXArray]) -> [MLXArray] {
        lock.withLock {
            let inVec = new_mlx_vector_array(inputs)
            defer { mlx_vector_array_free(inVec) }

            var outVec = mlx_vector_array_new()
            defer { mlx_vector_array_free(outVec) }

            let status = mlx_closure_apply(&outVec, combined, inVec)
            precondition(status == 0, "mlx_closure_apply failed (\(status))")

            return mlx_vector_array_values(outVec)
        }
    }
```

to:

```swift
    func call(_ inputs: [MLXArray]) -> [MLXArray] {
        lock.withLock { _ in
            let inVec = new_mlx_vector_array(inputs)
            defer { mlx_vector_array_free(inVec) }

            var outVec = mlx_vector_array_new()
            defer { mlx_vector_array_free(outVec) }

            let status = mlx_closure_apply(&outVec, combined, inVec)
            precondition(status == 0, "mlx_closure_apply failed (\(status))")

            return mlx_vector_array_values(outVec)
        }
    }
```

- [ ] **Step 4: Write the new test file**

Create `Tests/MLXTests/CustomFunctionTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class CustomFunctionTests: XCTestCase {

    func testCustomFunctionForwardOnly() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * 2] }
        }

        let result = f([MLXArray(Float(3))])
        XCTAssertEqual(result[0].item(), Float(6))
    }

    func testCustomFunctionForwardAndVJP() {
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * inputs[0]] }
            VJP { inputs, cotangents in [cotangents[0] * 2 * inputs[0]] }
        }

        let gradFn = grad { x in f([x])[0] }
        let dfdx = gradFn(MLXArray(Float(3)))

        // d/dx x^2 = 2x = 6 at x=3
        XCTAssertEqual(dfdx.item(), Float(6))
    }
}
```

- [ ] **Step 5: Build and run the new tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/CustomFunctionTests`

Expected: PASS, both tests. This is new coverage (there was none before), so
also sanity-check it's exercising real behavior: temporarily change the `* 2`
in `testCustomFunctionForwardOnly` to `* 3` and confirm the test now fails,
then change it back before committing.

- [ ] **Step 6: Commit**

```bash
git add Source/MLX/MLXCustomFunction.swift Tests/MLXTests/CustomFunctionTests.swift
git commit -m "Use Synchronization.Mutex for _CustomFunctionState; add CustomFunction test coverage"
```

---

### Task 6: `RandomState` — Mutex swap (stays `@unchecked Sendable`)

**Files:**
- Modify: `Source/MLX/State.swift`
- Test: `Tests/MLXTests/MLXRandomTests.swift`, `Tests/MLXTests/TransformTests.swift` (existing)

**Interfaces:**
- Produces: no change to `MLXRandom.RandomState` public API (`init()`, `init(seed:)`, `next()`, `seed(_:)`, `innerState()`, `asRandomKey()`).

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLX/State.swift`, change:

```swift
import Foundation

/// Protocol for types that can be used as a provider of random keys, e.g. for ``MLXRandom``.
```

to:

```swift
import Foundation
import Synchronization

/// Protocol for types that can be used as a provider of random keys, e.g. for ``MLXRandom``.
```

- [ ] **Step 2: Rewrite `RandomState` to use `Mutex<MLXArray>`**

Change:

```swift
    public class RandomState: RandomStateOrKey, Updatable, Evaluatable, @unchecked (Sendable) {
        private var state: MLXArray
        private let lock = NSLock()

        /// Initialize the RandomState with a seed based on the current time.
        public init() {
            let now = DispatchTime.now().uptimeNanoseconds
            state = MLXRandom.key(now)
        }

        /// Initialize the RandomState with the given seed value.
        public init(seed: UInt64) {
            state = MLXRandom.key(seed)
        }

        public func innerState() -> [MLXArray] {
            lock.withLock {
                [state]
            }
        }

        /// Split the current state and return a new Key.
        public func next() -> MLXArray {
            lock.withLock {
                let (a, b) = MLXRandom.split(key: state)
                self.state = a
                return b
            }
        }

        /// Reset the random state.
        public func seed(_ seed: UInt64) {
            lock.withLock {
                state = MLXRandom.key(seed)
            }
        }

        public func asRandomKey() -> MLXArray {
            next()
        }
    }
```

to:

```swift
    // `@unchecked Sendable`: `state` is an `MLXArray`, which is not `Sendable`; all access
    // to it goes through `lock`. `RandomState` is a non-`final` public class, so a checked
    // `Sendable` conformance isn't available even though wrapping the state in `Mutex`
    // would otherwise permit it.
    public class RandomState: RandomStateOrKey, Updatable, Evaluatable, @unchecked (Sendable) {
        private let lock: Mutex<MLXArray>

        /// Initialize the RandomState with a seed based on the current time.
        public init() {
            let now = DispatchTime.now().uptimeNanoseconds
            lock = Mutex(MLXRandom.key(now))
        }

        /// Initialize the RandomState with the given seed value.
        public init(seed: UInt64) {
            lock = Mutex(MLXRandom.key(seed))
        }

        public func innerState() -> [MLXArray] {
            lock.withLock { [$0] }
        }

        /// Split the current state and return a new Key.
        public func next() -> MLXArray {
            lock.withLock { state in
                let (a, b) = MLXRandom.split(key: state)
                state = a
                return b
            }
        }

        /// Reset the random state.
        public func seed(_ seed: UInt64) {
            lock.withLock { $0 = MLXRandom.key(seed) }
        }

        public func asRandomKey() -> MLXArray {
            next()
        }
    }
```

- [ ] **Step 3: Build and run the random-state tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/MLXRandomTests -only-testing:MLXTests/TransformTests`

Expected: PASS (`MLXRandomTests.testRandomStateOrKeySame`, `testRandomStateOrKeyDifferent`, plus the `withRandomState` usage in `TransformTests`).

- [ ] **Step 4: Commit**

```bash
git add Source/MLX/State.swift
git commit -m "Use Synchronization.Mutex for MLXRandom.RandomState"
```

---

### Task 7: `MLXNN.Cache` — Mutex swap, `final`, checked `Sendable` + new test coverage

`Cache` (used by `ALiBi`) has no direct or indirect test coverage today —
`ALiBi` itself isn't exercised anywhere in `Tests/`. Add coverage as part of
validating this refactor.

**Files:**
- Modify: `Source/MLXNN/Cache.swift`
- Create: `Tests/MLXTests/PositionalEncodingTests.swift`

**Interfaces:**
- Consumes: `ALiBi` (`Source/MLXNN/PositionalEncoding.swift:144-197`), specifically `public func callAsFunction(attentionScores: MLXArray, offset: Int = 0, mask: MLXArray? = nil) -> MLXArray`.
- Produces: `Cache<Key, Element>` becomes `final class Cache<Key: Hashable, Element>: Sendable` (was `class Cache<Key: Hashable, Element>: @unchecked (Sendable)`). `Cache` is `internal`, not part of the public API, and has no subclasses anywhere in the repo (confirmed during planning), so this is safe.

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLXNN/Cache.swift`, change:

```swift
import Foundation

/// Simple cache for holding prepared MLXArrays, etc.
```

to:

```swift
import Foundation
import Synchronization

/// Simple cache for holding prepared MLXArrays, etc.
```

- [ ] **Step 2: Rewrite `Cache` to use `Mutex`**

Change:

```swift
class Cache<Key: Hashable, Element>: @unchecked (Sendable) {

    let queue = DispatchQueue(label: "Cache")

    let maxSize: Int

    struct Entry {
        let value: Element
        let serial: Int
    }

    var contents = [Key: Entry]()
    var serial = 0

    init(maxSize: Int = 10) {
        self.maxSize = maxSize
    }

    subscript(key: Key) -> Element? {
        get {
            queue.sync {
                contents[key]?.value
            }
        }
        set {
            // store the key, value pair keeping the count <= maxSize
            queue.sync {
                if let newValue {
                    // handle wrap on the serial number
                    if serial == Int.max {
                        contents.removeAll()
                        serial = 0
                    }
                    contents[key] = Entry(value: newValue, serial: serial)
                    serial += 1

                    // if too large, remove oldest
                    if contents.count > maxSize {
                        let minKey = contents.min { lhs, rhs in
                            lhs.value.serial < rhs.value.serial
                        }?.key
                        if let minKey {
                            contents[minKey] = nil
                        }
                    }
                } else {
                    contents[key] = nil
                }
            }
        }
    }
}
```

to:

```swift
final class Cache<Key: Hashable, Element>: Sendable {

    struct Entry {
        let value: Element
        let serial: Int
    }

    private struct State {
        var contents = [Key: Entry]()
        var serial = 0
    }

    let maxSize: Int
    private let state = Mutex(State())

    init(maxSize: Int = 10) {
        self.maxSize = maxSize
    }

    subscript(key: Key) -> Element? {
        get {
            state.withLock { $0.contents[key]?.value }
        }
        set {
            // store the key, value pair keeping the count <= maxSize
            state.withLock { state in
                if let newValue {
                    // handle wrap on the serial number
                    if state.serial == Int.max {
                        state.contents.removeAll()
                        state.serial = 0
                    }
                    state.contents[key] = Entry(value: newValue, serial: state.serial)
                    state.serial += 1

                    // if too large, remove oldest
                    if state.contents.count > maxSize {
                        let minKey = state.contents.min { lhs, rhs in
                            lhs.value.serial < rhs.value.serial
                        }?.key
                        if let minKey {
                            state.contents[minKey] = nil
                        }
                    }
                } else {
                    state.contents[key] = nil
                }
            }
        }
    }
}
```

- [ ] **Step 3: Write the new test file**

Create `Tests/MLXTests/PositionalEncodingTests.swift`:

```swift
// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class PositionalEncodingTests: XCTestCase {

    /// Exercises `ALiBi`, which is the only consumer of the internal `Cache` type
    /// (`Source/MLXNN/Cache.swift`). Calling it twice with identical shape/dtype/offset
    /// hits the cached path on the second call; this validates the Mutex-backed cache
    /// still returns a correct, consistent result.
    func testALiBiCaching() {
        let alibi = ALiBi()
        let scores = MLXArray.zeros([1, 4, 5, 5])

        let first = alibi(attentionScores: scores)
        let second = alibi(attentionScores: scores)

        XCTAssertEqual(first.shape, [1, 4, 5, 5])
        XCTAssertTrue(allClose(first, second).all().item())
    }
}
```

- [ ] **Step 4: Build and run the tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/PositionalEncodingTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Source/MLXNN/Cache.swift Tests/MLXTests/PositionalEncodingTests.swift
git commit -m "Use Synchronization.Mutex for MLXNN.Cache; make it final Sendable; add ALiBi test coverage"
```

---

### Task 8: `WiredMemoryTicket.EndOnceGuard` — Mutex swap, checked `Sendable`

**Files:**
- Modify: `Source/MLX/WiredMemory.swift`
- Test: `Tests/MLXTests/WiredMemoryTests.swift` (existing)

**Interfaces:**
- Produces: no change to `WiredMemoryTicket`, `WiredMemoryManager`, or `withWiredLimit` public API.

- [ ] **Step 1: Add the `Synchronization` import**

In `Source/MLX/WiredMemory.swift`, change:

```swift
import Cmlx
import Foundation
```

to:

```swift
import Cmlx
import Foundation
import Synchronization
```

- [ ] **Step 2: Rewrite `EndOnceGuard` to use `Mutex`**

Change:

```swift
    private final class EndOnceGuard: @unchecked Sendable {
        private var _ended = false
        private let _lock = NSLock()

        /// Returns `true` exactly once; all subsequent calls return `false`.
        func tryMark() -> Bool {
            _lock.lock()
            defer { _lock.unlock() }
            if _ended { return false }
            _ended = true
            return true
        }
    }
```

to:

```swift
    private final class EndOnceGuard: Sendable {
        private let _ended = Mutex(false)

        /// Returns `true` exactly once; all subsequent calls return `false`.
        func tryMark() -> Bool {
            _ended.withLock { ended in
                if ended { return false }
                ended = true
                return true
            }
        }
    }
```

- [ ] **Step 3: Build and run the wired-memory tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/WiredMemoryTests`

Expected: PASS (skips are fine on non-GPU devices; the important thing is nothing errors or crashes).

- [ ] **Step 4: Commit**

```bash
git add Source/MLX/WiredMemory.swift
git commit -m "Use Synchronization.Mutex for WiredMemoryTicket.EndOnceGuard"
```

---

### Task 9: Sendable rationale comments, `MLXFastKernel`'s `evalLock` gap, `evalLock` doc comment

**Files:**
- Modify: `Source/MLX/Stream.swift`
- Modify: `Source/MLX/MLXFastKernel.swift`
- Modify: `Source/MLX/Transforms+Eval.swift`
- Test: `Tests/MLXTests/StreamTests.swift`, `Tests/MLXTests/MLXFastKernelTests.swift` (existing)

**Interfaces:**
- Produces: no signature changes anywhere in this task. `MLXFastKernel.callAsFunction` gets one behavioral fix (see Step 2).

- [ ] **Step 1: Add a `Sendable` rationale comment to `Stream`**

In `Source/MLX/Stream.swift`, change:

```swift
public final class Stream: @unchecked Sendable, Equatable {
```

to:

```swift
// `@unchecked Sendable`: `ctx` (`mlx_stream`) is `let` and never mutated after `init` --
// the imported C struct isn't recognized as `Sendable` by the compiler, but sharing an
// immutable, already-constructed `Stream` across threads is safe.
public final class Stream: @unchecked Sendable, Equatable {
```

- [ ] **Step 2: Add a `Sendable` rationale comment to `MLXFastKernel` (Metal branch) and fix the missing `evalLock`**

In `Source/MLX/MLXFastKernel.swift`, change:

```swift
        final public class MLXFastKernel: @unchecked Sendable {
            let kernel: mlx_fast_metal_kernel
            public let outputNames: [String]
```

to:

```swift
        // `@unchecked Sendable`: `kernel` (`mlx_fast_metal_kernel`) is `let` and never
        // mutated after `init` -- the imported C struct isn't recognized as `Sendable` by
        // the compiler, but sharing an immutable, already-constructed kernel across
        // threads is safe.
        final public class MLXFastKernel: @unchecked Sendable {
            let kernel: mlx_fast_metal_kernel
            public let outputNames: [String]
```

Then, in the same file, change (this is the actual behavioral fix -- every other
call that touches the backend graph goes through `evalLock`; this one didn't):

```swift
                mlx_fast_metal_kernel_config_set_verbose(config, verbose)

                let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
                defer { mlx_vector_array_free(inputs) }

                var result = mlx_vector_array_new()
                mlx_fast_metal_kernel_apply(&result, kernel, inputs, config, stream.ctx)
                defer { mlx_vector_array_free(result) }

                return mlx_vector_array_values(result)
```

to:

```swift
                mlx_fast_metal_kernel_config_set_verbose(config, verbose)

                let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
                defer { mlx_vector_array_free(inputs) }

                var result = mlx_vector_array_new()
                evalLock.withLock {
                    mlx_fast_metal_kernel_apply(&result, kernel, inputs, config, stream.ctx)
                }
                defer { mlx_vector_array_free(result) }

                return mlx_vector_array_values(result)
```

- [ ] **Step 3: Add a `Sendable` rationale comment to the non-Metal stub `MLXFastKernel`**

In the same file, change:

```swift
        final public class MLXFastKernel: @unchecked Sendable {
            public let outputNames: [String]

            init(
                name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
                source: String, header: String = "",
                ensureRowContiguous: Bool = true,
                atomicOutputs: Bool = false
            ) {
                self.outputNames = []
                fatalError("MLXFastKernel is not available without Metal")
            }
```

to:

```swift
        // `@unchecked Sendable`: stub type with no functional state; matches the
        // Metal-backed `MLXFastKernel` above for API parity on platforms without Metal.
        final public class MLXFastKernel: @unchecked Sendable {
            public let outputNames: [String]

            init(
                name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
                source: String, header: String = "",
                ensureRowContiguous: Bool = true,
                atomicOutputs: Bool = false
            ) {
                self.outputNames = []
                fatalError("MLXFastKernel is not available without Metal")
            }
```

- [ ] **Step 4: Document why `evalLock` stays a recursive lock**

In `Source/MLX/Transforms+Eval.swift`, change:

```swift
/// lock to be held while doing any eval or asyncEval.  This is
/// a recursive lock to handle any cases where a closure might
/// call back into eval.
let evalLock = NSRecursiveLock()
```

to:

```swift
/// lock to be held while doing any eval or asyncEval.  This is
/// a recursive lock to handle any cases where a closure might
/// call back into eval -- e.g. `CompiledFunction.innerCall` invokes the user's function
/// while control is inside `mlx_detail_compile`/`mlx_closure_apply`, and that user
/// function may itself call `eval()`. Do not replace this with `Synchronization.Mutex`:
/// `Mutex` is not reentrant, and swapping it in here would deadlock on that call path.
let evalLock = NSRecursiveLock()
```

- [ ] **Step 5: Build and run the affected tests**

Run: `swift build && xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests/StreamTests -only-testing:MLXTests/MLXFastKernelTests`

Expected: PASS. `MLXFastKernelTests` in particular validates the `evalLock` fix didn't
change kernel output (`testCustomKernelBasic`, `testCustomKernelArgs`, `testRoPEOutput`, etc.).

- [ ] **Step 6: Commit**

```bash
git add Source/MLX/Stream.swift Source/MLX/MLXFastKernel.swift Source/MLX/Transforms+Eval.swift
git commit -m "Document @unchecked Sendable rationale; hold evalLock during MLXFastKernel apply"
```

---

### Task 10: Full Swift 6 language mode in `Package.swift`

This is the task flagged in the design spec as having unpredictable fallout.
Follow the stop condition in Step 3 exactly.

**Files:**
- Modify: `Package.swift`

**Interfaces:**
- Produces: no change to any target's public product surface -- only the
  concurrency-checking mode used to compile it.

- [ ] **Step 1: Replace the experimental feature flag with full language mode**

In `Package.swift`, there are 7 identical occurrences of this block (one per
first-party target: `MLX`, `MLXRandom`, `MLXFast`, `MLXNN`, `MLXOptimizers`,
`MLXFFT`, `MLXLinalg`). Replace **all** of them:

Change (every occurrence):

```swift
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
```

to:

```swift
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
```

- [ ] **Step 2: Build**

Run: `swift build 2>&1 | tee /tmp/mlx-swift6-build.log`

- [ ] **Step 3: Triage any new diagnostics**

If the build in Step 2 **succeeds**, skip to Step 4.

If it **fails**, read every diagnostic in `/tmp/mlx-swift6-build.log` one at a
time. For each:

- If fixing it means changing code **internal** to a file (e.g. wrapping a
  file-private mutable static in a `Mutex`, adding `nonisolated(unsafe)` to a
  provably-safe internal `let`, or adding a documented `@unchecked Sendable`
  to an internal/private type) -- fix it using the same patterns as Tasks 1-9,
  rebuild, and repeat until clean.
- If fixing it would require changing a **public** declaration's signature
  (e.g. adding `@Sendable` to a public closure parameter, sealing a public
  class as `final` when it currently isn't, removing a public initializer, or
  adding a new generic constraint to a public API) -- **STOP**. Do not make
  the change. Revert this task's `Package.swift` edit
  (`git checkout -- Package.swift`), and report back exactly which public
  declaration is implicated, the diagnostic text, and why fixing it needs a
  public API change. This is the exact risk called out in the design spec
  ("Swift 6 language mode fallout is not fully predictable in advance") and
  needs an explicit decision, not a silent fix.

- [ ] **Step 4: Run the full test suite**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'`

Expected: PASS, no regressions from any of Tasks 1-9's changes surfacing under
the stricter mode.

- [ ] **Step 5: Commit**

```bash
git add Package.swift
git commit -m "Enable full Swift 6 language mode across first-party targets"
```

(If Step 3 required additional source fixes beyond `Package.swift`, `git add`
those files too and mention them in the commit body.)

---

### Task 11: Concurrency stress tests for the new `Mutex` paths

**Files:**
- Modify: `Tests/MLXTests/StreamTests.swift`
- Modify: `Tests/MLXTests/MemoryTests.swift`

**Interfaces:**
- Consumes: `Device.withDefaultDevice(_:_:)`, `Device.defaultDevice()`, `Memory.cacheLimit` (unchanged public API from Tasks 1-2).

- [ ] **Step 1: Add a concurrent `Device` test**

In `Tests/MLXTests/StreamTests.swift`, add inside `class StreamTests: XCTestCase { ... }`:

```swift
    func testDeviceDefaultConcurrentAccess() async {
        // Regression test for the Device._defaultDevice Mutex refactor (see
        // Source/MLX/Device.swift): hammer defaultDevice()/withDefaultDevice()
        // from many concurrent tasks. XCTest fails the run on a crash; running
        // under `--sanitize=thread` (Step 3) additionally catches data races.
        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 100 {
                group.addTask {
                    let device: Device = i.isMultiple(of: 2) ? .cpu : .gpu
                    Device.withDefaultDevice(device) {
                        _ = Device.defaultDevice()
                    }
                }
            }
        }
    }
```

- [ ] **Step 2: Add a concurrent `Memory` test**

In `Tests/MLXTests/MemoryTests.swift`, add inside `class MemoryTests: XCTestCase { ... }`:

```swift
    func testCacheLimitConcurrentAccess() async {
        // Regression test for the Memory.limits Mutex refactor (see
        // Source/MLX/Memory.swift): hammer cacheLimit get/set from many
        // concurrent tasks.
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< 100 {
                group.addTask {
                    Memory.cacheLimit = 1024 * (i + 1)
                    _ = Memory.cacheLimit
                }
            }
        }
    }
```

- [ ] **Step 3: Run under Thread Sanitizer**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -enableThreadSanitizer YES -only-testing:MLXTests/StreamTests`
Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -enableThreadSanitizer YES -only-testing:MLXTests/MemoryTests`

Expected: PASS with no TSan data-race reports. If TSan reports a race,
**stop and report it** -- do not suppress or work around a TSan finding
without understanding it first, since the entire point of this plan is to
make the concurrency story provably correct.

- [ ] **Step 4: Commit**

```bash
git add Tests/MLXTests/StreamTests.swift Tests/MLXTests/MemoryTests.swift
git commit -m "Add concurrency stress tests for Device and Memory Mutex-backed state"
```

---

### Task 12: Final verification pass

**Files:** none (verification only; only commit if Step 2 or 3 finds something to fix)

- [ ] **Step 1: Full test suite**

Run: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'`

Expected: PASS, all files.

- [ ] **Step 2: Release-config build**

Run: `swift build -c release`

Expected: succeeds (Sendable/concurrency diagnostics can occasionally differ
by optimization level; this confirms they don't here).

- [ ] **Step 3: Confirm no stray `NSLock`/`NSRecursiveLock`/`DispatchQueue` was left behind**

Read through the diffs on the branch (`git diff main --stat` then inspect each
changed file) and confirm the only remaining `NSRecursiveLock` in
`Source/MLX/` is `evalLock` in `Transforms+Eval.swift`, and no new
`NSLock`/`DispatchQueue` was introduced anywhere in this branch's changes.

- [ ] **Step 4: Report**

Summarize for the user: which of the 12 tasks completed cleanly, whether Task
10's Swift 6 language mode rollout required any stop-and-report deviation
(Task 10, Step 3), and confirmation that the full suite is green on
`modernize/concurrency`.
