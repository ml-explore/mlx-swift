// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX

// The ownership bookkeeping these tests read is debug-only.
#if DEBUG

    /// The `evalLock` → `CompiledFunction.lock` ordering invariant.
    ///
    /// `Source/CompileLockRepro` shows that the reverse order deadlocks.  It
    /// lives in its own executable because a failure wedges the process, and a
    /// passing run of it only shows that the race did not fire -- not that the
    /// ordering is right.  These tests assert the ordering itself, cannot hang,
    /// and are safe to run in the shared test process.
    ///
    /// A violation trips an `assert` at the point of acquisition, so a
    /// regression traps there rather than reaching the counter assertions
    /// below.  The counters are what make a *pass* meaningful: they show the
    /// check was reached at all, rather than the code path never running.
    class CompileLockOrderingTests: XCTestCase {

        override class func setUp() {
            setDefaultDevice()
        }

        /// The instrumentation must be able to report "not held", otherwise a
        /// violation count of zero would mean nothing.
        func testOwnershipTrackingIsNotVacuous() {
            XCTAssertFalse(EvalLockOwnership.isHeldByCurrentThread)
            XCTAssertEqual(EvalLockOwnership.depth, 0)

            withEvalLock {
                XCTAssertTrue(EvalLockOwnership.isHeldByCurrentThread)
                XCTAssertEqual(EvalLockOwnership.depth, 1)

                // evalLock is recursive; nesting is expected and must be counted
                withEvalLock {
                    XCTAssertEqual(EvalLockOwnership.depth, 2)
                }
                XCTAssertEqual(EvalLockOwnership.depth, 1)
            }

            XCTAssertFalse(EvalLockOwnership.isHeldByCurrentThread)
            XCTAssertEqual(EvalLockOwnership.depth, 0)
        }

        /// Ownership is per thread: another thread must not believe it holds the
        /// lock this one is holding.  (The other thread only reads its own
        /// bookkeeping -- it never waits on `evalLock` -- so this cannot hang.)
        func testOwnershipIsPerThread() {
            final class Result: @unchecked Sendable {
                var heldOnOtherThread = true
            }
            let result = Result()
            let done = DispatchSemaphore(value: 0)

            withEvalLock {
                let thread = Thread {
                    result.heldOnOtherThread = EvalLockOwnership.isHeldByCurrentThread
                    done.signal()
                }
                thread.start()
                XCTAssertEqual(done.wait(timeout: .now() + 10), .success)
            }

            XCTAssertFalse(result.heldOnOtherThread)
        }

        /// A compiled function with observed state really takes the per-instance
        /// lock -- a stateless one has nothing to guard and skips it -- so this
        /// exercises the checked acquisition.
        func testEvalLockIsHeldWhenTheInstanceLockIsTaken() {
            EvalLockOwnership.resetCounters()

            let state = MLXArray(2)
            func body(_ x: [MLXArray]) -> [MLXArray] {
                [x[0] + state]
            }
            let compiled = compile(inputs: [state], outputs: [state], body(_:))

            let r = compiled([MLXArray(5)])
            eval(r)
            XCTAssertEqual(r[0].item(Float.self), 7)

            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(
                counters.checks, 0,
                "the ordering invariant was never checked, so this test proves nothing")
            XCTAssertEqual(
                counters.violations, 0,
                "CompiledFunction.lock was taken without evalLock held")
        }

        /// The deadlock shape, single threaded: one compiled function called
        /// from inside another's trace.  The nested acquisition happens while an
        /// outer `evalLock` is already held, which is exactly the ordering the
        /// invariant describes.
        func testNestedCompiledCallKeepsTheOrdering() {
            EvalLockOwnership.resetCounters()

            let innerState = MLXArray(1)
            func innerBody(_ x: [MLXArray]) -> [MLXArray] {
                [x[0] * 2 + innerState]
            }
            let inner = compile(inputs: [innerState], outputs: [innerState], innerBody(_:))

            let outerState = MLXArray(3)
            nonisolated(unsafe) var tracedNested = false
            func outerBody(_ x: [MLXArray]) -> [MLXArray] {
                tracedNested = true
                return [inner([x[0]])[0] + outerState]
            }
            let outer = compile(inputs: [outerState], outputs: [outerState], outerBody(_:))

            let r = outer([MLXArray(5)])
            eval(r)
            XCTAssertEqual(r[0].item(Float.self), 14)

            XCTAssertTrue(
                tracedNested,
                "the traced body never ran, so no nested compiled call was exercised")

            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(counters.checks, 0)
            XCTAssertEqual(
                counters.violations, 0,
                "a nested CompiledFunction.lock was taken without evalLock held")
        }

        // MARK: - CompiledFunction.deinit vs. a concurrent compiled call

        /// A `CompiledFunction` whose only strong reference is `release()`-able
        /// on demand, so a test can make its `deinit` -- and therefore the
        /// compiler-cache erase -- happen at a chosen instant on a chosen
        /// thread.
        private final class Victim: @unchecked Sendable {
            private var compiled: (@Sendable (MLXArray) -> MLXArray)?

            init() {
                // A fresh compile() per instance, so the erase has a populated
                // cache entry to remove.  The unique constant keeps this from
                // sharing an entry with anything else.
                let bias = MLXArray(Float.random(in: 1 ... 2))
                compiled = MLX.compile { (x: MLXArray) in x * 3 + bias }
            }

            /// Populate the cache entry that `deinit` will erase.
            func warm() {
                let r = compiled!(MLXArray(1))
                eval(r)
            }

            /// Drop the last reference.  Returns once `deinit` has completed.
            func release() {
                compiled = nil
            }
        }

        /// The interleaving this is about: thread R's `deinit` erase lands while
        /// thread C is provably inside `compile_trace`, holding `evalLock` and
        /// holding a live reference into the cache entry it is filling in.
        ///
        /// Unlike `Source/CompileLockRepro`, this is safe in the shared test
        /// process: with the fix there is a single lock order, `evalLock` is
        /// only ever held for the duration of one call, and R's `deinit` cannot
        /// be what C is waiting for -- so R blocks for the length of the traced
        /// body and no longer.  The deadline turns a regression into a failure
        /// rather than a hung lane.
        func testDeinitEraseSerializesAgainstATraceInFlight() {
            EvalLockOwnership.resetCounters()

            final class Shared: @unchecked Sendable {
                let lock = NSLock()
                private var _insideTrace = false
                var insideTrace: Bool {
                    get { lock.withLock { _insideTrace } }
                    set { lock.withLock { _insideTrace = newValue } }
                }
                var tracedRan = false
                var insideTraceAfterErase = true
                var erasesAroundRelease = 0
                var result: Float = .nan
            }
            let shared = Shared()

            // Signalled from *inside* the traced body, so R only starts its
            // release once C is known to be inside the trace holding evalLock.
            let cIsInsideTrace = DispatchSemaphore(value: 0)
            let cDone = DispatchSemaphore(value: 0)
            let rDone = DispatchSemaphore(value: 0)

            // Built on this thread; released on R.
            let victim = Victim()
            victim.warm()

            // C's own compiled function is created and retained *here*, not on
            // C, so that C never destroys a `CompiledFunction` of its own --
            // the erase counted around `release()` below can then only be the
            // victim's.  Compiling does not trace; the trace happens on the
            // first call, which is on C.
            let state = MLXArray(Float(4))
            func body(_ x: [MLXArray]) -> [MLXArray] {
                shared.tracedRan = true
                shared.insideTrace = true
                cIsInsideTrace.signal()
                // Hold the window open long enough that R is certainly parked
                // in its erase rather than merely scheduled.
                usleep(50_000)
                shared.insideTrace = false
                return [x[0] + state]
            }
            let cCompiled = compile(inputs: [state], outputs: [state], body(_:))

            let c = Thread {
                let r = cCompiled([MLXArray(Float(6))])
                eval(r)
                shared.result = r[0].item(Float.self)
                cDone.signal()
            }

            let r = Thread {
                // Only proceed once C is inside the trace.
                guard cIsInsideTrace.wait(timeout: .now() + 30) == .success else {
                    rDone.signal()
                    return
                }
                let before = EvalLockOwnership.compileErases
                victim.release()
                let after = EvalLockOwnership.compileErases
                // Read the flag the instant the erase returns.  With the erase
                // under `evalLock` this can only be observed false: R cannot
                // have acquired the lock until C left the traced body.
                shared.insideTraceAfterErase = shared.insideTrace
                shared.erasesAroundRelease = after - before
                rDone.signal()
            }

            c.start()
            r.start()

            XCTAssertEqual(cDone.wait(timeout: .now() + 60), .success, "the compiled call wedged")
            XCTAssertEqual(rDone.wait(timeout: .now() + 60), .success, "the deinit erase wedged")

            XCTAssertTrue(shared.tracedRan, "the traced body never ran, so nothing was interleaved")
            XCTAssertEqual(
                shared.erasesAroundRelease, 1,
                "dropping the last reference did not run CompiledFunction.deinit, "
                    + "so no erase was interleaved with the trace")
            XCTAssertFalse(
                shared.insideTraceAfterErase,
                "the compiler-cache erase completed while another thread was still inside "
                    + "compile_trace -- the erase is not serialized against find/insert")

            XCTAssertEqual(shared.result, 10)
            withExtendedLifetime(cCompiled) {}

            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(counters.checks, 0)
            XCTAssertEqual(
                counters.violations, 0,
                "the compiler cache was mutated without evalLock held")
        }

        /// The same hazard without choreography: a stream of `deinit` erases
        /// against a stream of cold compiles, so erase runs against `find`'s
        /// `cache_[fun_id]` insert-and-rehash as well as against tracing.
        ///
        /// This one is a stress loop, so a pass is not by itself proof.  What it
        /// adds is that it cannot pass *vacuously*: the counters assert the
        /// erases happened and that every checked acquisition held `evalLock`.
        func testConcurrentDeinitErasesAndCompiledCalls() {
            EvalLockOwnership.resetCounters()

            let rounds = 200
            let start = DispatchSemaphore(value: 0)
            let ready = DispatchSemaphore(value: 0)
            let done = DispatchSemaphore(value: 0)

            final class Outcome: @unchecked Sendable {
                var callerValues = [Float]()
                var releases = 0
            }
            let outcome = Outcome()

            let caller = Thread {
                ready.signal()
                start.wait()
                var values = [Float]()
                for i in 0 ..< rounds {
                    // A fresh compile each round keeps the cache cold, so every
                    // round really performs an insert.
                    let compiled = MLX.compile { (x: MLXArray) in x * 2 + MLXArray(Float(i)) }
                    let r = compiled(MLXArray(Float(1)))
                    eval(r)
                    values.append(r.item(Float.self))
                }
                outcome.callerValues = values
                done.signal()
            }

            let releaser = Thread {
                ready.signal()
                start.wait()
                var released = 0
                for _ in 0 ..< rounds {
                    let victim = Victim()
                    victim.warm()
                    victim.release()
                    released += 1
                }
                outcome.releases = released
                done.signal()
            }

            caller.start()
            releaser.start()
            ready.wait()
            ready.wait()
            start.signal()
            start.signal()

            XCTAssertEqual(done.wait(timeout: .now() + 300), .success, "a thread wedged")
            XCTAssertEqual(done.wait(timeout: .now() + 300), .success, "a thread wedged")

            XCTAssertEqual(outcome.releases, rounds)
            XCTAssertEqual(outcome.callerValues.count, rounds)
            // Results must survive the erase traffic: a clobbered cache entry
            // would show up as a wrong value rather than only as a crash.
            for (i, value) in outcome.callerValues.enumerated() {
                XCTAssertEqual(value, Float(2 + i), "round \(i) produced \(value)")
            }

            XCTAssertGreaterThanOrEqual(
                EvalLockOwnership.compileErases, rounds,
                "the deinit erase path was never exercised")

            // Everything compiled here is stateless, so it skips the
            // per-instance lock: these checks all come from the erase path.
            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(
                counters.checks, 0,
                "no erase was checked for evalLock ownership, so this proves nothing")
            XCTAssertEqual(
                counters.violations, 0,
                "the compiler cache was mutated without evalLock held")
        }
    }

#endif
