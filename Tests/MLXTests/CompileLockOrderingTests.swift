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
    }

#endif
