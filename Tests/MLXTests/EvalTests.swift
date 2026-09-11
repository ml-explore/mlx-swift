// Copyright © 2026 Apple Inc.

import Foundation
import MLXNN
import MLXOptimizers
import Testing

@testable import MLX

// MARK: - Support

/// Run `body` once per thread on `count` real (non-cooperative) threads, all
/// released from a rendezvous so the calls actually overlap, and wait for them.
///
/// Real `Thread`s rather than a `TaskGroup`: `eval` blocks, and the point of
/// these tests is to have several threads blocked inside the eval machinery at
/// the same time.  Blocking the cooperative pool would make the amount of
/// overlap depend on the pool width instead of on the code under test.
///
/// - Returns: `true` if every thread finished within `timeout`.
@discardableResult
private func onThreads(
    _ count: Int, timeout: TimeInterval = 300, _ body: @escaping @Sendable (Int) -> Void
) -> Bool {
    // `ready` is a rendezvous rather than a completion group: it is entered
    // once per thread up front, and each thread leaves its own entry and then
    // waits for the group to drain -- so nobody starts until everybody has
    // arrived.  `finished` is the ordinary completion use.
    let ready = DispatchGroup()
    let finished = DispatchGroup()

    for i in 0 ..< count {
        ready.enter()
        finished.enter()

        let thread = Thread {
            ready.leave()
            // bounded, so a thread that never starts cannot wedge the others
            _ = ready.wait(timeout: .now() + 60)
            body(i)
            finished.leave()
        }
        thread.name = "EvalTests-\(i)"
        thread.stackSize = 4 << 20
        thread.start()
    }

    return finished.wait(timeout: .now() + timeout) == .success
}

/// Lock protected collection point for values produced on worker threads.
///
/// Only `Sendable` scalars are collected -- never an `MLXArray`.  `MLXArray` is
/// a reference type wrapping an `mlx_array` handle and is *not* thread safe:
/// `evalLock` serializes evaluation, not array lifetime or mutation.  Every
/// test below therefore builds and reads its arrays on a single thread, and
/// shares only the resulting numbers.
private final class Collected<Element: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var values = [Element]()

    func append(_ value: Element) {
        lock.withLock { values.append(value) }
    }

    var all: [Element] {
        lock.withLock { values }
    }
}

/// A `Bool` that can be set on one thread and read on another.
private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Bool

    init(_ value: Bool = false) {
        self.value = value
    }

    var isSet: Bool {
        lock.withLock { value }
    }

    func set(_ newValue: Bool) {
        lock.withLock { value = newValue }
    }
}

/// Has the array's data been computed?
///
/// Thin wrapper over the library's internal `isEvaluated`, reached through
/// `@testable import`.  Call this only on the thread that owns `array`.
private func isAvailable(_ array: MLXArray) -> Bool {
    MLX.isEvaluated(array)
}

/// A graph that takes a while to evaluate, used to open a window during which
/// another thread can be observed making progress (or not).
///
/// The values are chosen so the result is exact and independent of the number
/// of matmuls: ones × (ones / n) == ones, so the sum is always `size * size`.
private func heavyGraph(matmuls: Int, size: Int, stream: StreamOrDevice) -> MLXArray {
    var x = MLXArray.ones([size, size], stream: stream)
    let w = MLX.divide(
        MLXArray.ones([size, size], stream: stream), MLXArray(Float(size)), stream: stream)
    for _ in 0 ..< matmuls {
        x = MLX.matmul(x, w, stream: stream)
    }
    return x
}

private func timed(_ body: () -> Void) -> TimeInterval {
    let start = Date()
    body()
    return Date().timeIntervalSince(start)
}

// MARK: - Tests

/// Behavior of ``eval(_:)-(MLXArray...)``, ``asyncEval(_:)-(Collection<MLXArray>)`` and the
/// other `evalLock` holders when called from several threads at once.
///
/// `eval` schedules under `evalLock` (via `mlx_async_eval`) and then waits
/// *outside* it (via `mlx_eval`), so two properties need coverage:
///
/// - it is still synchronous -- results are available when it returns; and
/// - it no longer serializes the waiting, so unrelated work on other threads
///   makes progress while an eval is in flight.
///
/// Note what is *not* claimed: an individual `MLXArray` is not thread safe, so
/// each thread here owns the arrays it creates and only scalar results cross
/// threads.  What is under test is the shared global state behind eval --
/// the stream/scheduler, the compiler cache, the tracing state -- which
/// `evalLock` guards.
///
/// The suite is `.serialized` because several tests are stress or timing
/// shaped: running them against each other would make each one's measurement
/// depend on the others.
@Suite("eval + evalLock", .serialized)
struct EvalTests {

    init() {
        setDefaultDevice()
    }

    // MARK: Semantics

    /// `eval` must not return until the arrays are computed, even though the
    /// wait now happens outside `evalLock`.  `asyncEval` is included as the
    /// contrast: it is allowed to return early (and normally does, though that
    /// is a race, so only the "available after eval" direction is asserted).
    @Test func evalIsSynchronous() {
        for _ in 0 ..< 20 {
            let a = heavyGraph(matmuls: 20, size: 256, stream: .default)
            #expect(!isAvailable(a), "the graph was computed before eval was called")
            eval(a)
            #expect(isAvailable(a), "eval returned before the array was available")

            // variadic, collection and Any-collecting overloads all go through
            // the same schedule-then-wait path
            let b = heavyGraph(matmuls: 20, size: 256, stream: .default)
            let c = heavyGraph(matmuls: 20, size: 256, stream: .default)
            eval(b, c)
            #expect(isAvailable(b) && isAvailable(c))

            let d = heavyGraph(matmuls: 20, size: 256, stream: .default)
            eval([d])
            #expect(isAvailable(d))

            let e = heavyGraph(matmuls: 20, size: 256, stream: .default)
            eval(["key": e] as [String: MLXArray])
            #expect(isAvailable(e))

            let f = heavyGraph(matmuls: 20, size: 256, stream: .default)
            asyncEval(f)
            eval(f)
            #expect(isAvailable(f), "eval after asyncEval must still wait")
        }
    }

    /// `eval` of an already evaluated array, and repeated evals of the same
    /// array, must be harmless -- the double schedule (`async_eval` then
    /// `eval`) must not resubmit or fail on a computed graph.
    @Test func repeatedEvalIsIdempotent() {
        let a = MLXArray(0 ..< 8, [2, 4]).asType(.float32) * 2
        for _ in 0 ..< 10 {
            eval(a)
            #expect(isAvailable(a))
        }
        #expect(a.sum().item(Float.self) == 56)
    }

    // MARK: Concurrency, correctness

    /// The baseline: many threads, each building and evaluating its own graph.
    /// Every thread's result must be its own -- a graph or vector mixed up in
    /// the shared scheduler state shows up as a wrong value, not just as a
    /// crash.
    ///
    /// Note the shape used by every concurrent test here: worker threads only
    /// *collect*, and all assertions happen on the test's own thread.  A
    /// `#expect` evaluated on a raw `Thread` has no current-test context, so a
    /// failure there would not be attributed to the test.
    @Test func concurrentEvalOfIndependentGraphs() {
        let threads = 8
        let rounds = 50
        let results = Collected<[Float]>()
        let available = Collected<Bool>()

        let finished = onThreads(threads) { t in
            var values = [Float]()
            for round in 0 ..< rounds {
                let x = MLXArray(Float(t + 1))
                let y = MLXArray(Float(round))
                let r = x * 1000 + y
                eval(r)
                available.append(isAvailable(r))
                values.append(r.item(Float.self))
            }
            results.append(values)
        }

        #expect(finished, "a thread wedged in eval")
        #expect(available.all.allSatisfy { $0 }, "eval returned before the array was available")

        let all = results.all
        #expect(all.count == threads)
        // each thread's values identify it, so a mix-up is visible
        let expected = Set(
            (0 ..< threads).map { t in
                (0 ..< rounds).map { Float((t + 1) * 1000 + $0) }
            })
        #expect(Set(all) == expected)
    }

    /// The same graph shape evaluated on every thread at once.  Unlike the
    /// test above the threads compute *identical* values from identical
    /// (separately built) graphs, so all of them contend for the same kernels
    /// and cache entries while still owning their own arrays.
    @Test func concurrentEvalOfIdenticalGraphs() {
        let values = Collected<Float>()
        let available = Collected<Bool>()

        let finished = onThreads(8) { _ in
            for _ in 0 ..< 25 {
                let r = (MLXArray(0 ..< 64, [8, 8]).asType(.float32) * 3).sum()
                eval(r)
                available.append(isAvailable(r))
                values.append(r.item(Float.self))
            }
        }

        #expect(finished, "a thread wedged evaluating an identical graph")
        #expect(values.all.allSatisfy { $0 == 6048 }, "values: \(Set(values.all))")
        #expect(
            available.all.allSatisfy { $0 },
            "eval returned on some thread while its array was still unavailable")
    }

    /// `asyncEval` (schedule only) and `eval` (schedule then wait) submitted
    /// from different threads at the same time.  Each thread owns its arrays;
    /// what is shared is the scheduler they both push work into.
    @Test func concurrentAsyncEvalAndEval() {
        let values = Collected<Float>()
        let available = Collected<Bool>()

        let finished = onThreads(6) { t in
            for round in 0 ..< 30 {
                let a = heavyGraph(matmuls: 10, size: 128, stream: .default)
                let b = a.sum() + Float(round)

                switch t % 3 {
                case 0:
                    // schedule only, then wait for it later
                    asyncEval(b)
                    eval(b)
                case 1:
                    // schedule twice: the second must not lose the first
                    asyncEval(b)
                    asyncEval(b)
                    eval(b)
                default:
                    eval(b)
                }

                available.append(isAvailable(b))
                values.append(b.item(Float.self) - Float(round))
            }
        }

        #expect(finished, "a thread wedged mixing asyncEval and eval")
        #expect(
            available.all.allSatisfy { $0 },
            "eval returned before the array was available")
        #expect(values.all.count == 6 * 30)
        #expect(
            values.all.allSatisfy { $0 == Float(128 * 128) },
            "values: \(Set(values.all))")
    }

    /// `MLXArray.eval()`, `item()` and `description` take `evalLock`
    /// separately from the free `eval` functions; they must mix with it.
    @Test func concurrentItemDescriptionAndEval() {
        let items = Collected<Float>()
        let descriptions = Collected<Int>()

        let finished = onThreads(8) { t in
            for _ in 0 ..< 40 {
                let a = MLXArray(0 ..< 16, [4, 4]).asType(.float32)
                switch t % 4 {
                case 0:
                    let r = a.sum()
                    eval(r)
                    items.append(r.item(Float.self))
                case 1:
                    let r = a.sum()
                    r.eval()
                    items.append(r.item(Float.self))
                case 2:
                    // item() evaluates implicitly
                    items.append(a.sum().item(Float.self))
                default:
                    // description evaluates under the lock as well
                    descriptions.append((a * 1).description.count)
                }
            }
        }

        #expect(finished, "a thread wedged in item()/description")
        #expect(items.all.allSatisfy { $0 == 120 }, "unexpected values: \(Set(items.all))")
        #expect(descriptions.all.allSatisfy { $0 > 0 })
    }

    /// Each thread on its own stream.  This also exercises `Stream`'s
    /// creation, `deinit` and `synchronize`, all of which take `evalLock`, at
    /// the same time as evals are in flight on other streams.
    @Test func concurrentEvalAcrossStreams() {
        let results = Collected<Float>()

        let finished = onThreads(6) { t in
            for round in 0 ..< 20 {
                // a fresh stream each round, so Stream.init/deinit run against
                // the eval traffic as well
                let stream = Stream(round % 2 == 0 ? Device.gpu : Device.cpu)
                let s = StreamOrDevice.stream(stream)
                // MLXArray(_:) is a synchronous copy of host memory -- it has
                // no stream; only the operation is placed on one
                let x = MLXArray(Float(t + 1))
                let r = MLX.multiply(x, MLXArray(Float(7)), stream: s)
                eval(r)
                results.append(r.item(Float.self) / Float(t + 1))
                stream.synchronize()
            }
        }

        #expect(finished, "a thread wedged evaluating on its own stream")
        #expect(results.all.allSatisfy { $0 == 7 }, "values: \(Set(results.all))")
    }

    /// Compiled functions acquire `evalLock` and then their own per-instance
    /// lock, so they are the case where lock *ordering* matters.  Hammer them
    /// against plain evals, including stateful compiled functions whose state
    /// is evaluated concurrently.
    @Test func concurrentCompiledCallsAndEval() {
        #if DEBUG
            EvalLockOwnership.resetCounters()
        #endif

        let threads = 6
        let rounds = 40
        let results = Collected<[Float]>()
        let statelessResults = Collected<[Float]>()

        let finished = onThreads(threads) { t in
            // per-thread state and per-thread compiled function: sharing either
            // across threads would be a user-level data race on non-Sendable
            // values, not a lock bug
            let state = MLXArray(Float(t + 1))
            func body(_ x: [MLXArray]) -> [MLXArray] {
                [x[0] * 2 + state]
            }
            let compiled = compile(inputs: [state], outputs: [state], body(_:))
            let stateless = MLX.compile { (x: MLXArray) in x * 3 }

            var values = [Float]()
            var statelessValues = [Float]()
            for round in 0 ..< rounds {
                let r = compiled([MLXArray(Float(round))])
                eval(r)
                values.append(r[0].item(Float.self))

                let s = stateless(MLXArray(Float(round)))
                eval(s, state)
                statelessValues.append(s.item(Float.self))
            }
            results.append(values)
            statelessResults.append(statelessValues)
        }

        #expect(finished, "a thread wedged in a compiled call")

        // the stateless function is the same on every thread
        let statelessExpected = (0 ..< rounds).map { Float($0 * 3) }
        #expect(
            statelessResults.all.allSatisfy { $0 == statelessExpected },
            "a stateless compiled function produced wrong values")

        let all = results.all
        #expect(all.count == threads)
        let expected = Set(
            (0 ..< threads).map { t in
                (0 ..< rounds).map { Float($0 * 2 + t + 1) }
            })
        #expect(Set(all) == expected, "compiled results were mixed up across threads")

        #if DEBUG
            let counters = EvalLockOwnership.counters
            #expect(
                counters.checks > 0,
                "no lock ordering check was reached, so this test proves nothing")
            #expect(
                counters.violations == 0,
                "an inner lock was taken without evalLock held")
        #endif
    }

    /// `grad`/`vjp`/`vmap` trace under `evalLock` while `eval` no longer holds
    /// it for the whole wait; tracing must still be mutually exclusive, and a
    /// trace on one thread must not pick up another thread's graph.
    @Test func concurrentTracingAndEval() {
        #if DEBUG
            EvalLockOwnership.resetCounters()
        #endif

        let gradients = Collected<Float>()
        let vjps = Collected<Float>()
        let vmaps = Collected<Float>()

        let finished = onThreads(6) { t in
            for round in 0 ..< 25 {
                switch t % 3 {
                case 0:
                    // d/dx (x^2) at x = 3 is 6, regardless of what any other
                    // thread is tracing
                    let g = MLX.grad { (x: MLXArray) in (x * x).sum() }
                    let d = g(MLXArray(Float(3)))
                    eval(d)
                    gradients.append(d.item(Float.self))

                case 1:
                    let (_, cotangentProducts) = MLX.vjp(
                        { (x: [MLXArray]) in [x[0] * 3] },
                        primals: [MLXArray(Float(round))],
                        cotangents: [MLXArray(Float(1))])
                    eval(cotangentProducts)
                    vjps.append(cotangentProducts[0].item(Float.self))

                default:
                    let mapped = MLX.vmap { (x: MLXArray) in x * 2 }
                    let r = mapped(MLXArray(0 ..< 8).asType(.float32))
                    eval(r)
                    vmaps.append(r.sum().item(Float.self))
                }
            }
        }

        #expect(finished, "a thread wedged tracing")
        #expect(gradients.all.allSatisfy { $0 == 6 }, "gradients: \(Set(gradients.all))")
        #expect(vjps.all.allSatisfy { $0 == 3 }, "vjps: \(Set(vjps.all))")
        #expect(vmaps.all.allSatisfy { $0 == 56 }, "vmaps: \(Set(vmaps.all))")
        #expect(!gradients.all.isEmpty && !vjps.all.isEmpty && !vmaps.all.isEmpty)

        #if DEBUG
            #expect(EvalLockOwnership.counters.violations == 0)
        #endif
    }

    /// A whole training loop per thread: `valueAndGrad(model:)`, an optimizer
    /// update and `eval` over nested parameter dictionaries.  This is the
    /// realistic shape of concurrent eval use, and it checks progress (the
    /// loss must fall) rather than only absence of a crash.
    ///
    /// Each thread owns its model, optimizer and data; the inputs are
    /// deterministic rather than random so the global RNG state -- a separate
    /// thread-safety question -- is not part of the test.
    @Test func concurrentTrainingLoops() {
        final class Model: Module, UnaryLayer {
            let linear: Linear

            // fixed initial parameters, deliberately wrong, so training has
            // something to do and the loss trajectory is repeatable
            init(m: Float, b: Float) {
                linear = Linear(weight: MLXArray([m], [1, 1]), bias: MLXArray([b]))
            }

            func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
        }

        let threads = 4
        let losses = Collected<(Int, Float, Float)>()

        let finished = onThreads(threads) { t in
            let model = Model(m: 0, b: 0)
            eval(model)

            let optimizer = SGD(learningRate: 1e-2)
            func loss(model: Model, x: MLXArray, y: MLXArray) -> MLXArray {
                mseLoss(predictions: model(x), targets: y, reduction: .mean)
            }
            let lg = valueAndGrad(model: model, loss)

            // a different target per thread, so a leaked gradient or parameter
            // update from another thread would show up as a stalled loss
            let m = Float(t + 1)
            let b = Float(3)
            let x = MLXArray(Array(stride(from: Float(-2), to: Float(2), by: 0.25)), [16, 1])
            let y = m * x + b

            var first = Float.nan
            var last = Float.nan
            for step in 0 ..< 100 {
                let (l, grads) = lg(model, x, y)
                optimizer.update(model: model, gradients: grads)
                eval(model, optimizer)
                let value = l.item(Float.self)
                if step == 0 { first = value }
                last = value
            }
            losses.append((t, first, last))
        }

        #expect(finished, "a training thread wedged")

        let all = losses.all
        #expect(all.count == threads)
        for (t, first, last) in all {
            #expect(first.isFinite && last.isFinite, "thread \(t): \(first) -> \(last)")
            #expect(last < first, "thread \(t) did not train: \(first) -> \(last)")
        }
    }

    /// The other `evalLock` users -- memory/cache configuration, device
    /// queries, `tostring` -- run against a stream of evals.  These do not
    /// wait on the GPU, so all this asserts is that they interleave without
    /// deadlocking or corrupting results.
    @Test func concurrentEvalAndOtherEvalLockUsers() {
        let cacheLimit = Memory.cacheLimit
        let values = Collected<Float>()

        let finished = onThreads(6) { t in
            for round in 0 ..< 30 {
                switch t % 3 {
                case 0:
                    let r = heavyGraph(matmuls: 5, size: 128, stream: .default).sum()
                    eval(r)
                    values.append(r.item(Float.self))
                case 1:
                    _ = Memory.snapshot()
                    if round % 10 == 0 { Memory.clearCache() }
                    Memory.cacheLimit = cacheLimit
                default:
                    _ = Device.gpu.description
                    _ = Stream.gpu.description
                    _ = MLXArray(Float(round)).description
                }
            }
        }

        #expect(finished, "a thread wedged mixing eval with other evalLock users")
        #expect(values.all.allSatisfy { $0 == Float(128 * 128) }, "values: \(Set(values.all))")

        Memory.cacheLimit = cacheLimit
    }

    // MARK: Re-entrancy

    /// `evalLock` is recursive on purpose, so an `eval` reached from inside
    /// something that already holds it must not self-deadlock.  A regression
    /// here (a non-recursive lock, or a wait performed while holding it in a
    /// way that cannot complete) hangs, so this is kept small and time
    /// bounded.
    @Test func evalIsReentrantWhileHoldingEvalLock() {
        #if DEBUG
            #expect(EvalLockOwnership.depth == 0)
        #endif

        let done = DispatchSemaphore(value: 0)
        let value = Collected<Float>()
        let depthInsideEval = Collected<Int>()

        let thread = Thread {
            withEvalLock {
                let a = MLXArray(0 ..< 4).asType(.float32) * 2
                // eval() re-enters withEvalLock, and now also waits while the
                // outer level is still held
                eval(a)
                value.append(a.sum().item(Float.self))

                #if DEBUG
                    depthInsideEval.append(EvalLockOwnership.depth)
                #endif
            }
            done.signal()
        }
        thread.start()

        #expect(
            done.wait(timeout: .now() + 60) == .success,
            "eval deadlocked when called with evalLock already held")
        #expect(value.all == [12])

        #if DEBUG
            #expect(depthInsideEval.all == [1], "unbalanced evalLock ownership after a nested eval")
            #expect(EvalLockOwnership.depth == 0)
        #endif
    }

    /// `eval` inside a compiled function's traced body: the trace holds
    /// `evalLock`, so the inner eval is the re-entrant case reached through
    /// real API rather than through `withEvalLock` directly.
    ///
    /// The array evaluated inside the body is already computed and takes no
    /// part in the trace, so this stays a statement about locking rather than
    /// about evaluating a traced graph (which MLX does not allow).
    @Test func evalInsideACompiledBody() {
        nonisolated(unsafe) var traced = false
        let side = MLXArray(0 ..< 4).asType(.float32)
        eval(side)

        let compiled = MLX.compile { (x: MLXArray) in
            traced = true
            // re-enters eval (and therefore withEvalLock) while the trace holds it
            eval(side)
            return x * 2
        }

        let r = compiled(MLXArray(Float(3)))
        eval(r)

        #expect(traced, "the body never traced, so nothing was re-entered")
        #expect(r.item(Float.self) == 6)
    }

    // MARK: The reason for the change

    /// `eval` must not hold `evalLock` while waiting for the computation.
    ///
    /// Thread A evaluates a long graph on the GPU stream.  Thread B, on the
    /// CPU stream, does a trivial eval while A's is in flight.  With the wait
    /// inside the lock B cannot start until A finishes; with the wait outside
    /// it, B only needs the lock long enough to schedule.
    ///
    /// The check is not a bare timing threshold: A publishes a flag for the
    /// duration of its own `eval` call, and B verifies that flag is *still
    /// set* at the moment B's own eval returned, which establishes the overlap
    /// directly.  (B never touches A's arrays -- only the flag.)  Timing is
    /// used only to tell real serialization ("B took about as long as A") from
    /// A simply finishing too early to measure, which retries with more work.
    @Test func evalDoesNotHoldTheLockWhileWaiting() {
        // Calibrate: how many matmuls are needed for a ~2 second eval?
        let size = 1024
        let probeCount = 8
        let warm = heavyGraph(matmuls: probeCount, size: size, stream: .gpu)
        eval(warm)
        let probe = timed {
            let x = heavyGraph(matmuls: probeCount, size: size, stream: .gpu)
            eval(x)
        }
        let perMatmul = max(probe / Double(probeCount), 1e-5)
        var matmuls = min(max(Int(2.0 / perMatmul), probeCount), 20_000)

        for attempt in 0 ..< 4 {
            let heavyIsEvaluating = Flag()
            let aStarted = DispatchSemaphore(value: 0)
            let group = DispatchGroup()

            let heavyDurations = Collected<TimeInterval>()
            let smallDurations = Collected<TimeInterval>()
            let smallValues = Collected<Float>()
            let overlapped = Flag()

            let matmulCount = matmuls

            group.enter()
            let a = Thread {
                // built on this thread, and never touched by B
                let heavy = heavyGraph(matmuls: matmulCount, size: size, stream: .gpu)
                aStarted.signal()
                heavyIsEvaluating.set(true)
                let duration = timed { eval(heavy) }
                heavyIsEvaluating.set(false)
                heavyDurations.append(duration)
                group.leave()
            }

            group.enter()
            let b = Thread {
                guard aStarted.wait(timeout: .now() + 60) == .success else {
                    group.leave()
                    return
                }
                // let A get past scheduling and into its wait
                Thread.sleep(forTimeInterval: 0.25)

                let cpu = StreamOrDevice.cpu
                let x = MLXArray(Float(2))
                let r = MLX.multiply(x, MLXArray(Float(21)), stream: cpu)
                let duration = timed { eval(r) }
                // read immediately: if A is still inside its eval, B's eval
                // completed while A was waiting
                overlapped.set(heavyIsEvaluating.isSet)
                smallDurations.append(duration)
                smallValues.append(r.item(Float.self))
                group.leave()
            }

            a.start()
            b.start()

            #expect(group.wait(timeout: .now() + 600) == .success, "a thread wedged")
            #expect(smallValues.all == [42])

            guard let heavyDuration = heavyDurations.all.first,
                let smallDuration = smallDurations.all.first
            else {
                Issue.record("a thread did not report a duration")
                return
            }

            if overlapped.isSet {
                // B ran to completion while A's eval was still in flight
                #expect(
                    smallDuration < heavyDuration,
                    """
                    the small eval took \(smallDuration)s while the long eval took \
                    \(heavyDuration)s
                    """)
                return
            }

            // A finished before B could look.  Serialization and "A was simply
            // too fast" are distinguished by how long B waited.
            if smallDuration >= heavyDuration / 2 {
                Issue.record(
                    """
                    a trivial eval on another stream took \(smallDuration)s while a \
                    \(heavyDuration)s eval was in flight -- eval appears to hold evalLock while \
                    waiting
                    """)
                return
            }

            // grow the window and try again
            matmuls = min(matmuls * 4, 40_000)
            print(
                """
                evalDoesNotHoldTheLockWhileWaiting: attempt \(attempt) inconclusive \
                (long eval \(heavyDuration)s, small eval \(smallDuration)s); retrying with \
                \(matmuls) matmuls
                """)
        }

        Issue.record(
            """
            unable to keep a long eval in flight long enough to observe overlap -- \
            the concurrency of eval was not established by this run
            """)
    }
}
