// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Deterministic reproduction of a cross-thread lock-order inversion between
/// the per-instance `CompiledFunction.lock` and the process-global `evalLock`.
///
/// ## The cycle
///
/// `CompiledFunction.call` takes the per-instance `lock` (added by #226 to guard
/// compiled-function state) and then, inside `innerCall`, takes the global
/// `evalLock` (added by #339 to guard tracing) and holds it across
/// `mlx_closure_apply`.  On a *cold* cache entry `mlx_closure_apply` runs
/// `mlx::core::detail::compile_trace`, which calls back into the Swift body of
/// the function being compiled.  If that body calls a *different* compiled
/// function — which every `MLXNN` activation does, e.g. `silu` uses the
/// file-scope `compiledSilu` — the nested `CompiledFunction.call` takes a second
/// per-instance lock *while already holding `evalLock`*.
///
///     Thread H:  L_cf(outer)  -> evalLock     -> wants L_cf(shared)
///     Thread W:  L_cf(shared) -> wants evalLock
///
/// The two acquisition orders are inverted, so H and W deadlock permanently.
/// `evalLock` being an `NSRecursiveLock` does not help: this is not same-thread
/// re-entry.
///
/// ## Why this is deterministic and not a race we hope to lose
///
/// * Fresh `compile()` results per round guarantee a **cold** compiler cache.
///   The re-entry into Swift only happens while tracing, i.e. on first use, so a
///   warm instance would simply never open the window.
/// * A semaphore signalled **from inside the traced Swift body** releases W only
///   once H is provably inside `compile_trace` holding `evalLock`.
/// * A `usleep` in that same traced body holds the window open for 200 ms —
///   roughly a million times longer than the few instructions W needs — so W
///   always reaches `L_cf(shared)` first.
/// * Threads are owned here rather than borrowed from a test runner, so nothing
///   depends on runner parallelism.
///
/// ## Why it is its own executable
///
/// A wedged round deadlocks the threads permanently and they cannot be
/// unwedged, so this cannot live in a shared test process without taking the
/// whole lane down.  The deadline turns the hang into a hard, fast failure.
///
/// ## Running it
///
///     swift build --product CompileLockRepro
///     .build/debug/CompileLockRepro           # --device cpu|gpu
///
/// SwiftPM (command line) cannot build the Metal shaders, and the MLX scheduler
/// opens a GPU stream when it is constructed, so a `swift build` product needs a
/// `mlx.metallib` colocated with the binary even for `--device cpu`.  Copy one
/// built by `xcodebuild -project xcode/MLX.xcodeproj -scheme MLX` (it lands at
/// `Build/Products/Debug/Cmlx.framework/Versions/A/Resources/default.metallib`)
/// into `$(swift build --show-bin-path)/mlx.metallib`.
///
/// Exit codes: `0` no deadlock, `1` deadlock (the bug is present), `2`
/// inconclusive (compilation was disabled, so no trace ever ran).
@main
struct CompileLockRepro {

    /// Per-round completion deadline.  A round that has not finished by now is
    /// wedged: every thread is parked in `__psynch_mutexwait` and no amount of
    /// further waiting will help.
    static let deadline: TimeInterval = 10

    /// How long the traced body holds the race window open.
    static let widenMicroseconds: UInt32 = 200_000

    static let rounds = 20

    /// Minimal thread-safe boolean; the repro must not itself depend on any
    /// MLX locking.
    final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func set() {
            lock.lock()
            value = true
            lock.unlock()
        }
        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    static func die(_ code: Int32, _ message: String) -> Never {
        print(message)
        fflush(stdout)
        // The wedged threads hold `evalLock` and a `CompiledFunction.lock`.
        // A normal `exit()` would run atexit handlers that can block on exactly
        // those locks, turning a reported failure back into a hang.
        _exit(code)
    }

    static func main() {
        // The deadlock is entirely in the Swift locking + the backend-independent
        // `mlx::core::detail::compile` machinery, so the repro runs on either
        // device.  It defaults to `.cpu` because SwiftPM (command line) cannot
        // build the Metal shaders, so a plain `swift run` has no metallib.
        var device = Device.cpu
        if let index = CommandLine.arguments.firstIndex(of: "--device"),
            index + 1 < CommandLine.arguments.count
        {
            switch CommandLine.arguments[index + 1].lowercased() {
            case "cpu": device = .cpu
            case "gpu": device = .gpu
            default: die(2, "unknown --device \(CommandLine.arguments[index + 1])")
            }
        }
        Device.setDefault(device: device)

        print(
            "compile-lock repro: device=\(device), \(rounds) rounds, \(Int(deadline))s deadline per round"
        )

        // Keep every compiled function alive for the whole run: `id` is the
        // object address and `deinit` erases the cache entry, so recycling an
        // address could alias two rounds.
        var keepAlive: [Any] = []
        var tracedAtLeastOnce = false

        for round in 1 ... rounds {
            tracedAtLeastOnce = runRound(round, keepAlive: &keepAlive) || tracedAtLeastOnce
        }

        if !tracedAtLeastOnce {
            die(
                2,
                """
                INCONCLUSIVE: the traced body never ran, so no compile trace happened \
                and the deadlock window was never opened. Is MLX_DISABLE_COMPILE set?
                """)
        }

        print("PASS: \(rounds) rounds completed, no deadlock")
    }

    /// Runs one round.  Returns whether the traced body actually executed.
    static func runRound(_ round: Int, keepAlive: inout [Any]) -> Bool {
        let gate = DispatchSemaphore(value: 0)  // H (inside the trace) -> W
        let hDone = DispatchSemaphore(value: 0)
        let wDone = DispatchSemaphore(value: 0)
        let traced = Flag()  // the traced Swift body really ran
        let stop = Flag()  // main -> W: H is finished, you may stop

        // Fresh instances: a cold compiler cache is what makes `compile_trace`
        // — and therefore the re-entry into Swift under `evalLock` — happen.
        let shared: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
            x * 2 + 1
        }
        let outer: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
            // This body runs on thread H, on the C++ side of
            // `mlx_closure_apply`, inside `compile_trace`, with `evalLock` held.
            traced.set()
            gate.signal()  // W may now take L_cf(shared)
            usleep(widenMicroseconds)  // hold the window open
            return shared(x) + 3  // H blocks here on L_cf(shared)
        }
        keepAlive.append(shared)
        keepAlive.append(outer)

        let h = Thread {
            let a = MLXArray([1.0, 2.0, 3.0] as [Float])
            let r = outer(a)
            eval(r)
            hDone.signal()
        }
        h.name = "H-nested-compiled-call"

        let w = Thread {
            gate.wait()
            let b = MLXArray([4.0, 5.0, 6.0] as [Float])
            var iterations = 0
            repeat {
                let r = shared(b)  // W takes L_cf(shared), then wants evalLock
                eval(r)
                iterations += 1
            } while !stop.isSet && iterations < 1000
            wDone.signal()
        }
        w.name = "W-plain-compiled-call"

        h.start()
        w.start()

        if hDone.wait(timeout: .now() + deadline) == .timedOut {
            die(
                1,
                """
                FAIL: round \(round): thread H did not complete within \(Int(deadline))s.
                  H holds evalLock (taken in CompiledFunction.innerCall) and is blocked
                  acquiring the per-instance lock of the nested compiled function;
                  W holds that per-instance lock and is blocked acquiring evalLock.
                  enteredTrace=\(traced.isSet)
                  This is the lock-order inversion; the process is deadlocked.
                """)
        }
        stop.set()
        if wDone.wait(timeout: .now() + deadline) == .timedOut {
            die(
                1,
                """
                FAIL: round \(round): thread W did not complete within \(Int(deadline))s \
                after H finished. Blocked acquiring evalLock.
                """)
        }

        print("  round \(round): ok (enteredTrace=\(traced.isSet))")
        return traced.isSet
    }
}
