// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

/// Body of the exit test: synchronous (it blocks on semaphores) and capture free.
private func raceFirstUseOfDefaultStreamAgainstEvalLock() {
    let holdingEvalLock = DispatchSemaphore(value: 0)
    let finished = DispatchGroup()

    // B: hold evalLock, give A time to enter the once and block on
    // evalLock, then touch the default stream ourselves
    finished.enter()
    Thread {
        withEvalLock {
            holdingEvalLock.signal()
            Thread.sleep(forTimeInterval: 0.5)
            _ = StreamOrDevice.default.stream
        }
        finished.leave()
    }.start()

    // A: first touch of the default stream while B holds evalLock
    finished.enter()
    Thread {
        holdingEvalLock.wait()
        _ = Stream.defaultStream
        finished.leave()
    }.start()

    if finished.wait(timeout: .now() + 10) == .timedOut {
        // deadlocked: bail out without running exit handlers, which
        // could themselves block on the stuck threads
        FileHandle.standardError.write(
            Data("deadlock: first use of the default stream vs evalLock\n".utf8))
        _exit(1)
    }
}

/// First use of the global streams must not deadlock against `evalLock`.
///
/// The global stream pair is a lazily initialized `static let`, so its
/// initializer runs under a `dispatch_once`.  If that initializer takes
/// `evalLock` (to create the `mlx_stream`), there is a lock-order inversion:
///
/// - thread A: first touch of the default stream -> inside the once ->
///   waits for `evalLock`
/// - thread B: holds `evalLock` (an eval, or a `grad` / `compile` trace) ->
///   first touch of the default stream (e.g. `StreamOrDevice.default` while
///   building an op) -> waits for the once
///
/// This can only happen on the very first touch in a process, so it runs as
/// an exit test: the body runs in a fresh child process where the globals
/// have not been initialized yet.
@Suite("Stream initialization")
struct StreamInitializationTests {

    @Test func firstUseOfDefaultStreamWhileAnotherThreadHoldsEvalLock() async {
        await #expect(processExitsWith: .success) {
            raceFirstUseOfDefaultStreamAgainstEvalLock()
        }
    }
}
