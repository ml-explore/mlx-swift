import Foundation

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#elseif canImport(Darwin)
    import Darwin
#endif

func searchForCommand(_ name: String) -> URL? {
    let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
    for folder in path.split(separator: ":") {
        let url = URL(fileURLWithPath: String(folder)).appendingPathComponent(name)
        if FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
    }
    return nil
}

#if os(macOS) || os(Linux)
    extension Process {
        /// Waits for the spawned tool to finish and returns its exit status,
        /// reaping the child directly rather than going through Foundation.
        ///
        /// **Why this does not use `waitUntilExit()`.** On aarch64 Linux that
        /// call was observed never returning, and the previous workaround here
        /// — polling `isRunning` in a 50ms sleep loop — did not fix it. It
        /// only changed how the hang presented, because `isRunning` reads the
        /// same Foundation bookkeeping that stalls.
        ///
        /// Observed during a parallel SwiftPM build of the CUDA backend for
        /// linux/arm64: `nvcc` finishes and writes its output in full, then
        /// becomes a zombie that is never reaped, and `encuda-tool` spins in
        /// `hrtimer_nanosleep` indefinitely holding a completed 15MB result on
        /// disk. It is load-dependent — `--jobs 2` transpiled all 96 `.cu`
        /// files without incident, while an unconstrained build (SwiftPM
        /// defaults to every core) deadlocked — which is consistent with pipe
        /// file descriptors leaking between concurrently spawned siblings:
        /// `encuda-tool` was holding both the read and the write end of the
        /// same pipes, so no reader could ever observe EOF.
        ///
        /// `waitpid` answers the only question this code actually has — did
        /// the child exit, and with what status — without depending on any of
        /// that.
        ///
        /// - Returns: the child's exit code, or `128 + signal` if it was
        ///   terminated by a signal, matching the shell's convention.
        func waitForExitStatus() -> Int32 {
            var status: Int32 = 0
            while true {
                let reaped = waitpid(processIdentifier, &status, 0)
                if reaped == processIdentifier {
                    // Decode the wait status without the C macros, which are
                    // not imported into Swift. This encoding is shared by
                    // Linux and Darwin: the low 7 bits hold the terminating
                    // signal (0 when the process exited normally), and the
                    // next 8 bits hold the exit code.
                    let terminatingSignal = status & 0x7F
                    if terminatingSignal == 0 {
                        return (status >> 8) & 0xFF
                    }
                    return 128 + terminatingSignal
                }
                if reaped == -1 && errno == EINTR {
                    continue
                }
                // `waitpid` has nothing to reap. Overwhelmingly this is ECHILD:
                // Foundation runs its own reaper thread over every `Process` it
                // spawns, and on a parallel build it sometimes gets there
                // first. Whatever the reason, the child is gone and Foundation's
                // bookkeeping is now the only available source of truth.
                return foundationTerminationStatus()
            }
        }

        /// Foundation's view of the exit status, waited for rather than read
        /// straight away.
        ///
        /// `terminationStatus` is not safe to read the instant `waitpid`
        /// returns ECHILD. Foundation publishes the result in two steps, and
        /// reading the status inside the window before `isRunning` goes false
        /// **traps the process**:
        ///
        /// ```
        /// *** Program crashed: System trap at 0x0000ffff90b6fb50 ***
        /// Thread 0 "encuda-tool" crashed:
        ///   0  Process.terminationStatus.getter + 80 in libFoundation.so
        ///   1  Process.waitForExitStatus() + 107 in encuda-tool
        /// ```
        ///
        /// Observed on aarch64 Linux with Swift 6.3.1 during an unconstrained
        /// SwiftPM build of mlx's CUDA backend: one transpile in ~250 dies this
        /// way, which kills `swift build` with exit code 1 and *no diagnostic*,
        /// because the crash report goes to the transpiler's own output rather
        /// than surfacing as a build error.
        ///
        /// So wait for the flag — but on a deadline. An unbounded wait here
        /// would reintroduce exactly the hang this whole function exists to
        /// avoid. If the deadline expires, report a non-zero status instead:
        /// the caller turns that into a build failure, which is diagnosable,
        /// whereas a hang and a trap are not.
        private func foundationTerminationStatus() -> Int32 {
            let deadline = Date().addingTimeInterval(reaperSettleTimeout)
            while isRunning && Date() < deadline {
                usleep(1000)
            }
            guard !isRunning else {
                return reaperTimedOutStatus
            }
            return terminationStatus
        }

        /// How long to let Foundation finish publishing a child's exit status
        /// before giving up on it. Generous, because exceeding it means a build
        /// failure: the window being waited on is normally sub-millisecond.
        private var reaperSettleTimeout: TimeInterval { 30 }

        /// Reported when Foundation never publishes a status. Chosen from the
        /// range no compiler returns, so it is recognisable in a build log.
        private var reaperTimedOutStatus: Int32 { 254 }
    }
#endif
