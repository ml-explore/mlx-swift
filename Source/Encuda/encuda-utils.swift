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
                // ECHILD means Foundation's own reaper won the race and has
                // already collected the child, so its bookkeeping is complete
                // and authoritative. Anything else is unexpected; fall back
                // the same way rather than inventing a status.
                return terminationStatus
            }
        }
    }
#endif
