// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// Runs a test body in several processes that form a ring, the way the Python
/// distributed tests are run under a launcher.
///
/// The test process is itself rank 0.  It reserves a loopback port for every
/// rank, writes a temporary hostfile, re-executes the test bundle once for each
/// of the other ranks and then runs the same body, so no separate worker
/// executable is needed.
///
/// MLX caches the communication group per process, so a process can only join
/// one ring.  Each test class that needs ranks therefore has a single test
/// method and is run in its own `xctest` invocation; a second body in the same
/// process is skipped rather than left hanging against a ring that the first
/// one built.
enum DistributedHarness {

    /// Set in the environment of the ranks the harness spawns.
    ///
    /// `MLX_RANK` alone would not do: a whole test run can be launched with it
    /// already set, and then every process would wait to be spawned.
    static let spawnedRankVariable = "MLX_TEST_DISTRIBUTED_SPAWNED_RANK"

    /// Run `body` in `ranks` processes.
    ///
    /// - Parameters:
    ///   - ranks: how many processes take part, including this one
    ///   - testName: the `Class/method` selector every rank runs
    ///   - timeout: how long the spawned ranks may run before giving up
    ///   - body: the test body, which every rank runs with its own group
    static func run(
        ranks: Int, testName: String, timeout: TimeInterval = 120,
        body: (MLXDistributed.Group) throws -> Void
    ) throws {
        let environment = ProcessInfo.processInfo.environment

        if environment[spawnedRankVariable] == "1" {
            // spawned by rank 0 -- the hostfile and rank are already set
            try runRank(ranks: ranks, body)
            return
        }

        guard environment["MLX_TEST_DISTRIBUTED"] == "1" else {
            throw XCTSkip("Set MLX_TEST_DISTRIBUTED=1 to run the multi process tests.")
        }

        guard state.claim() else {
            throw XCTSkip(
                """
                Another multi process test already formed a group in this process.  \
                Run this class in its own xctest invocation.
                """)
        }

        guard let runner = testRunner() else {
            throw XCTSkip("Could not locate the test bundle to re-execute as a rank.")
        }

        guard let ports = reserveFreePorts(count: ranks) else {
            throw XCTSkip("Could not reserve \(ranks) loopback ports.")
        }

        let hostfile = try writeHostfile(ports: ports)
        defer { try? FileManager.default.removeItem(at: hostfile) }

        let spawned = (1 ..< ranks).map {
            makeRank($0, runner: runner, testName: testName, hostfile: hostfile)
        }
        let watchdog = Watchdog(spawned: spawned, timeout: timeout)
        defer {
            watchdog.disarm()
            for process in spawned where process.isRunning {
                process.terminate()
            }
        }
        for process in spawned {
            try process.run()
        }

        setenv("MLX_HOSTFILE", hostfile.path, 1)
        setenv("MLX_RANK", "0", 1)
        defer {
            unsetenv("MLX_HOSTFILE")
            unsetenv("MLX_RANK")
        }

        try runRank(ranks: ranks, body)

        for (index, process) in spawned.enumerated() {
            process.waitUntilExit()
            XCTAssertEqual(
                process.terminationStatus, 0,
                "rank \(index + 1) failed -- see its output above")
        }
    }

    /// Join the ring and run the body.
    ///
    /// The body is wrapped in `withError` so that an MLX error becomes a test
    /// failure naming the rank.  Without an active scope the global handler
    /// calls `fatalError()` and the message never reaches XCTest.
    private static func runRank(
        ranks: Int, _ body: (MLXDistributed.Group) throws -> Void
    ) throws {
        do {
            let group = try MLXDistributed.initialize(backend: .ring, strict: true)

            // a group of one makes every assertion about sharding pass without
            // proving anything, so the size is part of the contract
            print("[rank \(group.rank)] joined a group of size \(group.size)")
            XCTAssertEqual(group.size, ranks, "joined a group of the wrong size")

            try withError {
                try body(group)
            }
        } catch {
            let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }

    // MARK: - Process helpers

    /// The `xctest` tool running this bundle, plus the bundle itself.
    private static func testRunner() -> (executable: URL, bundle: URL)? {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0])
        guard executable.lastPathComponent == "xctest" else { return nil }

        let bundle = Bundle(for: DistributedRingTests.self).bundleURL
        guard bundle.pathExtension == "xctest" else { return nil }

        return (executable, bundle)
    }

    /// Write the JSON hostfile the ring backend expects.
    private static func writeHostfile(ports: [Int]) throws -> URL {
        let hosts = ports.map { ["127.0.0.1:\($0)"] }
        let data = try JSONSerialization.data(withJSONObject: hosts)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-ring-hostfile-\(UUID().uuidString).json")
        try data.write(to: url)

        return url
    }

    /// Prepare the test bundle to be re-executed as the given rank.
    ///
    /// The environment is inherited without XCTest's own session variables.
    /// Under `xcodebuild test` those make the spawned `xctest` wait for Xcode
    /// instead of running the test named on its command line.
    private static func makeRank(
        _ rank: Int, runner: (executable: URL, bundle: URL), testName: String, hostfile: URL
    ) -> Process {
        let process = Process()
        process.executableURL = runner.executable
        process.arguments = ["-XCTest", testName, runner.bundle.path]

        var environment = ProcessInfo.processInfo.environment.filter { key, value in
            !key.hasPrefix("XCTest") && !key.hasPrefix("XCInject")
                && !(key == "DYLD_INSERT_LIBRARIES" && value.contains("XCTest"))
        }
        environment["MLX_RANK"] = "\(rank)"
        environment["MLX_HOSTFILE"] = hostfile.path
        environment[spawnedRankVariable] = "1"
        process.environment = environment

        return process
    }
}

/// Tracks whether this process has already formed a group.
private final class HarnessState: @unchecked Sendable {

    private let lock = NSLock()
    private var used = false

    /// Returns `true` the first time it is called in a process.
    func claim() -> Bool {
        lock.withLock {
            if used {
                return false
            }
            used = true
            return true
        }
    }
}

private let state = HarnessState()

/// Ends the test process when a spawned rank fails or the ranks take too long.
///
/// Forming the ring blocks in `accept()` until every rank has connected, and
/// MLX has no timeout, so a rank that dies on startup would leave rank 0
/// waiting forever.  Nothing can interrupt that thread, so the watchdog
/// terminates the spawned ranks and exits: a failure rather than a hang.
private final class Watchdog: @unchecked Sendable {

    private let lock = NSLock()
    private var armed = true
    private let spawned: [Process]

    init(spawned: [Process], timeout: TimeInterval) {
        self.spawned = spawned

        for (index, process) in spawned.enumerated() {
            process.terminationHandler = { [weak self] terminated in
                let status = terminated.terminationStatus
                guard status != 0 else { return }

                // rank 0 may still report the failure itself, e.g. as an error
                // from a collective, so give it a moment first
                DispatchQueue.global().asyncAfter(deadline: .now() + 10) {
                    self?.fire("rank \(index + 1) exited with status \(status)")
                }
            }
        }

        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { [weak self] in
            self?.fire("the ranks did not finish within \(Int(timeout)) seconds")
        }
    }

    func disarm() {
        lock.withLock { armed = false }
    }

    private func fire(_ reason: String) {
        guard lock.withLock({ armed }) else { return }

        for process in spawned where process.isRunning {
            process.terminate()
        }
        FileHandle.standardError.write(Data("DistributedHarness: \(reason), exiting\n".utf8))
        exit(1)
    }
}

/// Reserve loopback ports by binding them and closing them again.
///
/// Every socket stays open until all ports are picked, so the same port is
/// not handed out twice.
///
/// This lives at file scope on purpose: inside an `XCTestCase` the name
/// `bind` resolves to `NSObject.bind(_:to:withKeyPath:options:)` rather than
/// the socket call.
private func reserveFreePorts(count: Int) -> [Int]? {
    var ports = [Int]()
    var sockets = [Int32]()
    defer { sockets.forEach { close($0) } }

    for _ in 0 ..< count {
        // Glibc types SOCK_STREAM as an enum rather than the Int32 that
        // socket() takes.
        #if canImport(Darwin)
            let socketType = SOCK_STREAM
        #else
            let socketType = Int32(SOCK_STREAM.rawValue)
        #endif

        let fd = socket(AF_INET, socketType, 0)
        guard fd >= 0 else { return nil }
        sockets.append(fd)

        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return nil }

        var assigned = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &assigned) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        guard named == 0 else { return nil }

        ports.append(Int(UInt16(bigEndian: assigned.sin_port)))
    }

    return ports
}
