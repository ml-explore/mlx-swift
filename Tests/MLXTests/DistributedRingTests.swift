// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// Multi process tests for the ring backend.
///
/// These start additional processes and open loopback sockets, so they are
/// skipped unless `MLX_TEST_DISTRIBUTED=1` is set:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 xcrun xctest -XCTest DistributedRingTests \
///     .../MLXTests.xctest
/// ```
///
/// The test process is itself rank 0.  It reserves a loopback port for every
/// rank, writes a temporary hostfile, re-executes the test bundle once for each
/// of the other ranks and then runs the same body.  There are three ranks so
/// that the left and right neighbors of a rank are different processes.
///
/// There is deliberately a single test method: MLX caches the group per
/// process, so a second method would find rank 0 reusing the first ring while
/// freshly spawned ranks built a new one.  A single method also guarantees
/// every rank issues the same operations in the same order, which the ring
/// backend requires.
class DistributedRingTests: XCTestCase {

    static let testName = "DistributedRingTests/testRingCollectives"
    static let rankCount = 3

    /// How long the spawned ranks may run before the test process gives up.
    static let timeout: TimeInterval = 120

    /// Set in the environment of the ranks this test spawns.
    static let spawnedRankVariable = "MLX_TEST_DISTRIBUTED_SPAWNED_RANK"

    func testRingCollectives() throws {
        let environment = ProcessInfo.processInfo.environment
        if environment[Self.spawnedRankVariable] == "1" {
            // spawned by rank 0 -- the hostfile and rank are already set
            try runRank()
            return
        }

        guard environment["MLX_TEST_DISTRIBUTED"] == "1" else {
            throw XCTSkip(
                "Set MLX_TEST_DISTRIBUTED=1 to run the multi process ring tests.")
        }

        guard let runner = Self.testRunner() else {
            throw XCTSkip("Could not locate the test bundle to re-execute as a rank.")
        }

        guard let ports = reserveFreePorts(count: Self.rankCount) else {
            throw XCTSkip("Could not reserve \(Self.rankCount) loopback ports.")
        }

        let hostfile = try Self.writeHostfile(ports: ports)
        defer { try? FileManager.default.removeItem(at: hostfile) }

        let spawned = (1 ..< Self.rankCount).map {
            Self.makeRank($0, runner: runner, hostfile: hostfile)
        }
        let watchdog = Watchdog(spawned: spawned, timeout: Self.timeout)
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

        try runRank()

        for (index, process) in spawned.enumerated() {
            process.waitUntilExit()
            XCTAssertEqual(
                process.terminationStatus, 0,
                "rank \(index + 1) failed -- see its output above")
        }
    }

    /// The body every rank runs.
    ///
    /// The body is wrapped in `withError` so that an MLX error becomes a test
    /// failure naming the rank.  Without an active scope the global handler
    /// calls `fatalError()` and the message never reaches XCTest.
    private func runRank() throws {
        do {
            let group = try MLXDistributed.initialize(backend: .ring, strict: true)
            try withError {
                try runRankBody(group)
            }
        } catch {
            let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }

    private func runRankBody(_ group: MLXDistributed.Group) throws {
        let rank = group.rank
        let size = group.size
        print("[rank \(rank)] joined a group of size \(size)")
        XCTAssertEqual(size, Self.rankCount)
        XCTAssertTrue(rank >= 0 && rank < size)

        try reductions(group)

        // every rank contributes base * (rank + 1)
        let base = MLXArray([1, 2, 3]).asType(.float32)
        let x = base * Float(rank + 1)

        // allGather concatenates along the first axis in rank order
        assertEqual(
            MLXDistributed.allGather(x, group: group),
            concatenated((0 ..< size).map { base * Float($0 + 1) }))

        // send to both neighbors, one pair at a time so that every rank issues
        // the same operations in the same order.  The ring backend connects
        // only neighbors, and with three ranks the left and right ones differ.
        for sender in 0 ..< size {
            for receiver in [(sender + 1) % size, (sender + size - 1) % size] {
                if rank == sender {
                    try checkedEval(MLXDistributed.send(x, to: receiver, group: group))
                } else if rank == receiver {
                    let received = MLXDistributed.recvLike(x, from: sender, group: group)
                    try checkedEval(received)
                    assertEqual(received, base * Float(sender + 1))
                }
            }
        }
    }

    /// allSum, allMax and allMin across types and sizes, as in the Python
    /// distributed tests.  The large sizes exercise the ring's chunked
    /// transfers, which a handful of elements never reach.
    private func reductions(_ group: MLXDistributed.Group) throws {
        let rank = group.rank
        let size = group.size

        let dtypes: [(DType, Float)] = [
            (.int8, 0), (.uint8, 0), (.int32, 0), (.uint32, 0),
            (.float32, 1e-6), (.float16, 5e-3), (.bfloat16, 1e-1),
        ]
        let shapes = [[7], [10], [1024], [1024, 1024]]
        let key = MLXRandom.key(0)

        for (dtype, rtol) in dtypes {
            for shape in shapes {
                // every rank generates the same array and contributes its row
                let x = (MLXRandom.uniform(0 ..< 1, [size] + shape, key: key) * 10)
                    .asType(dtype)
                let name = "\(dtype) \(shape)"

                let sum = MLXDistributed.allSum(x[rank], group: group)
                let expected = x.sum(axis: 0)
                var error = abs(sum - expected)
                if rtol > 0 {
                    error = error / abs(expected)
                }
                try checkedEval(error)
                XCTAssertLessThanOrEqual(
                    error.max().asType(.float32).item(Float.self), rtol, "allSum \(name)")

                let maximum = MLXDistributed.allMax(x[rank], group: group)
                try checkedEval(maximum)
                XCTAssertTrue(
                    (maximum .== x.max(axis: 0)).all().item(Bool.self), "allMax \(name)")

                let minimum = MLXDistributed.allMin(x[rank], group: group)
                try checkedEval(minimum)
                XCTAssertTrue(
                    (minimum .== x.min(axis: 0)).all().item(Bool.self), "allMin \(name)")
            }
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
        _ rank: Int, runner: (executable: URL, bundle: URL), hostfile: URL
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
        FileHandle.standardError.write(Data("DistributedRingTests: \(reason), exiting\n".utf8))
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
