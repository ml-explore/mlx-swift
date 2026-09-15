// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

#if canImport(Darwin)
    import Darwin
#endif

/// Multi process tests for the ring backend.
///
/// These are skipped unless `MLX_TEST_DISTRIBUTED=1` is set, matching how the
/// Python distributed tests are excluded from CI:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 swift test --filter DistributedRingTests
/// ```
///
/// The test process is itself a rank.  The first process picks two free ports
/// on the loopback interface, writes a temporary hostfile, re-executes the
/// test bundle as rank 1 and then runs as rank 0.  Both ranks execute the
/// same body, so no separate worker executable is needed.
///
/// There is deliberately a single test method: MLX caches the group per
/// process, so a second method would find the parent reusing the first ring
/// while a freshly spawned child built a new one.  A single method also
/// guarantees both ranks issue the same collectives in the same order, which
/// the ring backend requires.
class DistributedRingTests: XCTestCase {

    static let testName = "DistributedRingTests/testRingCollectives"

    func testRingCollectives() throws {
        if ProcessInfo.processInfo.environment["MLX_RANK"] != nil {
            // spawned as a rank -- the hostfile and rank are already set
            try runRank()
            return
        }

        guard ProcessInfo.processInfo.environment["MLX_TEST_DISTRIBUTED"] == "1" else {
            throw XCTSkip(
                "Set MLX_TEST_DISTRIBUTED=1 to run the multi process ring tests.")
        }

        guard let runner = Self.testRunner() else {
            throw XCTSkip("Could not locate the test bundle to re-execute as a rank.")
        }

        guard let ports = reserveFreePorts(count: 2) else {
            throw XCTSkip("Could not reserve two loopback ports.")
        }

        let hostfile = try Self.writeHostfile(ports: ports)
        defer { try? FileManager.default.removeItem(at: hostfile) }

        let child = try Self.spawnRank(1, runner: runner, hostfile: hostfile)
        defer {
            if child.isRunning {
                child.terminate()
            }
        }

        setenv("MLX_HOSTFILE", hostfile.path, 1)
        setenv("MLX_RANK", "0", 1)
        defer {
            unsetenv("MLX_HOSTFILE")
            unsetenv("MLX_RANK")
        }

        try runRank()

        child.waitUntilExit()
        XCTAssertEqual(child.terminationStatus, 0, "rank 1 failed -- see its output above")
    }

    /// The body every rank runs.
    ///
    /// The body is wrapped in `withError` so that an MLX error becomes a test
    /// failure naming the rank.  Without an active scope the global handler
    /// calls `fatalError()` and the message never reaches XCTest.
    private func runRank() throws {
        do {
            // a failed init returns nil rather than a group whose rank and
            // size silently read as zero
            let group = try withError {
                MLXDistributed.initialize(backend: .ring, strict: true)
            }
            try withError {
                try runRankBody(XCTUnwrap(group))
            }
        } catch {
            let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }

    private func runRankBody(_ group: MLXDistributed.Group) throws {
        print("[rank \(group.rank)] joined a group of size \(group.size)")
        XCTAssertEqual(group.size, 2)
        XCTAssertTrue(group.rank == 0 || group.rank == 1)

        let rank = group.rank
        let other = 1 - rank

        // allSum: [1,2,3] * (rank + 1) summed is [3,6,9]
        let x = MLXArray([1, 2, 3]).asType(.float32) * Float(rank + 1)
        assertEqual(
            MLXDistributed.allSum(x, group: group),
            MLXArray([3, 6, 9]).asType(.float32))

        // allMax picks the rank 1 contribution, allMin the rank 0 one
        assertEqual(
            MLXDistributed.allMax(x, group: group),
            MLXArray([2, 4, 6]).asType(.float32))
        assertEqual(
            MLXDistributed.allMin(x, group: group),
            MLXArray([1, 2, 3]).asType(.float32))

        // allGather concatenates along the first axis
        let gathered = MLXDistributed.allGather(x, group: group)
        XCTAssertEqual(gathered.shape, [6])
        assertEqual(gathered, MLXArray([1, 2, 3, 2, 4, 6]).asType(.float32))

        // send/recv: rank 0 sends, rank 1 receives, then the reverse
        for sender in 0 ..< 2 {
            if rank == sender {
                let sent = MLXDistributed.send(x, to: other, group: group)
                try checkedEval(sent)
            } else {
                let received = MLXDistributed.recvLike(x, from: other, group: group)
                try checkedEval(received)
                assertEqual(
                    received,
                    MLXArray([1, 2, 3]).asType(.float32) * Float(sender + 1))
            }
        }

        try shardedLayers(group: group, rank: rank)
    }

    /// Tensor parallelism: a sharded layer must reproduce the result of the
    /// `Linear` it was built from.
    ///
    /// Every rank seeds identically so that the layer being sharded -- and the
    /// expected result -- are the same everywhere.
    private func shardedLayers(group: MLXDistributed.Group, rank: Int) throws {
        try XCTContext.runActivity(named: "sharded linear layers") { _ in
            MLXRandom.seed(42)
            let linear = Linear(4, 8)
            let x = MLXArray(0 ..< 12, [3, 4]).asType(.float32)

            let expected = linear(x)
            try checkedEval(expected)

            // all-to-sharded: each rank holds half of the output dimensions and
            // computes that slice of the result
            let allToSharded = AllToShardedLinear(linear, group: group)
            XCTAssertEqual(allToSharded.weight.shape, [4, 4])

            let partial = allToSharded(x)
            XCTAssertEqual(partial.shape, [3, 4])

            // allGather concatenates along the first axis, so gather the
            // transpose to reassemble the output columns in rank order
            let reassembled = MLXDistributed.allGather(partial.T, group: group).T
            try checkedEval(reassembled)
            assertEqual(reassembled, expected)

            // sharded-to-all: each rank holds half of the input dimensions and
            // is fed the matching slice; the layer sums the partial products
            let shardedToAll = ShardedToAllLinear(linear, group: group)
            XCTAssertEqual(shardedToAll.weight.shape, [8, 2])

            let combined = shardedToAll(x.split(parts: 2, axis: 1)[rank])
            try checkedEval(combined)
            assertEqual(combined, expected)
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

    /// Re-execute the test bundle as the given rank.
    private static func spawnRank(
        _ rank: Int, runner: (executable: URL, bundle: URL), hostfile: URL
    ) throws -> Process {
        let process = Process()
        process.executableURL = runner.executable
        process.arguments = ["-XCTest", testName, runner.bundle.path]

        var environment = ProcessInfo.processInfo.environment
        environment["MLX_RANK"] = "\(rank)"
        environment["MLX_HOSTFILE"] = hostfile.path
        environment["MLX_TEST_DISTRIBUTED"] = "1"
        process.environment = environment

        try process.run()

        return process
    }
}

/// Reserve loopback ports by binding and immediately closing them.
///
/// This lives at file scope on purpose: inside an `XCTestCase` the name
/// `bind` resolves to `NSObject.bind(_:to:withKeyPath:options:)` rather than
/// the socket call.
private func reserveFreePorts(count: Int) -> [Int]? {
    var ports = [Int]()

    for _ in 0 ..< count {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

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
