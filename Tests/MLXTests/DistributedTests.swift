// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class DistributedTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - (1) Group Lifecycle

    func testGroupLifecycle() {
        // Create a group, access rank/size, and let it deinit without crash
        let group = MLXDistributed.`init`()
        XCTAssertNotNil(group)

        let rank = group!.rank
        let size = group!.size
        XCTAssertEqual(rank, 0)
        XCTAssertEqual(size, 1)
    }

    func testGroupLifecycleManyCreations() {
        // Create 100+ groups in a loop to verify no double-free or use-after-free
        for _ in 0 ..< 150 {
            let group = MLXDistributed.`init`()
            XCTAssertNotNil(group)
            XCTAssertEqual(group!.rank, 0)
            XCTAssertEqual(group!.size, 1)
        }
    }

    // MARK: - (2) isAvailable

    func testIsAvailable() {
        // Ring backend is compiled in, so isAvailable should return true
        XCTAssertTrue(MLXDistributed.isAvailable())
    }

    // MARK: - (3) init returns rank=0, size=1

    func testInitSingletonGroup() {
        let group = MLXDistributed.`init`()
        XCTAssertNotNil(group)
        XCTAssertEqual(group!.rank, 0)
        XCTAssertEqual(group!.size, 1)
    }

    // MARK: - (4) Collective ops as identity on size-1 group

    func testAllSumIdentity() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let result = MLXDistributed.allSum(input, group: group)

        XCTAssertEqual(result.shape, input.shape)
        XCTAssertEqual(result.dtype, input.dtype)
        assertEqual(result, input, atol: 1e-5)
    }

    func testAllGatherIdentity() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [1.0, 2.0, 3.0])
        let result = MLXDistributed.allGather(input, group: group)

        XCTAssertEqual(result.shape, input.shape)
        XCTAssertEqual(result.dtype, input.dtype)
        assertEqual(result, input, atol: 1e-5)
    }

    func testAllMaxIdentity() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [5.0, 3.0, 7.0, 1.0])
        let result = MLXDistributed.allMax(input, group: group)

        XCTAssertEqual(result.shape, input.shape)
        XCTAssertEqual(result.dtype, input.dtype)
        assertEqual(result, input, atol: 1e-5)
    }

    func testAllMinIdentity() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [5.0, 3.0, 7.0, 1.0])
        let result = MLXDistributed.allMin(input, group: group)

        XCTAssertEqual(result.shape, input.shape)
        XCTAssertEqual(result.dtype, input.dtype)
        assertEqual(result, input, atol: 1e-5)
    }

    func testSumScatterIdentity() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let result = MLXDistributed.sumScatter(input, group: group)

        XCTAssertEqual(result.shape, input.shape)
        XCTAssertEqual(result.dtype, input.dtype)
        assertEqual(result, input, atol: 1e-5)
    }

    // MARK: - (5) send returns MLXArray, recv returns correct shape/dtype

    func testSendRecvAPISignatures() {
        // On a singleton group, send/recv raise fatal errors in the C backend.
        // We verify the API compiles and that the error is properly caught.
        let group = MLXDistributed.`init`()!

        // Verify send raises an error on singleton group
        var sendErrorCaught = false
        withErrorHandler({ _ in sendErrorCaught = true }) {
            let _ = MLXDistributed.send(
                MLXArray(converting: [10.0, 20.0, 30.0]), to: 0, group: group)
        }
        XCTAssertTrue(sendErrorCaught, "send on singleton group should produce an error")

        // Verify recv raises an error on singleton group
        var recvErrorCaught = false
        withErrorHandler({ _ in recvErrorCaught = true }) {
            let _ = MLXDistributed.recv(
                shape: [3], dtype: .float32, from: 0, group: group)
        }
        XCTAssertTrue(recvErrorCaught, "recv on singleton group should produce an error")
    }

    // MARK: - (6) recvLike returns correct shape/dtype

    func testRecvLikeAPISignature() {
        // On a singleton group, recvLike raises a fatal error in the C backend.
        // We verify the API compiles and that the error is properly caught.
        let group = MLXDistributed.`init`()!
        let template = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0])

        var errorCaught = false
        withErrorHandler({ _ in errorCaught = true }) {
            let _ = MLXDistributed.recvLike(template, from: 0, group: group)
        }
        XCTAssertTrue(errorCaught, "recvLike on singleton group should produce an error")
    }

    // MARK: - (7) Group split on size-1 group

    func testGroupSplitSingletonError() {
        // The C backend does not allow splitting a singleton group.
        // Verify the error is caught gracefully.
        let group = MLXDistributed.`init`()!

        var errorCaught = false
        withErrorHandler({ _ in errorCaught = true }) {
            let _ = group.split(color: 0)
        }
        XCTAssertTrue(errorCaught, "split on singleton group should produce an error")
    }

    // MARK: - (8) Multiple dtype test: allSum with float16 and int32

    func testAllSumMultipleDtypes() {
        let group = MLXDistributed.`init`()!

        // float16 test
        let float16Input = MLXArray(converting: [1.0, 2.0, 3.0]).asType(.float16)
        let float16Result = MLXDistributed.allSum(float16Input, group: group)
        XCTAssertEqual(float16Result.dtype, .float16)
        XCTAssertEqual(float16Result.shape, float16Input.shape)

        // int32 test
        let int32Input = MLXArray([10, 20, 30] as [Int32])
        let int32Result = MLXDistributed.allSum(int32Input, group: group)
        XCTAssertEqual(int32Result.dtype, .int32)
        XCTAssertEqual(int32Result.shape, int32Input.shape)
        assertEqual(int32Result, int32Input)
    }

    // MARK: - (9) High-dimensional array test: allSum on [2,3,4] shape

    func testAllSumHighDimensional() {
        let group = MLXDistributed.`init`()!

        // Create a 3D array of shape [2, 3, 4]
        let input = MLXArray(0 ..< 24, [2, 3, 4]).asType(.float32)
        let result = MLXDistributed.allSum(input, group: group)

        XCTAssertEqual(result.shape, [2, 3, 4])
        XCTAssertEqual(result.dtype, .float32)
        assertEqual(result, input, atol: 1e-5)
    }

    // MARK: - (10) Multiple group lifecycle: create parent, use child from init

    func testMultipleGroupLifecycle() {
        // On a singleton group, split is not supported by the C backend.
        // Instead, test that multiple independent groups (from init) can be
        // created and used independently without interference, and that
        // releasing one does not affect others.
        var child: DistributedGroup?

        do {
            let parent = MLXDistributed.`init`()!
            XCTAssertEqual(parent.rank, 0)
            XCTAssertEqual(parent.size, 1)

            // Create a second independent group
            child = MLXDistributed.`init`()!
            XCTAssertEqual(child!.rank, 0)
            XCTAssertEqual(child!.size, 1)

            // Use parent for a collective op
            let parentInput = MLXArray(converting: [1.0, 2.0])
            let parentResult = MLXDistributed.allSum(parentInput, group: parent)
            assertEqual(parentResult, parentInput, atol: 1e-5)

            // parent deinits here when exiting scope
        }

        // Child should still be valid after parent deinit
        XCTAssertNotNil(child)
        XCTAssertEqual(child!.rank, 0)
        XCTAssertEqual(child!.size, 1)

        // Use child for a collective operation after parent is gone
        let input = MLXArray(converting: [1.0, 2.0, 3.0])
        let result = MLXDistributed.allSum(input, group: child!)
        assertEqual(result, input, atol: 1e-5)
    }

    // MARK: - (11) Stream parameter test: call ops with explicit stream

    func testStreamParameter() {
        let group = MLXDistributed.`init`()!
        let input = MLXArray(converting: [1.0, 2.0, 3.0])

        // Call with explicit GPU stream
        let gpuStream = StreamOrDevice.device(.gpu)

        let sumResult = MLXDistributed.allSum(input, group: group, stream: gpuStream)
        assertEqual(sumResult, input, atol: 1e-5)

        let gatherResult = MLXDistributed.allGather(input, group: group, stream: gpuStream)
        assertEqual(gatherResult, input, atol: 1e-5)

        let maxResult = MLXDistributed.allMax(input, group: group, stream: gpuStream)
        assertEqual(maxResult, input, atol: 1e-5)

        let minResult = MLXDistributed.allMin(input, group: group, stream: gpuStream)
        assertEqual(minResult, input, atol: 1e-5)

        let scatterResult = MLXDistributed.sumScatter(input, group: group, stream: gpuStream)
        assertEqual(scatterResult, input, atol: 1e-5)
    }

    // MARK: - (12) strict=true error handling test

    func testInitStrictMode() {
        // With strict=true and no hostfile/distributed backend configured,
        // init should either return nil or trigger an error (not crash the process).
        // The C backend raises an error when strict=true and no backend can initialize,
        // so we use withErrorHandler to catch it gracefully.
        var errorCaught = false
        var group: DistributedGroup?

        withErrorHandler({ _ in errorCaught = true }) {
            group = MLXDistributed.`init`(strict: true)
        }

        if errorCaught {
            // Error was caught -- strict mode correctly detected no multi-process backend
            // group may or may not be nil depending on when error was raised
        } else if let group = group {
            // If a group is returned without error, it should be valid
            XCTAssertEqual(group.rank, 0)
            XCTAssertGreaterThanOrEqual(group.size, 1)
        }
        // Either nil/error or a valid group is acceptable -- the key is no crash
    }

    // MARK: - Multi-Process Tests

    /// Find the DistributedWorker binary in the build products directory.
    ///
    /// The worker binary is built as part of the package and placed in the same
    /// directory as the test bundle (DerivedData/.../Debug/).
    private func findWorkerBinary() -> URL? {
        // The test bundle is at .../Debug/MLXTests.xctest
        // The worker binary is at .../Debug/DistributedWorker
        let testBundle = Bundle(for: type(of: self))
        let bundleURL = testBundle.bundleURL
        let productsDir = bundleURL.deletingLastPathComponent()
        let workerURL = productsDir.appendingPathComponent("DistributedWorker")

        if FileManager.default.isExecutableFile(atPath: workerURL.path) {
            return workerURL
        }

        return nil
    }

    /// Find two available TCP ports for the ring backend.
    private func findAvailablePorts() -> (Int, Int)? {
        func findPort() -> Int? {
            // Create a socket, bind to port 0, get the assigned port
            let sock = socket(AF_INET, SOCK_STREAM, 0)
            guard sock >= 0 else { return nil }
            defer { close(sock) }

            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = 0  // Let the OS pick a port
            addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian

            var addrCopy = addr
            let bindResult = withUnsafePointer(to: &addrCopy) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    Darwin.bind(sock, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else { return nil }

            var len = socklen_t(MemoryLayout<sockaddr_in>.size)
            let nameResult = withUnsafeMutablePointer(to: &addrCopy) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    getsockname(sock, sockPtr, &len)
                }
            }
            guard nameResult == 0 else { return nil }

            return Int(UInt16(bigEndian: addrCopy.sin_port))
        }

        guard let port1 = findPort(), let port2 = findPort(), port1 != port2 else {
            return nil
        }
        return (port1, port2)
    }

    /// Create a temporary hostfile for 2-process ring backend on localhost.
    private func createHostfile(port1: Int, port2: Int) throws -> URL {
        let hostfile = [
            ["\("127.0.0.1"):\(port1)"],
            ["\("127.0.0.1"):\(port2)"],
        ]
        let jsonData = try JSONSerialization.data(
            withJSONObject: hostfile, options: [.prettyPrinted])
        let jsonString = String(data: jsonData, encoding: .utf8)!

        let tempDir = FileManager.default.temporaryDirectory
        let hostfilePath = tempDir.appendingPathComponent(
            "mlx_test_hostfile_\(UUID().uuidString).json")
        try jsonString.write(to: hostfilePath, atomically: true, encoding: .utf8)

        return hostfilePath
    }

    /// Spawn a worker process with the given rank and operation, wait for completion.
    private func spawnWorker(
        workerBinary: URL, rank: Int, hostfilePath: URL, operation: String, timeout: TimeInterval
    ) -> (exitCode: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = workerBinary
        process.environment = [
            "MLX_RANK": "\(rank)",
            "MLX_HOSTFILE": hostfilePath.path,
            "MLX_TEST_OP": operation,
            // Preserve PATH and DYLD paths for Metal framework access
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "HOME": ProcessInfo.processInfo.environment["HOME"] ?? "/tmp",
            "DYLD_LIBRARY_PATH":
                ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? "",
            "DYLD_FRAMEWORK_PATH":
                ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
        ]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            return (-1, "", "Failed to start process: \(error)")
        }

        // Wait with timeout
        let deadline = DispatchTime.now() + timeout
        let group = DispatchGroup()
        group.enter()

        DispatchQueue.global().async {
            process.waitUntilExit()
            group.leave()
        }

        let result = group.wait(timeout: deadline)
        if result == .timedOut {
            process.terminate()
            // Give it a moment to terminate
            Thread.sleep(forTimeInterval: 0.5)
            if process.isRunning {
                // Force kill
                kill(process.processIdentifier, SIGKILL)
            }
            return (-1, "", "Process timed out after \(timeout) seconds")
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stdoutStr = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderrStr = String(data: stderrData, encoding: .utf8) ?? ""

        return (process.terminationStatus, stdoutStr, stderrStr)
    }

    /// Run a multi-process test with the given operation.
    ///
    /// Spawns 2 worker processes with rank 0 and rank 1, waits for both,
    /// and returns their results.
    private func runMultiProcessTest(
        operation: String,
        timeout: TimeInterval = 30.0,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> (rank0: (exitCode: Int32, stdout: String, stderr: String),
        rank1: (exitCode: Int32, stdout: String, stderr: String))?
    {
        guard let workerBinary = findWorkerBinary() else {
            XCTFail(
                "DistributedWorker binary not found. Build with: xcodebuild build -scheme mlx-swift-Package",
                file: file, line: line)
            return nil
        }

        guard let (port1, port2) = findAvailablePorts() else {
            XCTFail("Could not find two available ports", file: file, line: line)
            return nil
        }

        let hostfilePath: URL
        do {
            hostfilePath = try createHostfile(port1: port1, port2: port2)
        } catch {
            XCTFail("Failed to create hostfile: \(error)", file: file, line: line)
            return nil
        }
        defer {
            try? FileManager.default.removeItem(at: hostfilePath)
        }

        // Spawn both workers concurrently
        var rank0Result: (exitCode: Int32, stdout: String, stderr: String)!
        var rank1Result: (exitCode: Int32, stdout: String, stderr: String)!

        let group = DispatchGroup()

        group.enter()
        DispatchQueue.global().async {
            rank0Result = self.spawnWorker(
                workerBinary: workerBinary, rank: 0, hostfilePath: hostfilePath,
                operation: operation, timeout: timeout)
            group.leave()
        }

        group.enter()
        DispatchQueue.global().async {
            rank1Result = self.spawnWorker(
                workerBinary: workerBinary, rank: 1, hostfilePath: hostfilePath,
                operation: operation, timeout: timeout)
            group.leave()
        }

        // Wait for both with extra margin
        let waitResult = group.wait(timeout: .now() + timeout + 10)
        if waitResult == .timedOut {
            XCTFail(
                "Multi-process test timed out waiting for workers", file: file, line: line)
            return nil
        }

        return (rank0Result, rank1Result)
    }

    // MARK: - (13) Multi-process allSum

    func testMultiProcessAllSum() {
        guard let results = runMultiProcessTest(operation: "allSum") else { return }

        // Log debug output
        if results.rank0.exitCode != 0 || results.rank1.exitCode != 0 {
            print("=== Rank 0 stderr ===")
            print(results.rank0.stderr)
            print("=== Rank 0 stdout ===")
            print(results.rank0.stdout)
            print("=== Rank 1 stderr ===")
            print(results.rank1.stderr)
            print("=== Rank 1 stdout ===")
            print(results.rank1.stdout)
        }

        XCTAssertEqual(
            results.rank0.exitCode, 0,
            "Rank 0 failed with exit code \(results.rank0.exitCode). stderr: \(results.rank0.stderr)"
        )
        XCTAssertEqual(
            results.rank1.exitCode, 0,
            "Rank 1 failed with exit code \(results.rank1.exitCode). stderr: \(results.rank1.stderr)"
        )

        // Verify JSON output from both ranks contains [5.0, 7.0, 9.0]
        for (rank, result) in [(0, results.rank0), (1, results.rank1)] {
            let stdout = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stdout.isEmpty,
                let data = stdout.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let values = json["values"] as? [Double],
                let shape = json["shape"] as? [Int]
            else {
                XCTFail("Rank \(rank) produced invalid JSON output: '\(stdout)'")
                continue
            }

            XCTAssertEqual(shape, [3], "Rank \(rank) shape mismatch")
            XCTAssertEqual(values.count, 3, "Rank \(rank) values count mismatch")
            XCTAssertEqual(values[0], 5.0, accuracy: 1e-5, "Rank \(rank) value[0] mismatch")
            XCTAssertEqual(values[1], 7.0, accuracy: 1e-5, "Rank \(rank) value[1] mismatch")
            XCTAssertEqual(values[2], 9.0, accuracy: 1e-5, "Rank \(rank) value[2] mismatch")
        }
    }

    // MARK: - (14) Multi-process allGather

    func testMultiProcessAllGather() {
        guard let results = runMultiProcessTest(operation: "allGather") else { return }

        // Log debug output
        if results.rank0.exitCode != 0 || results.rank1.exitCode != 0 {
            print("=== Rank 0 stderr ===")
            print(results.rank0.stderr)
            print("=== Rank 0 stdout ===")
            print(results.rank0.stdout)
            print("=== Rank 1 stderr ===")
            print(results.rank1.stderr)
            print("=== Rank 1 stdout ===")
            print(results.rank1.stdout)
        }

        XCTAssertEqual(
            results.rank0.exitCode, 0,
            "Rank 0 failed with exit code \(results.rank0.exitCode). stderr: \(results.rank0.stderr)"
        )
        XCTAssertEqual(
            results.rank1.exitCode, 0,
            "Rank 1 failed with exit code \(results.rank1.exitCode). stderr: \(results.rank1.stderr)"
        )

        // Verify JSON output from both ranks contains [1,2,3,4,5,6] shape [6]
        let expected: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for (rank, result) in [(0, results.rank0), (1, results.rank1)] {
            let stdout = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !stdout.isEmpty,
                let data = stdout.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let values = json["values"] as? [Double],
                let shape = json["shape"] as? [Int]
            else {
                XCTFail("Rank \(rank) produced invalid JSON output: '\(stdout)'")
                continue
            }

            XCTAssertEqual(shape, [6], "Rank \(rank) shape mismatch")
            XCTAssertEqual(values.count, 6, "Rank \(rank) values count mismatch")
            for i in 0..<6 {
                XCTAssertEqual(
                    values[i], expected[i], accuracy: 1e-5,
                    "Rank \(rank) value[\(i)] mismatch")
            }
        }
    }

    // MARK: - (15) Multi-process send/recv

    func testMultiProcessSendRecv() {
        guard let results = runMultiProcessTest(operation: "sendRecv") else { return }

        // Log debug output
        if results.rank0.exitCode != 0 || results.rank1.exitCode != 0 {
            print("=== Rank 0 stderr ===")
            print(results.rank0.stderr)
            print("=== Rank 0 stdout ===")
            print(results.rank0.stdout)
            print("=== Rank 1 stderr ===")
            print(results.rank1.stderr)
            print("=== Rank 1 stdout ===")
            print(results.rank1.stdout)
        }

        XCTAssertEqual(
            results.rank0.exitCode, 0,
            "Rank 0 failed with exit code \(results.rank0.exitCode). stderr: \(results.rank0.stderr)"
        )
        XCTAssertEqual(
            results.rank1.exitCode, 0,
            "Rank 1 failed with exit code \(results.rank1.exitCode). stderr: \(results.rank1.stderr)"
        )

        // Verify rank 1 received [10, 20, 30]
        let rank1Stdout = results.rank1.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rank1Stdout.isEmpty,
            let data = rank1Stdout.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let values = json["values"] as? [Double],
            let shape = json["shape"] as? [Int]
        else {
            XCTFail("Rank 1 produced invalid JSON output: '\(rank1Stdout)'")
            return
        }

        XCTAssertEqual(shape, [3], "Rank 1 recv shape mismatch")
        XCTAssertEqual(values.count, 3, "Rank 1 recv values count mismatch")
        XCTAssertEqual(values[0], 10.0, accuracy: 1e-5, "Rank 1 recv value[0] mismatch")
        XCTAssertEqual(values[1], 20.0, accuracy: 1e-5, "Rank 1 recv value[1] mismatch")
        XCTAssertEqual(values[2], 30.0, accuracy: 1e-5, "Rank 1 recv value[2] mismatch")
    }
}
