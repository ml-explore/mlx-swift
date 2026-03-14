// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// A helper executable for multi-process distributed tests.
///
/// This program is spawned by `DistributedTests` with environment variables:
/// - `MLX_RANK`: the rank of this process (0 or 1)
/// - `MLX_HOSTFILE`: path to the JSON hostfile for the ring backend
/// - `MLX_TEST_OP`: which operation to test ("allSum", "allGather", "sendRecv")
///
/// The program performs the distributed operation and prints results as JSON
/// to stdout. Exit code 0 means success, non-zero means failure.
@main
struct DistributedWorker {
    static func main() {
        guard let rankStr = ProcessInfo.processInfo.environment["MLX_RANK"],
            let rank = Int(rankStr)
        else {
            fputs("ERROR: MLX_RANK not set\n", stderr)
            exit(1)
        }

        guard ProcessInfo.processInfo.environment["MLX_HOSTFILE"] != nil else {
            fputs("ERROR: MLX_HOSTFILE not set\n", stderr)
            exit(1)
        }

        guard let testOp = ProcessInfo.processInfo.environment["MLX_TEST_OP"] else {
            fputs("ERROR: MLX_TEST_OP not set\n", stderr)
            exit(1)
        }

        fputs("Worker rank=\(rank) starting operation=\(testOp)\n", stderr)

        // Distributed operations only have CPU implementations, so use CPU device
        MLX.Device.setDefault(device: .cpu)

        // Initialize distributed with strict=true (ring backend must be available)
        guard let group = MLXDistributed.`init`(strict: true) else {
            fputs("ERROR: Failed to initialize distributed group (strict=true)\n", stderr)
            exit(1)
        }

        fputs(
            "Worker rank=\(rank) initialized: group.rank=\(group.rank) group.size=\(group.size)\n",
            stderr)

        guard group.rank == rank else {
            fputs("ERROR: group.rank (\(group.rank)) != expected rank (\(rank))\n", stderr)
            exit(1)
        }

        guard group.size == 2 else {
            fputs("ERROR: group.size (\(group.size)) != 2\n", stderr)
            exit(1)
        }

        switch testOp {
        case "allSum":
            runAllSum(rank: rank, group: group)
        case "allGather":
            runAllGather(rank: rank, group: group)
        case "sendRecv":
            runSendRecv(rank: rank, group: group)
        default:
            fputs("ERROR: Unknown test operation: \(testOp)\n", stderr)
            exit(1)
        }

        fputs("Worker rank=\(rank) completed successfully\n", stderr)
        exit(0)
    }

    /// allSum test: rank 0 has [1,2,3], rank 1 has [4,5,6], both should get [5,7,9]
    static func runAllSum(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 2.0, 3.0])
        } else {
            input = MLXArray(converting: [4.0, 5.0, 6.0])
        }

        let result = MLXDistributed.allSum(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        // Output result as JSON to stdout
        print(
            "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        // Verify locally
        let expected: [Float] = [5.0, 7.0, 9.0]
        for i in 0 ..< 3 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: allSum mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// allGather test: rank 0 has [1,2,3], rank 1 has [4,5,6], both should get [1,2,3,4,5,6]
    static func runAllGather(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 2.0, 3.0])
        } else {
            input = MLXArray(converting: [4.0, 5.0, 6.0])
        }

        let result = MLXDistributed.allGather(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        // Output result as JSON to stdout
        print(
            "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        // Verify locally
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        guard shape == [6] else {
            fputs("ERROR: allGather shape mismatch: got \(shape), expected [6]\n", stderr)
            exit(1)
        }
        for i in 0 ..< 6 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: allGather mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// send/recv test: rank 0 sends [10,20,30], rank 1 receives and verifies
    static func runSendRecv(rank: Int, group: DistributedGroup) {
        if rank == 0 {
            let data = MLXArray(converting: [10.0, 20.0, 30.0])
            let token = MLXDistributed.send(data, to: 1, group: group)
            eval(token)

            // Output success to stdout
            print("{\"sent\": [10.0,20.0,30.0]}")
        } else {
            let received = MLXDistributed.recv(
                shape: [3], dtype: .float32, from: 0, group: group)
            eval(received)

            let values = received.asArray(Float.self)
            let shape = received.shape

            // Output result as JSON to stdout
            print(
                "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
            )

            // Verify locally
            let expected: [Float] = [10.0, 20.0, 30.0]
            guard shape == [3] else {
                fputs("ERROR: recv shape mismatch: got \(shape), expected [3]\n", stderr)
                exit(1)
            }
            for i in 0 ..< 3 {
                if abs(values[i] - expected[i]) > 1e-5 {
                    fputs(
                        "ERROR: recv mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                        stderr)
                    exit(1)
                }
            }
        }
    }
}
