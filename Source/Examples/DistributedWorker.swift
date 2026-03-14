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
        case "split":
            runSplit(rank: rank, group: group)
        case "allMax":
            runAllMax(rank: rank, group: group)
        case "allMin":
            runAllMin(rank: rank, group: group)
        case "sumScatter":
            runSumScatter(rank: rank, group: group)
        case "recvLike":
            runRecvLike(rank: rank, group: group)
        case "sendRecvIterative":
            runSendRecvIterative(rank: rank, group: group)
        case "allSumMultiDtype":
            runAllSumMultiDtype(rank: rank, group: group)
        case "allSumMultiShape":
            runAllSumMultiShape(rank: rank, group: group)
        case "allGatherVjp":
            runAllGatherVjp(rank: rank, group: group)
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

    /// split test: exercises group.split(color:key:) across multiple processes.
    ///
    /// Currently, the ring backend (and all other MLX backends) do NOT support
    /// group split — they throw "[ring] Group split not supported." This test
    /// verifies that:
    /// 1. The split call is attempted and the error is detected (not a crash)
    /// 2. The parent group remains usable after the failed split
    /// 3. An allSum on the original parent group still works correctly
    ///
    /// When upstream adds split support, this test should be updated to verify
    /// the child group works independently after parent deinit.
    static func runSplit(rank: Int, group: DistributedGroup) {
        // Attempt to split — expect an error from the ring backend
        var splitErrorCaught = false
        withErrorHandler({ errMsg in
            fputs("Worker rank=\(rank) split error (expected): \(errMsg)\n", stderr)
            splitErrorCaught = true
        }) {
            let _ = group.split(color: 0, key: rank)
        }

        if !splitErrorCaught {
            // If split succeeds in the future (backend support added), this
            // path should be expanded to test child group functionality.
            fputs("Worker rank=\(rank) split unexpectedly succeeded\n", stderr)
        }

        // Verify the parent group is still usable after the failed split
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

        // Output result as JSON to stdout — include split error status
        print(
            "{\"splitErrorCaught\": \(splitErrorCaught), \"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        // Verify allSum locally
        let expected: [Float] = [5.0, 7.0, 9.0]
        for i in 0 ..< 3 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: split allSum mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// allMax test: rank 0 has [1,5,3], rank 1 has [4,2,6], both should get [4,5,6]
    static func runAllMax(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 5.0, 3.0])
        } else {
            input = MLXArray(converting: [4.0, 2.0, 6.0])
        }

        let result = MLXDistributed.allMax(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        print(
            "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        let expected: [Float] = [4.0, 5.0, 6.0]
        for i in 0 ..< 3 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: allMax mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// allMin test: rank 0 has [1,5,3], rank 1 has [4,2,6], both should get [1,2,3]
    static func runAllMin(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 5.0, 3.0])
        } else {
            input = MLXArray(converting: [4.0, 2.0, 6.0])
        }

        let result = MLXDistributed.allMin(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        print(
            "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        let expected: [Float] = [1.0, 2.0, 3.0]
        for i in 0 ..< 3 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: allMin mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// sumScatter test: rank 0 and rank 1 each have [1,2,3,4], result shape is halved,
    /// each rank gets its slice of the element-wise sum [2,4,6,8].
    ///
    /// NOTE: The ring backend currently does not implement ReduceScatter for
    /// multi-process groups. This test detects the error gracefully and reports
    /// the backend limitation rather than crashing.
    static func runSumScatter(rank: Int, group: DistributedGroup) {
        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])

        // Use withErrorHandler to catch the C++ backend error. When eval()
        // triggers an error, the handler is called. We must print the result
        // and exit immediately from within the handler because the C++ code
        // may continue executing undefined behavior after the handler returns.
        withErrorHandler({ errMsg in
            fputs("Worker rank=\(rank) sumScatter error (expected): \(errMsg)\n", stderr)
            print("{\"errorCaught\": true, \"errorMessage\": \"ReduceScatter not implemented\"}")
            exit(0)
        }) {
            let result = MLXDistributed.sumScatter(input, group: group)
            eval(result)

            let values = result.asArray(Float.self)
            let shape = result.shape

            print(
                "{\"errorCaught\": false, \"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
            )

            // The element-wise sum is [2,4,6,8], split in half:
            // rank 0 gets [2,4], rank 1 gets [6,8]
            guard shape == [2] else {
                fputs("ERROR: sumScatter shape mismatch: got \(shape), expected [2]\n", stderr)
                exit(1)
            }

            let expected: [Float] = rank == 0 ? [2.0, 4.0] : [6.0, 8.0]
            for i in 0 ..< 2 {
                if abs(values[i] - expected[i]) > 1e-5 {
                    fputs(
                        "ERROR: sumScatter mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                        stderr)
                    exit(1)
                }
            }
        }
    }

    /// recvLike test: rank 0 sends [42.0, 43.0, 44.0], rank 1 receives via recvLike
    /// using a template array and verifies shape/dtype/values match
    static func runRecvLike(rank: Int, group: DistributedGroup) {
        if rank == 0 {
            let data = MLXArray(converting: [42.0, 43.0, 44.0])
            let token = MLXDistributed.send(data, to: 1, group: group)
            eval(token)

            print("{\"sent\": [42.0,43.0,44.0]}")
        } else {
            let template = MLXArray(converting: [0.0, 0.0, 0.0])
            let received = MLXDistributed.recvLike(template, from: 0, group: group)
            eval(received)

            let values = received.asArray(Float.self)
            let shape = received.shape
            let dtype = received.dtype

            print(
                "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))], \"dtype\": \"\(dtype)\"}"
            )

            guard shape == [3] else {
                fputs("ERROR: recvLike shape mismatch: got \(shape), expected [3]\n", stderr)
                exit(1)
            }
            guard dtype == .float32 else {
                fputs("ERROR: recvLike dtype mismatch: got \(dtype), expected float32\n", stderr)
                exit(1)
            }

            let expected: [Float] = [42.0, 43.0, 44.0]
            for i in 0 ..< 3 {
                if abs(values[i] - expected[i]) > 1e-5 {
                    fputs(
                        "ERROR: recvLike mismatch at index \(i): got \(values[i]), expected \(expected[i])\n",
                        stderr)
                    exit(1)
                }
            }
        }
    }

    /// Iterative send/recv test: 10 rounds of alternating send/recv with doubling values.
    /// rank 0 starts with 1, sends to rank 1, rank 1 doubles and sends back, etc.
    static func runSendRecvIterative(rank: Int, group: DistributedGroup) {
        let rounds = 10
        var value: Double = 1.0

        for round in 0 ..< rounds {
            if rank == 0 {
                // Rank 0 sends on even rounds, receives on odd rounds
                if round % 2 == 0 {
                    let data = MLXArray(converting: [value])
                    let token = MLXDistributed.send(data, to: 1, group: group)
                    eval(token)
                } else {
                    let received = MLXDistributed.recv(
                        shape: [1], dtype: .float32, from: 1, group: group)
                    eval(received)
                    value = Double(received.asArray(Float.self)[0])
                }
            } else {
                // Rank 1 receives on even rounds, doubles and sends on odd rounds
                if round % 2 == 0 {
                    let received = MLXDistributed.recv(
                        shape: [1], dtype: .float32, from: 0, group: group)
                    eval(received)
                    value = Double(received.asArray(Float.self)[0])
                    value *= 2.0
                } else {
                    let data = MLXArray(converting: [value])
                    let token = MLXDistributed.send(data, to: 0, group: group)
                    eval(token)
                }
            }
        }

        // After 10 rounds (5 complete send-receive cycles):
        // Round 0: rank 0 sends 1 -> rank 1 receives 1, doubles to 2
        // Round 1: rank 1 sends 2 -> rank 0 receives 2
        // Round 2: rank 0 sends 2 -> rank 1 receives 2, doubles to 4
        // ...
        // Round 9: rank 1 sends 32 -> rank 0 receives 32
        // Final: rank 0 = 32.0 (received last), rank 1 = 32.0 (doubled last)

        print("{\"finalValue\": \(value)}")

        let expected: Double = 32.0
        if abs(value - expected) > 1e-5 {
            fputs(
                "ERROR: iterative send/recv final value mismatch: got \(value), expected \(expected)\n",
                stderr)
            exit(1)
        }
    }

    /// Multi-dtype allSum test: float16 and int32 arrays across 2 processes
    static func runAllSumMultiDtype(rank: Int, group: DistributedGroup) {
        // float16 test
        let float16Input: MLXArray
        if rank == 0 {
            float16Input = MLXArray(converting: [1.0, 2.0, 3.0]).asType(.float16)
        } else {
            float16Input = MLXArray(converting: [4.0, 5.0, 6.0]).asType(.float16)
        }

        let float16Result = MLXDistributed.allSum(float16Input, group: group)
        eval(float16Result)

        let float16Values = float16Result.asArray(Float.self)
        let float16Dtype = float16Result.dtype

        // int32 test
        let int32Input: MLXArray
        if rank == 0 {
            int32Input = MLXArray([10, 20, 30] as [Int32])
        } else {
            int32Input = MLXArray([40, 50, 60] as [Int32])
        }

        let int32Result = MLXDistributed.allSum(int32Input, group: group)
        eval(int32Result)

        let int32Values = int32Result.asArray(Int32.self)
        let int32Dtype = int32Result.dtype

        print(
            "{\"float16Values\": [\(float16Values.map { String($0) }.joined(separator: ","))], \"float16Dtype\": \"\(float16Dtype)\", \"int32Values\": [\(int32Values.map { String($0) }.joined(separator: ","))], \"int32Dtype\": \"\(int32Dtype)\"}"
        )

        // Verify float16
        let expectedFloat16: [Float] = [5.0, 7.0, 9.0]
        guard float16Dtype == .float16 else {
            fputs("ERROR: float16 dtype mismatch: got \(float16Dtype)\n", stderr)
            exit(1)
        }
        for i in 0 ..< 3 {
            if abs(float16Values[i] - expectedFloat16[i]) > 0.1 {
                fputs(
                    "ERROR: float16 allSum mismatch at \(i): got \(float16Values[i]), expected \(expectedFloat16[i])\n",
                    stderr)
                exit(1)
            }
        }

        // Verify int32
        let expectedInt32: [Int32] = [50, 70, 90]
        guard int32Dtype == .int32 else {
            fputs("ERROR: int32 dtype mismatch: got \(int32Dtype)\n", stderr)
            exit(1)
        }
        for i in 0 ..< 3 {
            if int32Values[i] != expectedInt32[i] {
                fputs(
                    "ERROR: int32 allSum mismatch at \(i): got \(int32Values[i]), expected \(expectedInt32[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// Multi-shape allSum test: [2,3] shaped arrays across 2 processes
    static func runAllSumMultiShape(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshaped([2, 3])
        } else {
            input = MLXArray(converting: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).reshaped([2, 3])
        }

        let result = MLXDistributed.allSum(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        print(
            "{\"values\": [\(values.map { String($0) }.joined(separator: ","))], \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )

        guard shape == [2, 3] else {
            fputs(
                "ERROR: multi-shape allSum shape mismatch: got \(shape), expected [2, 3]\n",
                stderr)
            exit(1)
        }

        let expected: [Float] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        for i in 0 ..< 6 {
            if abs(values[i] - expected[i]) > 1e-5 {
                fputs(
                    "ERROR: multi-shape allSum mismatch at \(i): got \(values[i]), expected \(expected[i])\n",
                    stderr)
                exit(1)
            }
        }
    }

    /// allGather VJP test: compute grad through allGather
    /// On a 2-process group, grad of allGather(x)[0] w.r.t. x should be:
    /// - rank 0: 1.0 (own slice contributes to result[0])
    /// - rank 1: 0.0 (rank 1's slice does not contribute to result[0])
    static func runAllGatherVjp(rank: Int, group: DistributedGroup) {
        let gradFn = grad { (x: MLXArray) -> MLXArray in
            let gathered = MLXDistributed.allGather(x, group: group)
            return gathered[0]
        }

        let x = MLXArray(converting: [1.0])
        let dfdx = gradFn(x)
        eval(dfdx)

        let value = dfdx.asArray(Float.self)[0]

        print("{\"gradValue\": \(value)}")

        let expected: Float = rank == 0 ? 1.0 : 0.0
        if abs(value - expected) > 1e-5 {
            fputs(
                "ERROR: allGather VJP mismatch: got \(value), expected \(expected)\n",
                stderr)
            exit(1)
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
