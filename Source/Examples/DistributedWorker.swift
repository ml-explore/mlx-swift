// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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
        MLX.Device.withDefaultDevice(.cpu) {
            runWorker(rank: rank, testOp: testOp)
        }
    }

    static func runWorker(rank: Int, testOp: String) {
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
        case "shardLinearForward":
            runShardLinearForward(rank: rank, group: group)
        case "shardLinearBackward":
            runShardLinearBackward(rank: rank, group: group)
        case "averageGradients":
            runAverageGradients(rank: rank, group: group)
        case "sendRecvMultiDtype":
            runSendRecvMultiDtype(rank: rank, group: group)
        case "allGatherMultiDtype":
            runAllGatherMultiDtype(rank: rank, group: group)
        case "sendRecv2D":
            runSendRecv2D(rank: rank, group: group)
        case "allGather2D":
            runAllGather2D(rank: rank, group: group)
        case "recvLikeMultiDtype":
            runRecvLikeMultiDtype(rank: rank, group: group)
        default:
            fputs("ERROR: Unknown test operation: \(testOp)\n", stderr)
            exit(1)
        }

        fputs("Worker rank=\(rank) completed successfully\n", stderr)

        // Flush all output buffers before terminating. Swift's print() may buffer
        // stdout, so we must ensure JSON results are fully written to the pipe
        // before the process exits.
        fflush(stdout)
        fflush(stderr)

        // Use _exit(0) instead of exit(0) to force immediate process termination.
        // The ring backend's TCP sockets can block in their destructor waiting for
        // peer socket closure, causing exit(0) (which runs atexit handlers and C++
        // destructors) to hang indefinitely. _exit(0) bypasses all cleanup handlers
        // and terminates the process immediately.
        _exit(0)
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
    /// The ring and JACCL backends do not support split. MPI does support it
    /// but is not available on macOS. The ring backend throws
    /// "[ring] Group split not supported." This test verifies that:
    /// 1. The split call is attempted and the error is detected (not a crash)
    /// 2. The parent group remains usable after the failed split
    /// 3. An allSum on the original parent group still works correctly
    ///
    /// When upstream adds split support, this test should be updated to verify
    /// the child group works independently after parent deinit.
    static func runSplit(rank: Int, group: DistributedGroup) {
        // Attempt to split — expect an error from the ring backend
        var splitErrorCaught = false
        do {
            try withError {
                let _ = group.split(color: 0, key: rank)
            }
        } catch {
            fputs("Worker rank=\(rank) split error (expected): \(error)\n", stderr)
            splitErrorCaught = true
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
    /// NOTE: The ring backend does not implement ReduceScatter. Other backends
    /// (NCCL on Linux/CUDA, MPI) do support it. This test detects the error
    /// gracefully and reports the backend limitation rather than crashing.
    static func runSumScatter(rank: Int, group: DistributedGroup) {
        let input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])

        do {
            try withError {
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
        } catch {
            fputs("Worker rank=\(rank) sumScatter error (expected): \(error)\n", stderr)
            print("{\"errorCaught\": true, \"errorMessage\": \"ReduceScatter not implemented\"}")
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

    /// shardLinearForward test: matching Python test_shard_linear forward parity.
    ///
    /// Both ranks seed the PRNG identically, create the same Linear(1024, 1024),
    /// shard it, and forward. Verify:
    ///   - AllToSharded: y[part] == slin1(x) where part is rank's output slice
    ///   - ShardedToAll: y == slin2(x[part]) where part is rank's input slice
    static func runShardLinearForward(rank: Int, group: DistributedGroup) {
        let N = group.size

        // Seed identically on all ranks so Linear weights are the same
        MLXRandom.seed(0xF0F0_F0F0)

        // Create the same input and linear layer on all ranks
        let x = MLXRandom.normal([4, 1024])
        let lin = Linear(1024, 1024, bias: true)
        eval(x, lin)

        // Compute the non-sharded reference output
        let y = lin(x)
        eval(y)

        // Shard to AllToShardedLinear and ShardedToAllLinear
        let slin1 = shardLinear(module: lin, sharding: .allToSharded, group: group) as! UnaryLayer
        let slin2 = shardLinear(module: lin, sharding: .shardedToAll, group: group) as! UnaryLayer
        eval(slin1, slin2)

        // AllToShardedLinear forward: input is full x, output is a slice
        let y1 = slin1(x)
        eval(y1)

        // ShardedToAllLinear forward: input is a slice of x, output is full
        // The input slice for this rank: columns [rank * 1024/N ..< (rank+1) * 1024/N]
        let colStart = rank * 1024 / N
        let colEnd = (rank + 1) * 1024 / N
        let xPart = x[0..., colStart ..< colEnd]
        eval(xPart)
        let y2 = slin2(xPart)
        eval(y2)

        // Verify AllToSharded: y[part] should match y1
        // The output slice for this rank: columns [rank * 1024/N ..< (rank+1) * 1024/N]
        let rowStart = rank * 1024 / N
        let rowEnd = (rank + 1) * 1024 / N
        let yPart = y[0..., rowStart ..< rowEnd]
        eval(yPart)

        // Check AllToSharded forward parity
        let allToShardedClose = yPart.allClose(y1, rtol: 1e-4, atol: 1e-5).item(Bool.self)

        // Check ShardedToAll forward parity
        let shardedToAllClose = y.allClose(y2, rtol: 1e-4, atol: 1e-5).item(Bool.self)

        print(
            "{\"allToShardedMatch\": \(allToShardedClose), \"shardedToAllMatch\": \(shardedToAllClose), \"y1Shape\": [\(y1.shape.map { String($0) }.joined(separator: ","))], \"y2Shape\": [\(y2.shape.map { String($0) }.joined(separator: ","))]}"
        )

        if !allToShardedClose {
            fputs("ERROR: AllToSharded forward parity failed\n", stderr)
            // Print some debug info
            let diff = abs(yPart - y1).max().item(Float.self)
            fputs("  max diff: \(diff)\n", stderr)
            exit(1)
        }

        if !shardedToAllClose {
            fputs("ERROR: ShardedToAll forward parity failed\n", stderr)
            let diff = abs(y - y2).max().item(Float.self)
            fputs("  max diff: \(diff)\n", stderr)
            exit(1)
        }
    }

    /// shardLinearBackward test: matching Python test_shard_linear backward parity.
    ///
    /// Both ranks seed the PRNG identically, create a 4-layer model:
    ///   layers[0] = Linear(128, 128) -> allToSharded
    ///   layers[1] = Linear(128, 128) -> shardedToAll
    ///   layers[2] = Linear(128, 128) -> allToSharded
    ///   layers[3] = Linear(128, 128) -> shardedToAll
    ///
    /// Compute gradient of dummy_loss = sum(model(x) * y).
    /// Verify that each rank's sharded weight/bias gradients match the
    /// corresponding slice of the non-sharded model's gradients.
    static func runShardLinearBackward(rank: Int, group: DistributedGroup) {
        let N = group.size

        // Seed identically on all ranks
        MLXRandom.seed(0xF0F0_F0F0)

        // Create the non-sharded 4-layer model
        let mod = Sequential(
            layers:
                Linear(128, 128, bias: true),
            Linear(128, 128, bias: true),
            Linear(128, 128, bias: true),
            Linear(128, 128, bias: true)
        )
        eval(mod)

        // Create the sharded version from the same weights
        let smod = Sequential(
            layers:
                shardLinear(
                    module: mod.layers[0], sharding: .allToSharded,
                    group: group) as! UnaryLayer,
            shardLinear(
                module: mod.layers[1], sharding: .shardedToAll,
                group: group) as! UnaryLayer,
            shardLinear(
                module: mod.layers[2], sharding: .allToSharded,
                group: group) as! UnaryLayer,
            shardLinear(
                module: mod.layers[3], sharding: .shardedToAll,
                group: group) as! UnaryLayer
        )
        eval(smod)

        // Create the same input and target on all ranks
        let x = MLXRandom.normal([4, 128])
        let yTarget = MLXRandom.normal([4, 128])
        eval(x, yTarget)

        // Define loss function: sum(model(x) * y)
        func dummyLoss(model: Sequential, x: MLXArray, y: MLXArray) -> MLXArray {
            (model(x) * y).sum()
        }

        // Compute value and gradients for the non-sharded model
        let grad1 = valueAndGrad(model: mod, dummyLoss)
        let (l1, g1) = grad1(mod, x, yTarget)
        eval(l1, g1)

        // Compute value and gradients for the sharded model
        let grad2 = valueAndGrad(model: smod, dummyLoss)
        let (l2, g2) = grad2(smod, x, yTarget)
        eval(l2, g2)

        // The rank's slice for dimension 128
        let part = rank * 128 / N ..< (rank + 1) * 128 / N

        // Verify losses match
        let lossMatch = l1.allClose(l2).item(Bool.self)

        // Extract gradients via flattened key paths.
        // The flattened keys for a Sequential of Linears are:
        //   "layers.0.weight", "layers.0.bias", "layers.1.weight", ...
        let g1Flat = Dictionary(uniqueKeysWithValues: g1.flattened())
        let g2Flat = Dictionary(uniqueKeysWithValues: g2.flattened())

        // Helper to get a gradient array by key path
        func g1Array(_ key: String) -> MLXArray { g1Flat[key]! }
        func g2Array(_ key: String) -> MLXArray { g2Flat[key]! }

        // Check layer 0 (allToSharded): g1.weight[part, :] == g2.weight
        let l0WeightMatch = g1Array("layers.0.weight")[part].allClose(
            g2Array("layers.0.weight"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 0 bias: g1.bias[part] == g2.bias
        let l0BiasMatch = g1Array("layers.0.bias")[part].allClose(
            g2Array("layers.0.bias"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 1 (shardedToAll): g1.weight[:, part] == g2.weight
        let l1WeightMatch = g1Array("layers.1.weight")[0..., part].allClose(
            g2Array("layers.1.weight"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 1 bias: g1.bias == g2.bias (shardedToAll bias is not sharded)
        let l1BiasMatch = g1Array("layers.1.bias").allClose(
            g2Array("layers.1.bias"), rtol: 1e-4, atol: 1e-5
        ).item(Bool.self)

        // Check layer 2 (allToSharded): g1.weight[part, :] == g2.weight
        let l2WeightMatch = g1Array("layers.2.weight")[part].allClose(
            g2Array("layers.2.weight"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 2 bias: g1.bias[part] == g2.bias
        let l2BiasMatch = g1Array("layers.2.bias")[part].allClose(
            g2Array("layers.2.bias"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 3 (shardedToAll): g1.weight[:, part] == g2.weight
        let l3WeightMatch = g1Array("layers.3.weight")[0..., part].allClose(
            g2Array("layers.3.weight"), rtol: 1e-4, atol: 1e-6
        ).item(Bool.self)

        // Check layer 3 bias: g1.bias == g2.bias (shardedToAll bias is not sharded)
        let l3BiasMatch = g1Array("layers.3.bias").allClose(
            g2Array("layers.3.bias"), rtol: 1e-4, atol: 1e-5
        ).item(Bool.self)

        print(
            "{\"lossMatch\": \(lossMatch), \"l0WeightMatch\": \(l0WeightMatch), \"l0BiasMatch\": \(l0BiasMatch), \"l1WeightMatch\": \(l1WeightMatch), \"l1BiasMatch\": \(l1BiasMatch), \"l2WeightMatch\": \(l2WeightMatch), \"l2BiasMatch\": \(l2BiasMatch), \"l3WeightMatch\": \(l3WeightMatch), \"l3BiasMatch\": \(l3BiasMatch)}"
        )

        // Verify all match
        if !lossMatch {
            fputs("ERROR: Losses don't match between sharded and non-sharded models\n", stderr)
            let diff = abs(l1 - l2).item(Float.self)
            fputs("  loss diff: \(diff)\n", stderr)
            exit(1)
        }

        let checks: [(String, Bool)] = [
            ("layer0 weight", l0WeightMatch),
            ("layer0 bias", l0BiasMatch),
            ("layer1 weight", l1WeightMatch),
            ("layer1 bias", l1BiasMatch),
            ("layer2 weight", l2WeightMatch),
            ("layer2 bias", l2BiasMatch),
            ("layer3 weight", l3WeightMatch),
            ("layer3 bias", l3BiasMatch),
        ]

        for (name, matched) in checks {
            if !matched {
                fputs("ERROR: \(name) gradient parity failed\n", stderr)
                exit(1)
            }
        }
    }

    /// averageGradients test: exercises batched allSum, non-batched, and communicationType
    /// paths with a 2-process group (N==2), so the early-return `if N == 1` is bypassed.
    ///
    /// Rank 0: weight=[2,4,6], bias=[10]
    /// Rank 1: weight=[4,8,12], bias=[20]
    /// Expected average: weight=[3,6,9], bias=[15]
    static func runAverageGradients(rank: Int, group: DistributedGroup) {
        // Build a gradient tree with known per-rank values
        let weight: MLXArray
        let bias: MLXArray
        if rank == 0 {
            weight = MLXArray(converting: [2.0, 4.0, 6.0])
            bias = MLXArray(converting: [10.0])
        } else {
            weight = MLXArray(converting: [4.0, 8.0, 12.0])
            bias = MLXArray(converting: [20.0])
        }
        eval(weight, bias)

        var grads = ModuleParameters()
        grads["weight"] = .value(weight)
        grads["bias"] = .value(bias)

        let expectedWeight: [Float] = [3.0, 6.0, 9.0]
        let expectedBias: [Float] = [15.0]

        // 1. Default averageGradients (batched allSum path)
        let avg1 = averageGradients(gradients: grads, group: group)
        let avg1Flat = Dictionary(uniqueKeysWithValues: avg1.flattened())
        let avg1Weight = avg1Flat["weight"]!.asArray(Float.self)
        let avg1Bias = avg1Flat["bias"]!.asArray(Float.self)

        var defaultMatch = true
        for i in 0 ..< 3 {
            if abs(avg1Weight[i] - expectedWeight[i]) > 1e-4 { defaultMatch = false }
        }
        if abs(avg1Bias[0] - expectedBias[0]) > 1e-4 { defaultMatch = false }

        // 2. Non-batched path (allReduceSize=0)
        let avg2 = averageGradients(gradients: grads, group: group, allReduceSize: 0)
        let avg2Flat = Dictionary(uniqueKeysWithValues: avg2.flattened())
        let avg2Weight = avg2Flat["weight"]!.asArray(Float.self)
        let avg2Bias = avg2Flat["bias"]!.asArray(Float.self)

        var unbatchedMatch = true
        for i in 0 ..< 3 {
            if abs(avg2Weight[i] - expectedWeight[i]) > 1e-4 { unbatchedMatch = false }
        }
        if abs(avg2Bias[0] - expectedBias[0]) > 1e-4 { unbatchedMatch = false }

        // 3. communicationType: .float16 (cast-on-wire)
        let avg3 = averageGradients(
            gradients: grads, group: group, communicationType: .float16)
        let avg3Flat = Dictionary(uniqueKeysWithValues: avg3.flattened())
        let avg3Weight = avg3Flat["weight"]!
        let avg3Bias = avg3Flat["bias"]!
        let avg3WeightValues = avg3Weight.asArray(Float.self)
        let avg3BiasValues = avg3Bias.asArray(Float.self)

        // Verify the output dtype is still float32 (preserved after round-trip)
        let commTypeDtype = String(describing: avg3Weight.dtype)

        var commTypeMatch = true
        for i in 0 ..< 3 {
            // float16 round-trip allows slightly larger tolerance
            if abs(avg3WeightValues[i] - expectedWeight[i]) > 0.1 { commTypeMatch = false }
        }
        if abs(avg3BiasValues[0] - expectedBias[0]) > 0.1 { commTypeMatch = false }

        print(
            "{\"defaultMatch\": \(defaultMatch), \"unbatchedMatch\": \(unbatchedMatch), \"commTypeMatch\": \(commTypeMatch), \"commTypeDtype\": \"\(commTypeDtype)\"}"
        )
    }

    /// sendRecvMultiDtype test: rank 0 sends float16, int32, bfloat16 arrays to rank 1
    static func runSendRecvMultiDtype(rank: Int, group: DistributedGroup) {
        if rank == 0 {
            let f16 = MLXArray(converting: [1.0, 2.0]).asType(.float16)
            let i32 = MLXArray([100, 200] as [Int32])
            let bf16 = MLXArray(converting: [0.5, 1.5]).asType(.bfloat16)
            eval(f16, i32, bf16)

            let t1 = MLXDistributed.send(f16, to: 1, group: group)
            eval(t1)
            let t2 = MLXDistributed.send(i32, to: 1, group: group)
            eval(t2)
            let t3 = MLXDistributed.send(bf16, to: 1, group: group)
            eval(t3)

            print(
                "{\"float16Match\": true, \"int32Match\": true, \"bfloat16Match\": true}"
            )
        } else {
            let recvF16 = MLXDistributed.recv(
                shape: [2], dtype: .float16, from: 0, group: group)
            eval(recvF16)
            let recvI32 = MLXDistributed.recv(
                shape: [2], dtype: .int32, from: 0, group: group)
            eval(recvI32)
            let recvBf16 = MLXDistributed.recv(
                shape: [2], dtype: .bfloat16, from: 0, group: group)
            eval(recvBf16)

            let f16Values = recvF16.asArray(Float.self)
            let i32Values = recvI32.asArray(Int32.self)
            let bf16Values = recvBf16.asArray(Float.self)

            let float16Match =
                abs(f16Values[0] - 1.0) < 0.1 && abs(f16Values[1] - 2.0) < 0.1
            let int32Match = i32Values[0] == 100 && i32Values[1] == 200
            let bfloat16Match =
                abs(bf16Values[0] - 0.5) < 0.1 && abs(bf16Values[1] - 1.5) < 0.1

            print(
                "{\"float16Match\": \(float16Match), \"int32Match\": \(int32Match), \"bfloat16Match\": \(bfloat16Match)}"
            )
        }
    }

    /// allGatherMultiDtype test: float16 and int32 allGather across 2 processes
    static func runAllGatherMultiDtype(rank: Int, group: DistributedGroup) {
        // float16 test: rank 0 [1,2], rank 1 [3,4] -> gathered [1,2,3,4]
        let f16Input: MLXArray
        if rank == 0 {
            f16Input = MLXArray(converting: [1.0, 2.0]).asType(.float16)
        } else {
            f16Input = MLXArray(converting: [3.0, 4.0]).asType(.float16)
        }
        eval(f16Input)

        let f16Result = MLXDistributed.allGather(f16Input, group: group)
        eval(f16Result)

        let f16Values = f16Result.asArray(Float.self)
        let f16Expected: [Float] = [1.0, 2.0, 3.0, 4.0]
        var float16Match = f16Result.shape == [4]
        for i in 0 ..< 4 {
            if abs(f16Values[i] - f16Expected[i]) > 0.1 { float16Match = false }
        }

        // int32 test: rank 0 [10], rank 1 [20] -> gathered [10, 20]
        let i32Input: MLXArray
        if rank == 0 {
            i32Input = MLXArray([10] as [Int32])
        } else {
            i32Input = MLXArray([20] as [Int32])
        }
        eval(i32Input)

        let i32Result = MLXDistributed.allGather(i32Input, group: group)
        eval(i32Result)

        let i32Values = i32Result.asArray(Int32.self)
        let int32Match =
            i32Result.shape == [2] && i32Values[0] == 10 && i32Values[1] == 20

        print(
            "{\"float16Match\": \(float16Match), \"int32Match\": \(int32Match), \"float16Shape\": [\(f16Result.shape.map { String($0) }.joined(separator: ","))], \"int32Shape\": [\(i32Result.shape.map { String($0) }.joined(separator: ","))]}"
        )
    }

    /// sendRecv2D test: rank 0 sends a [2,3] float32 array, rank 1 receives and verifies
    static func runSendRecv2D(rank: Int, group: DistributedGroup) {
        if rank == 0 {
            let data = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshaped([2, 3])
            eval(data)
            let token = MLXDistributed.send(data, to: 1, group: group)
            eval(token)

            print("{\"valuesMatch\": true, \"shape\": [2,3]}")
        } else {
            let received = MLXDistributed.recv(
                shape: [2, 3], dtype: .float32, from: 0, group: group)
            eval(received)

            let values = received.asArray(Float.self)
            let shape = received.shape

            let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            var valuesMatch = shape == [2, 3]
            for i in 0 ..< 6 {
                if abs(values[i] - expected[i]) > 1e-5 { valuesMatch = false }
            }

            print(
                "{\"valuesMatch\": \(valuesMatch), \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
            )
        }
    }

    /// allGather2D test: rank 0 [[1,2],[3,4]], rank 1 [[5,6],[7,8]]
    /// After allGather along axis 0: [[1,2],[3,4],[5,6],[7,8]] shape [4,2]
    static func runAllGather2D(rank: Int, group: DistributedGroup) {
        let input: MLXArray
        if rank == 0 {
            input = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped([2, 2])
        } else {
            input = MLXArray(converting: [5.0, 6.0, 7.0, 8.0]).reshaped([2, 2])
        }
        eval(input)

        let result = MLXDistributed.allGather(input, group: group)
        eval(result)

        let values = result.asArray(Float.self)
        let shape = result.shape

        let expected: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        var valuesMatch = shape == [4, 2]
        for i in 0 ..< 8 {
            if abs(values[i] - expected[i]) > 1e-5 { valuesMatch = false }
        }

        print(
            "{\"valuesMatch\": \(valuesMatch), \"shape\": [\(shape.map { String($0) }.joined(separator: ","))]}"
        )
    }

    /// recvLikeMultiDtype test: rank 0 sends float16 and int32 arrays,
    /// rank 1 uses recvLike with matching templates to verify dtype preservation
    static func runRecvLikeMultiDtype(rank: Int, group: DistributedGroup) {
        if rank == 0 {
            let f16 = MLXArray(converting: [1.0, 2.0]).asType(.float16)
            let i32 = MLXArray([100, 200] as [Int32])
            eval(f16, i32)

            let t1 = MLXDistributed.send(f16, to: 1, group: group)
            eval(t1)
            let t2 = MLXDistributed.send(i32, to: 1, group: group)
            eval(t2)

            print(
                "{\"float16Match\": true, \"float16Dtype\": \"float16\", \"int32Match\": true, \"int32Dtype\": \"int32\"}"
            )
        } else {
            let f16Template = MLXArray(converting: [0.0, 0.0]).asType(.float16)
            let i32Template = MLXArray([0, 0] as [Int32])
            eval(f16Template, i32Template)

            let recvF16 = MLXDistributed.recvLike(f16Template, from: 0, group: group)
            eval(recvF16)
            let recvI32 = MLXDistributed.recvLike(i32Template, from: 0, group: group)
            eval(recvI32)

            let f16Values = recvF16.asArray(Float.self)
            let i32Values = recvI32.asArray(Int32.self)

            let float16Match =
                abs(f16Values[0] - 1.0) < 0.1 && abs(f16Values[1] - 2.0) < 0.1
            let int32Match = i32Values[0] == 100 && i32Values[1] == 200
            let float16Dtype = String(describing: recvF16.dtype)
            let int32Dtype = String(describing: recvI32.dtype)

            print(
                "{\"float16Match\": \(float16Match), \"float16Dtype\": \"\(float16Dtype)\", \"int32Match\": \(int32Match), \"int32Dtype\": \"\(int32Dtype)\"}"
            )
        }
    }
}
