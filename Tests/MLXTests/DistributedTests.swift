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
}
