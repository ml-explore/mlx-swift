// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import Testing

@testable import MLX

/// Single process tests for ``MLXDistributed``.
///
/// Without `MLX_HOSTFILE` / `MLX_RANK` no backend can form a communication
/// group, so `initialize()` yields an empty group of size one.  At that size
/// MLX short circuits the collectives to the identity and rejects the point to
/// point operations, which is enough to cover the whole API surface in CI.
///
/// Multi process behaviour is verified by launching several copies of a
/// program with a hostfile, matching how the Python tests are run.
///
/// The assertions here describe a group of size one, so they are not valid in a
/// multi process run -- see ``DistributedRingTests``.
///
/// Serialized because the tests share process-wide state: the environment
/// (``initializeThrowsWhenConfiguredBackendFails()`` sets `MLX_HOSTFILE` /
/// `MLX_RANK`, which `initialize()` reads) and the registered backends
/// (``defaultGroupIsSingleton()`` calls `clearBackends()`).
@Suite(
    .serialized,
    .disabled(
        if: ProcessInfo.processInfo.environment["MLX_TEST_DISTRIBUTED"] == "1",
        "Single process assertions; skipped during a multi process run."))
struct DistributedTests {

    // MARK: - Availability

    @Test func isAvailableLinks() {
        for backend in MLXDistributed.Backend.allCases {
            _ = MLXDistributed.isAvailable(backend)
        }
    }

    /// The ring backend is built on all platforms -- it needs only TCP sockets.
    @Test func ringIsAvailable() {
        #expect(MLXDistributed.isAvailable(.ring))

        // `any` is true when at least one backend is available
        #expect(MLXDistributed.isAvailable(.any))
    }

    /// Backends that are not compiled in report `false` rather than trapping.
    @Test func unavailableBackendsReportFalse() {
        #expect(!MLXDistributed.isAvailable(.mpi))
        #if !os(macOS)
            #expect(!MLXDistributed.isAvailable(.jaccl))
        #endif

        // nccl is CUDA-only and is never built on Apple platforms
        #if !os(Linux)
            #expect(!MLXDistributed.isAvailable(.nccl))
        #endif
    }

    // MARK: - Group

    @Test func defaultGroupIsSingleton() throws {
        let group = try MLXDistributed.initialize()
        #expect(try group.rank == 0)
        #expect(try group.size == 1)

        try MLXDistributed.clearBackends()
        #expect(try group.size == 1)
    }

    @Test func emptyGroupGettersReportErrors() {
        let group = MLXDistributed.Group(mlx_distributed_group_new())

        #expect(throws: MLXError.self) { try group.rank }
        #expect(throws: MLXError.self) { try group.size }
    }

    /// A strict init throws when no backend can form a group.
    ///
    /// Note this uses `.any`: MLX caches successfully registered groups per
    /// backend name, but never caches under "any" when initialization fails,
    /// so this is independent of the order the tests run in.
    @Test func strictInitializeThrows() {
        #expect(throws: MLXError.self) {
            try MLXDistributed.initialize(backend: .any, strict: true)
        }
    }

    /// A backend that is configured but cannot form its group throws even when
    /// the init is not strict -- here the ring backend with a missing hostfile.
    @Test(
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_HOSTFILE"] != nil
                || ProcessInfo.processInfo.environment["MLX_RANK"] != nil,
            "MLX_HOSTFILE or MLX_RANK is already set"))
    func initializeThrowsWhenConfiguredBackendFails() throws {
        let hostfile = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-hostfile-\(UUID().uuidString).json")
        setenv("MLX_HOSTFILE", hostfile.path, 1)
        setenv("MLX_RANK", "0", 1)
        defer {
            unsetenv("MLX_HOSTFILE")
            unsetenv("MLX_RANK")
        }

        #expect(throws: MLXError.self) {
            try MLXDistributed.initialize(backend: .any)
        }
    }

    @Test func splitSingletonGroupThrows() throws {
        let group = try MLXDistributed.initialize()
        #expect(throws: MLXError.self) { try group.split(color: 0) }
    }

    // MARK: - Metadata all-gather initialization

    @Test func metadataAllGatherBridgeForwardsBorrowedBuffers() {
        let expected: [UInt8] = [0, 255, 127, 128]
        let bridge = makeMetadataAllGatherBridge { rank, size, source, destination in
            #expect(rank == 1)
            #expect(size == 2)
            #expect(Array(source) == [0, 255])
            #expect(destination.count == expected.count)
            guard destination.count == expected.count else { return }
            destination.copyBytes(from: expected)
        }
        defer { bridge.dtor(bridge.payload) }

        let source: [CChar] = [0, -1]
        var destination = [CChar](repeating: -2, count: 4)
        let status = source.withUnsafeBufferPointer { source in
            destination.withUnsafeMutableBufferPointer { destination in
                bridge.callback(
                    1, 2, source.baseAddress, destination.baseAddress, source.count, bridge.payload)
            }
        }

        #expect(status == 0)
        #expect(destination.map { UInt8(bitPattern: $0) } == expected)
    }

    @Test func metadataAllGatherBridgeConvertsThrownErrorToFailureStatus() {
        let bridge = makeMetadataAllGatherBridge { _, _, _, destination in
            destination[0] = 42
            throw MetadataAllGatherTestError.callbackFailure
        }
        defer { bridge.dtor(bridge.payload) }

        let source: [CChar] = [42]
        var destination: [CChar] = [-2, -2]
        let status = source.withUnsafeBufferPointer { source in
            destination.withUnsafeMutableBufferPointer { destination in
                bridge.callback(
                    0, 2, source.baseAddress, destination.baseAddress, source.count, bridge.payload)
            }
        }

        #expect(status != 0)
        #expect(destination == [42, -2])
    }

    @Test func metadataAllGatherFailureReleasesCapture() {
        weak var captured: MetadataAllGatherCapture?
        do {
            let capture = MetadataAllGatherCapture()
            captured = capture
            #expect(throws: MLXError.self) {
                try MLXDistributed.initialize(backend: .ring) { [capture] _, _, _, _ in
                    let _ = withExtendedLifetime(capture) {
                        Issue.record("Callback should not run when initialization fails")
                    }
                }
            }
        }
        #expect(captured == nil)
    }

    // MARK: - Collectives

    /// In a group of size one the collectives return their input unchanged.
    @Test func collectivesAreIdentityInSingletonGroup() {
        let x = MLXArray([1, 2, 3, 4], [2, 2])

        expectEqual(MLXDistributed.allSum(x), x)
        expectEqual(MLXDistributed.allMax(x), x)
        expectEqual(MLXDistributed.allMin(x), x)
        expectEqual(MLXDistributed.allGather(x), x)
        expectEqual(MLXDistributed.sumScatter(x), x)
    }

    // MARK: - Point to point

    @Test func sendInSingletonGroupReportsError() {
        let x = MLXArray([1, 2, 3])
        #expect(throws: MLXError.self) {
            try withError {
                MLXDistributed.send(x, to: 0)
            }
        }
    }

    @Test func recvInSingletonGroupReportsError() {
        #expect(throws: MLXError.self) {
            try withError {
                MLXDistributed.recv([3], dtype: .int32, from: 0)
            }
        }
    }

    @Test func recvLikeInSingletonGroupReportsError() {
        let x = MLXArray([1, 2, 3])
        #expect(throws: MLXError.self) {
            try withError {
                MLXDistributed.recvLike(x, from: 0)
            }
        }
    }
}

private enum MetadataAllGatherTestError: Error {
    case callbackFailure
}

private final class MetadataAllGatherCapture: Sendable {}
