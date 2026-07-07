// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import XCTest

@testable import MLX

/// Test-local throwing sync point (the proposed `try eval` API from the
/// sync-points patch), shadowing the non-throwing `MLX.eval` pending the
/// enum-vs-struct / throwing-API decision on #270. The *mechanism* under test
/// (thread-local slot -> checkStatus -> native throw, poison propagation)
/// lives entirely in the library.
private func eval(_ arrays: MLXArray...) throws {
    for a in arrays { try a.throwIfPoisoned() }
    let vector = new_mlx_vector_array(arrays)
    defer { mlx_vector_array_free(vector) }
    try checkStatus(mlx_eval(vector))
}

/// Exercises the pull-based structured error boundary (issue #270).
final class ErrorBoundaryTests: XCTestCase {

    /// Route errors through status codes + thread-local slot instead of the
    /// legacy push handler (which would exit/fatalError the test process).
    override class func setUp() {
        super.setUp()
        installPullErrorBarrier()
    }

    /// Eager error: broadcast mismatch throws at the synchronization point with
    /// the correct classification, in a plain `do/catch` — no `withError`.
    func testBroadcastMismatchThrows() throws {
        let a = MLXArray(0 ..< 10, [2, 5])
        let b = MLXArray(0 ..< 15, [3, 5])

        do {
            let c = a + b            // poisoned, non-throwing op
            try MLXTests.eval(c)              // rethrows here
            XCTFail("expected MLXError")
        } catch let error as MLXError {
            XCTAssertEqual(error.code, .invalidArgument)
            XCTAssertFalse(error.message.isEmpty)
        }
    }

    /// Deferred error: an allocation failure during eval surfaces as `.outOfMemory`.
    ///
    /// Note: `Memory.memoryLimit` is a *soft* scheduler target in current MLX and
    /// exceeding it does not reliably raise. The deterministic trigger is a single
    /// allocation over Metal's max buffer size, which throws
    /// `[metal::malloc] ... greater than the maximum allowed buffer size` —
    /// classified to OOM by the mlx-c refinement.
    func testOutOfMemoryClassified() throws {
        do {
            // ~4 PiB request: exceeds max buffer size on every device.
            let big = MLXArray.ones([1 << 20, 1 << 20], dtype: .float32)
            try MLXTests.eval(big)
            XCTFail("expected OOM")
        } catch let error as MLXError {
            XCTAssertEqual(error.code, .outOfMemory)
        }
    }

    /// I/O error: loading a corrupt safetensors throws `.io`, distinguishable in
    /// the same catch as a shape bug.
    func testCorruptLoadThrowsIO() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("corrupt.safetensors")
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try loadArrays(url: url)) { error in
            guard let mlx = error as? MLXError else { return XCTFail("wrong type") }
            XCTAssertEqual(mlx.code, .io)
        }
    }

    /// The bug that fatalErrors today: an error raised on a *background* thread
    /// with no task-local handler must reach the caller's `do/catch`, because
    /// the error slot is thread-local and read via the status code.
    func testErrorOnBackgroundThreadReachesCaller() throws {
        let expectation = expectation(description: "caught on worker")
        var caught: MLXError?

        DispatchQueue.global().async {
            do {
                let a = MLXArray(0 ..< 10, [2, 5])
                let b = MLXArray(0 ..< 15, [3, 5])
                try MLXTests.eval(a + b)
            } catch let error as MLXError {
                caught = error
            } catch {}
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5)
        XCTAssertEqual(caught?.code, .invalidArgument)
    }

    /// No cross-thread error bleed: a failure on one thread must not corrupt a
    /// concurrent success on another. Errors are raised at graph-construction
    /// time (eager shape inference, no Metal), so this exercises the
    /// thread-local slot + poison isolation deterministically. Concurrent GPU
    /// `eval` is deliberately avoided: mlx core does not guarantee thread
    /// safety for concurrent evaluation (crashes in the Metal encoder), which
    /// is an upstream constraint independent of the error boundary.
    func testNoCrossThreadBleed() throws {
        let group = DispatchGroup()
        let failures = NSMutableArray()
        let lock = NSLock()

        for i in 0 ..< 64 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                if i.isMultiple(of: 2) {
                    // broadcast mismatch: poisoned with .invalidArgument on THIS thread
                    let bad = MLXArray(0 ..< 10, [2, 5]) + MLXArray(0 ..< 15, [3, 5])
                    if bad.poisonError?.code != .invalidArgument {
                        lock.withLock {
                            failures.add("even \(i): expected invalidArgument poison, got \(String(describing: bad.poisonError))")
                        }
                    }
                } else {
                    // clean add: must NOT observe any other thread's error
                    let ok = MLXArray(0 ..< 10, [2, 5]) + MLXArray(0 ..< 10, [2, 5])
                    if let error = ok.poisonError {
                        lock.withLock { failures.add("odd \(i): unexpected poison \(error)") }
                    }
                }
            }
        }

        group.wait()
        XCTAssertEqual(failures.count, 0, "\(failures)")

        // sync-point sanity check, serialized after the storm
        let clean = MLXArray(0 ..< 10, [2, 5]) + MLXArray(0 ..< 10, [2, 5])
        try MLXTests.eval(clean)
    }

    /// Poison stops the zombie cascade: using the failed array again rethrows
    /// the *original* first error, not a secondary "empty array" error.
    func testPoisonCarriesFirstError() throws {
        let a = MLXArray(0 ..< 10, [2, 5])
        let b = MLXArray(0 ..< 15, [3, 5])
        let bad = a + b                 // poisoned

        XCTAssertEqual(bad.poisonError?.code, .invalidArgument)

        let downstream = bad + a        // still poisoned with first error
        XCTAssertThrowsError(try MLXTests.eval(downstream)) { error in
            XCTAssertEqual((error as? MLXError)?.code, .invalidArgument)
        }
    }
}
