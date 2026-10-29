// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

private func triggerError() {
    let a = MLXArray(0 ..< 10, [2, 5])
    let b = MLXArray(0 ..< 15, [3, 5])

    // note: there are actually two errors here:
    // - broadcast
    // - eval: non-empty mlx_vector_array
    eval(a + b)
}

@Suite
struct ErrorTests {

    @Test func errorHandler() throws {
        // a test with a custom error handler that captures the first error seen
        final class BoxedError: @unchecked (Sendable) {
            let lock = NSLock()
            var _error: String?
            var error: String? {
                get {
                    lock.withLock { _error }
                }
                set {
                    lock.withLock {
                        // take the first error
                        if _error == nil {
                            _error = newValue
                        }
                    }
                }
            }
        }
        let boxedError = BoxedError()

        @Sendable
        func handler(_ message: String) {
            boxedError.error = message
        }

        withErrorHandler(handler) {
            triggerError()
        }

        let error = try #require(boxedError.error, "boxedError.error should be set")
        let prefix = error.split(separator: ". at")[0]
        #expect(prefix == "[broadcast_shapes] Shapes (2,5) and (3,5) cannot be broadcast")
    }

    @Test func withErrorCheck() {
        #expect(throws: MLXError.self) {
            try withError { error in
                triggerError()
                try error.check()
            }
        }
    }

    @Test func withErrorThrow() {
        // outer container should throw
        #expect(throws: MLXError.self) {
            try withError {
                triggerError()
            }
        }
    }

    @Test func withErrorThrowAsync() async {
        await #expect(throws: MLXError.self) {
            try await withError {
                let t = Task {
                    triggerError()
                }
                await t.value
            }
        }
    }

    @Test func withErrorThrowNested() throws {
        // the outer layer does not produce an error -- if it throws the test fails
        let _ = try withError {
            // the inner container should throw
            #expect(throws: MLXError.self) {
                try withError {
                    triggerError()
                }
            }
        }
    }

}
