// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

private func triggerError() {
    let a = MLXArray(0 ..< 10, [2, 5])
    let b = MLXArray(0 ..< 15, [3, 5])

    // note: there are actually two errors here:
    // - broadcast
    // - eval: non-empty mlx_vector_array
    eval(a + b)
}

class ErrorTests: XCTestCase {

    func testErrorHandler() {
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

        if let error = boxedError.error {
            let prefix = error.split(separator: ". at")[0]
            XCTAssertEqual(prefix, "[broadcast_shapes] Shapes (2,5) and (3,5) cannot be broadcast")
        } else {
            XCTFail("boxedError.error should be set")
        }
    }

    func testWithErrorCheck() {
        do {
            try withError { error in
                triggerError()
                try error.check()
            }
            XCTFail("should throw")
        } catch {
            print(error)
        }
    }

    func testWithErrorThrow() {
        do {
            // outer container should throw
            try withError {
                triggerError()
            }
            XCTFail("should throw")
        } catch {
            print(error)
        }
    }

    func testWithErrorThrowAsync() async {
        do {
            try await withError {
                let t = Task {
                    triggerError()
                }
                await t.value
            }
            XCTFail("should throw")
        } catch {
            print(error)
        }
    }

    func testWithErrorThrowNested() {
        do {
            // outer container should throw
            try withError {
                do {
                    try withError {
                        triggerError()
                    }
                    XCTFail("should throw")
                } catch {
                    // expected
                }
                // the outer layer does not produce an error
            }
        } catch {
            XCTFail("unexpected outer error: \(error)")
        }
    }

}
