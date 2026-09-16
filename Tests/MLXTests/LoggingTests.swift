// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

private let lock = NSLock()
nonisolated(unsafe) private var logs = [String: [String]]()

@Suite(.serialized) struct LoggingTests {

    class CollectHandler: MLXLogHandler, @unchecked (Sendable) {

        private let label: String
        public init(label: String) {
            self.label = label
        }

        func log(
            level: MLX.LogLevel, message: () -> String, metadata: () -> [String: String]?,
            file: StaticString, function: StaticString, line: UInt
        ) {
            let message = level.description + ": " + message()
            lock.withLock {
                logs[label, default: []].append(message)
            }
        }
    }

    @Test func testLoggingFactory() async throws {
        MLXLogger.factory = { CollectHandler(label: $0) }
        let logger = MLXLogger(label: "case1")

        logger.debug("test debug")
        logger.info("test info")
        logger.error("test error")

        let list = lock.withLock {
            logs["case1"] ?? []
        }

        #expect(
            list == [
                "debug: test debug",
                "info: test info",
                "error: test error",
            ])
    }

    #if canImport(OSLog)
        @Test func testOSLog() async throws {
            OSLogHandler.install()
            let logger = MLXLogger(label: "case2")

            // no assertions but should run without error
            logger.debug("test debug")
            logger.info("test info")
            logger.error("test error")

            // and not collect
            let list = lock.withLock {
                logs["case2"]
            }
            #expect(list == nil)
        }
    #endif

    @Test func testStderr() async throws {
        StderrHandler.install()
        let logger = MLXLogger(label: "case3")

        // no assertions but should run without error
        logger.debug("test debug")
        logger.info("test info")
        logger.error("test error")

        // and not collect
        let list = lock.withLock {
            logs["case3"]
        }
        #expect(list == nil)
    }

}
