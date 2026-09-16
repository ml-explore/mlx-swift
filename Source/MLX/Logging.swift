// Copyright © 2026 Apple Inc.

import Foundation

public enum LogLevel: Int, Comparable, Sendable, CustomStringConvertible {
    case trace = 0
    case debug = 1
    case info = 2
    case warning = 3
    case error = 4

    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String {
        switch self {
        case .trace: "trace"
        case .debug: "debug"
        case .info: "info"
        case .warning: "warning"
        case .error: "error"
        }
    }
}

/// Protocol for backing of MLXLogger.
///
/// Install with ``MLXLogger/factory`` or ``StderrHandler/install()``
public protocol MLXLogHandler: Sendable {
    func log(
        level: LogLevel,
        message: () -> String,
        metadata: () -> [String: String]?,
        file: StaticString,
        function: StaticString,
        line: UInt
    )
}

public typealias MLXLogHandlerFactory = @Sendable (String) -> any MLXLogHandler

#if canImport(OSLog)
    private let defaultFactory: MLXLogHandlerFactory = { OSLogHandler(label: $0) }
#else
    private let defaultFactory: MLXLogHandlerFactory = { StderrHandler(label: $0) }
#endif

private let factoryLock = NSLock()
#if swift(>=5.10)
    nonisolated(unsafe) private var handlerFactory = defaultFactory
#else
    private var handlerFactory = defaultFactory
#endif

/// Thin wrapper for logging in MLX and MLX libraries.  This
/// provides a hook for library code if it needs logging without
/// adding a dependency or requiring a certain logging backend.
///
/// By default it will log to `OSLog` (macOS, iOS) or `stderr`
/// (others).  Developers can override the backing by implementing
/// ``MLXLogHandler`` and installing it with ``MLXLogger/factory``,
/// e.g.
///
/// ```swift
/// MLXLogger.factory = { StderrHandler(label: $0) }
/// ```
///
/// The factory is used at the time of the first log for the given
/// logger -- whatever factory is installed at that time will be used.
///
/// Callers can use this the same way that `swift-log` is used:
///
/// ```swift
/// private let logger = MLXLogger(label: "KVCache")
///
/// func ...() {
///     logger.error("Failed to ...")
/// }
/// ```
public final class MLXLogger: @unchecked (Sendable) {

    private let label: String

    /// backing that is realized at first log time
    private let lock = NSLock()
    private var backing: MLXLogHandler?

    public static var factory: MLXLogHandlerFactory {
        get {
            factoryLock.withLock { handlerFactory }
        }
        set {
            factoryLock.withLock { handlerFactory = newValue }
        }
    }

    public init(label: String) {
        self.label = label
    }

    private func _backing() -> MLXLogHandler {
        if let backing = lock.withLock({ backing }) {
            return backing
        }

        let backing = Self.factory(label)
        lock.withLock { self.backing = backing }
        return backing
    }

    fileprivate func _log(
        level: LogLevel,
        message: () -> String, metadata: () -> [String: String]?,
        file: StaticString, function: StaticString, line: UInt
    ) {
        _backing().log(
            level: level,
            message: message, metadata: metadata,
            file: file, function: function, line: line)
    }
}

extension MLXLogger {

    /// Log with the given level and message.
    public func log(
        level: LogLevel,
        message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]?,
        file: StaticString = #fileID,
        function: StaticString = #function, line: UInt = #line
    ) {
        _log(
            level: level, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }

    /// Log a trace message.
    public func trace(
        _ message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]? = nil,
        file: StaticString = #fileID,
        function: StaticString = #function,
        line: UInt = #line
    ) {
        _log(
            level: .trace, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }

    /// Log a debug message.
    public func debug(
        _ message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]? = nil,
        file: StaticString = #fileID,
        function: StaticString = #function,
        line: UInt = #line
    ) {
        _log(
            level: .debug, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }

    /// Log a info message.
    public func info(
        _ message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]? = nil,
        file: StaticString = #fileID,
        function: StaticString = #function,
        line: UInt = #line
    ) {
        _log(
            level: .info, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }

    /// Log a warning message.
    public func warning(
        _ message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]? = nil,
        file: StaticString = #fileID,
        function: StaticString = #function,
        line: UInt = #line
    ) {
        _log(
            level: .warning, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }

    /// Log an error message.
    public func error(
        _ message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]? = nil,
        file: StaticString = #fileID,
        function: StaticString = #function,
        line: UInt = #line
    ) {
        _log(
            level: .error, message: message, metadata: metadata,
            file: file, function: function, line: line)
    }
}

#if canImport(OSLog)
    import OSLog

    /// On systems that support it, a log handler that is backed by OSLog.
    public struct OSLogHandler: MLXLogHandler {

        private let logger: os.Logger

        public static func install() {
            MLXLogger.factory = { OSLogHandler(label: $0) }
        }

        public init(label: String) {
            self.logger = Logger(subsystem: "mlx-swift", category: label)
        }

        public func log(
            level: LogLevel,
            message: () -> String, metadata: () -> [String: String]?,
            file: StaticString, function: StaticString, line: UInt
        ) {
            let level: OSLogType =
                switch level {
                case .trace: .debug
                case .debug: .debug
                case .info: .info
                case .warning: .error
                case .error: .error
                }
            if logger.isEnabled(type: level) {
                let message = message()
                logger.log(level: level, "\(message, privacy: .public)")
            }
        }
    }
#endif

/// A logger that logs to `stderr`.
///
/// Callers can set `StderrHandler.logLevel` to filter output by level.
public struct StderrHandler: MLXLogHandler {

    private let label: String

    private static let lock = NSLock()
    #if swift(>=5.10)
        nonisolated(unsafe) private static var _logLevel = LogLevel.info
    #else
        private static var _logLevel = LogLevel.info
    #endif

    /// Get or set the lowest log level.  Default is `.info`.
    public static var logLevel: LogLevel {
        get { lock.withLock { _logLevel } }
        set { lock.withLock { _logLevel = newValue } }
    }

    public static func install() {
        MLXLogger.factory = { StderrHandler(label: $0) }
    }

    public init(label: String) {
        self.label = label
    }

    public func log(
        level: LogLevel, message: @autoclosure () -> String,
        metadata: @autoclosure () -> [String: String]?, file: StaticString, function: StaticString,
        line: UInt
    ) {
        if level >= Self.lock.withLock({ Self._logLevel }) {
            let ts = Date().formatted(date: .numeric, time: .standard)
            let message = "\(ts) [\(level)]: \(message())\n"
            FileHandle.standardError.write(Data(message.utf8))
        }
    }
}
