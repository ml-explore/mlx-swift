import Cmlx
import Foundation

/// Sets the error handler. The default error handler will simply print out the error, then exit.
/// - Parameters:
///   - handler: An error handler. Pass nil to reset to the default error handler. Pass
///   ``fatalErrorHandler`` to make the error handler call `fatalError` for improved Xcode debugging.
@available(*, deprecated, message: "please use withErrorHandler() or withError()")
public func setErrorHandler(
    _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
    data: UnsafeMutableRawPointer? = nil,
    dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
) {
    errorHandler.setGlobalHandler(handler, data: data, dtor: dtor)
}

/// An error handler that calls `fatalError`.
@available(*, deprecated, message: "please use withErrorHandler() or withError()")
public let fatalErrorHandler:
    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { message, _ in
        fatalError(message.map { String(cString: $0) } ?? "")
    }

/// Evaluate the block with a scoped MLX error handler.
///
/// For example this will print a message instead of exiting the process:
///
/// ```swift
/// func printHandler(_ message: String) {
///     print("Caught error: \(message)")
/// }
///
/// withErrorHandler(printHandler) {
///     let a = MLXArray(0 ..< 10, [2, 5])
///     let b = MLXArray(0 ..< 15, [3, 5])
///
///     // this will trigger a broadcast error
///     return a + b
/// }
/// ```
///
/// Note: using the ``MLXArray`` produced after an error will likely result in additional errors.
/// See ``withError(_:)-6g4wn`` for examples that convert the errors into Swift `throws`.
///
/// - Parameters:
///   - handler: the scoped handler
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-6g4wn``
/// - ``withError(_:)-2wfiu``
public func withErrorHandler<R>(
    _ handler: @escaping @Sendable (String) -> Void, _ body: () throws -> R
) rethrows -> R {
    try errorHandler.withErrorHandler(handler, body)
}

/// Evaluate the block with a scoped MLX error handler (async).
///
/// For example this will print a message instead of exiting the process:
///
/// ```swift
/// func printHandler(_ message: String) {
///     print("Caught error: \(message)")
/// }
///
/// withErrorHandler(printHandler) {
///     let a = MLXArray(0 ..< 10, [2, 5])
///     let b = MLXArray(0 ..< 15, [3, 5])
///
///     // this will trigger a broadcast error
///     return a + b
/// }
/// ```
///
/// Note: using the ``MLXArray`` produced after an error will likely result in additional errors.
/// See ``withError(_:)-4tvdu`` for examples that convert the errors into Swift `throws`.
///
/// - Parameters:
///   - handler: the scoped handler
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-4tvdu``
/// - ``withError(_:)-7f0hv``
public func withErrorHandler<R>(
    _ handler: @escaping @Sendable (String) -> Void, _ body: () async throws -> R
) async rethrows -> R {
    try await errorHandler.withErrorHandler(handler, body)
}

/// Evaluate the block and convert MLX errors into Swift `throws`
/// with explicit visibility of the error.
///
/// For example this will throw a Swift `Error`:
///
/// ```swift
/// try withError { error in
///     let a = MLXArray(0 ..< 10, [2, 5])
///     let b = MLXArray(0 ..< 15, [3, 5])
///
///     // this will trigger a broadcast error
///     let x = a + b
///
///     // explicitly check for errors or return from the block
///     try error.check()
///     return x
/// }
/// ```
///
/// Note: the `error` argument can be omitted to just check for errors on block exit.
/// Nested calls to `withError` produce scoped error collection.
///
/// - Parameters:
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-6g4wn``
/// - ``withError(_:)-7f0hv``
public func withError<R>(_ body: (ErrorBox) throws -> R) throws -> R {
    try errorHandler.withError(body)
}

/// Evaluate the block and convert MLX errors into Swift `throws` on
/// block exit.
///
/// For example this will throw a Swift `Error`:
///
/// ```swift
/// try withError {
///     let a = MLXArray(0 ..< 10, [2, 5])
///     let b = MLXArray(0 ..< 15, [3, 5])
///
///     // this will trigger a broadcast error
///     return a + b
/// }
/// ```
///
/// Nested calls to `withError` produce scoped error collection.
///
/// - Parameters:
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-2wfiu``
/// - ``withError(_:)-4tvdu``
public func withError<R>(_ body: () throws -> R) throws -> R {
    try errorHandler.withError({ _ in try body() })
}

/// Evaluate the block and convert MLX errors into Swift `throws`
/// with explicit visibility of the error (async).
///
/// For example this will throw a Swift `Error`:
///
/// ```swift
/// try await withError { error in
///     let t = Task {
///         let a = MLXArray(0 ..< 10, [2, 5])
///         let b = MLXArray(0 ..< 15, [3, 5])
///
///         // this will trigger a broadcast error
///         let x = a + b
///
///         // explicitly check for errors or return from the block
///         try error.check()
///         return x
///     }
///     try await t.value
/// }
/// ```
///
/// Note: the `error` argument can be omitted to just check for errors on block exit.
/// Nested calls to `withError` produce scoped error collection.
///
/// - Parameters:
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-2wfiu``
/// - ``withError(_:)-4tvdu``
public func withError<R>(_ body: (ErrorBox) async throws -> R) async throws -> R {
    try await errorHandler.withError(body)
}

/// Evaluate the block and convert MLX errors into Swift `throws` on
/// block exit (async).
///
/// For example this will throw a Swift `Error`:
///
/// ```swift
/// try await withError {
///     let t = Task {
///         let a = MLXArray(0 ..< 10, [2, 5])
///         let b = MLXArray(0 ..< 15, [3, 5])
///
///         // this will trigger a broadcast error
///         return a + b
///     }
///     await t.value
/// }
/// ```
///
/// Nested calls to `withError` produce scoped error collection.
///
/// - Parameters:
///   - body: the code where the handler is to be active
///
/// ### See Also
/// - ``withError(_:)-6g4wn``
/// - ``withError(_:)-7f0hv``
public func withError<R>(_ body: () async throws -> R) async throws -> R {
    try await errorHandler.withError({ _ in try await body() })
}

/// Error type for caught errors during ``withError(_:)-6g4wn``.
public enum MLXError: LocalizedError, Sendable, Equatable {
    case caught(String)

    public var errorDescription: String? {
        switch self {
        case .caught(let message): "MLX Error: \(message)"
        }
    }
}

/// Boxed error type usable with ``withError(_:)-2wfiu``.
///
/// For example:
///
/// ```swift
/// try withError { error in
///     let a = MLXArray(0 ..< 10, [2, 5])
///     let b = MLXArray(0 ..< 15, [3, 5])
///
///     // no errors yet
///     try error.check()
///
///     // this will trigger a broadcast error
///     let x = a + b
///
///     // explicitly check for errors or return from the block
///     try error.check()
///     return x
/// }
/// ```
///
/// In some cases it may be more convenient to use the ``withError(_:)-6g4wn`` form
/// that doesn't expose this value -- any error will be thrown when the block exits.
public final class ErrorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _firstError: Error?

    /// The first error encountered, if any.
    public var firstError: Error? {
        get {
            lock.withLock { _firstError }
        }
        set {
            lock.withLock {
                if _firstError == nil {
                    _firstError = newValue
                }
            }
        }
    }

    /// Throw the ``firstError`` if set, otherwise do nothing.
    public func check() throws {
        if let _firstError {
            throw _firstError
        }
    }
}

/// Internal hook for the MLX error handling -- this sets the `mlx_set_error_handler` once to the
/// `errorHandlerTrampoline` which forwards it to the singleton `ErrorHandler`.
///
/// This will be called once (and only once) when there is a call to `withError`.
private let errorHandler: ErrorHandler = {
    mlx_set_error_handler(errorHandlerTrampoline(message:data:), nil, nil)
    return ErrorHandler()
}()

/// Ensure that the error handler is installed.
func initError() {
    _ = errorHandler
}

/// Forward the error to the `ErrorHandler` singleton.  See `errorHandler` (above) for how this is
/// installed.
private func errorHandlerTrampoline(message: UnsafePointer<CChar>?, data: UnsafeMutableRawPointer?)
{
    errorHandler.dispatch(message.map { String(cString: $0) } ?? "")
}

/// Thread safe and task local implementation of error handling.
private final class ErrorHandler: @unchecked Sendable {

    /// task local error handler stack, if any
    @TaskLocal static var errorHandler: [@Sendable (String) -> Void] = []

    /// the global handler, if any -- this is called if there is no task local error handler
    let lock = NSLock()
    var globalHandler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)? =
        nil
    var globalData: UnsafeMutableRawPointer? = nil
    var globalDtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil

    init() {
    }

    deinit {
        if let globalData = self.globalData, let globalDtor = self.globalDtor {
            globalDtor(globalData)
        }
    }

    func setGlobalHandler(
        _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
        data: UnsafeMutableRawPointer? = nil,
        dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
    ) {
        lock.withLock {
            if let globalData = self.globalData, let globalDtor = self.globalDtor {
                globalDtor(globalData)
            }
            globalHandler = handler
            globalData = data
            globalDtor = dtor
        }
    }

    /// entry point when an error is encountered in the C++ MLX layer
    func dispatch(_ message: String) {
        if let handler = Self.errorHandler.last {
            handler(message)
        } else {
            lock.withLock {
                if let globalHandler {
                    globalHandler(message, globalData)
                } else {
                    fatalError(message)
                }
            }
        }
    }

    public func withErrorHandler<R>(
        _ handler: @escaping @Sendable (String) -> Void, _ body: () throws -> R
    ) rethrows -> R {
        try ErrorHandler.$errorHandler.withValue(ErrorHandler.errorHandler + [handler]) {
            try body()
        }
    }

    public func withErrorHandler<R>(
        _ handler: @escaping @Sendable (String) -> Void, _ body: () async throws -> R
    ) async rethrows -> R {
        try await ErrorHandler.$errorHandler.withValue(ErrorHandler.errorHandler + [handler]) {
            try await body()
        }
    }

    public func withError<R>(_ body: (ErrorBox) throws -> R) throws -> R {
        let errorBox = ErrorBox()

        @Sendable
        func errorHandler(_ message: String) {
            errorBox.firstError = MLXError.caught(message)
        }

        return try withErrorHandler(errorHandler) {
            let r = try body(errorBox)
            try errorBox.check()
            return r
        }
    }

    public func withError<R>(_ body: (ErrorBox) async throws -> R) async throws -> R {
        let errorBox = ErrorBox()

        @Sendable
        func errorHandler(_ message: String) {
            errorBox.firstError = MLXError.caught(message)
        }

        return try await withErrorHandler(errorHandler) {
            let r = try await body(errorBox)
            try errorBox.check()
            return r
        }
    }
}
