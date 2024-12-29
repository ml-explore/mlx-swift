import Cmlx
import Foundation

/// Sets the error handler. The default error handler will simply print out the error, then exit.
/// - Parameters:
///   - handler: An error handler. Pass nil to reset to the default error handler. Pass
///   ``fatalErrorHandler`` to make the error handler call `fatalError` for improved Xcode debugging.
public func setErrorHandler(
    _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?,
    data: UnsafeMutableRawPointer? = nil,
    dtor: (@convention(c) (UnsafeMutableRawPointer?) -> Void)? = nil
) {
    mlx_set_error_handler(handler, data, dtor)
}

/// An error handler that calls `fatalError`.
public let fatalErrorHandler:
    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { message, _ in
        fatalError(message.map { String(cString: $0) } ?? "")
    }
