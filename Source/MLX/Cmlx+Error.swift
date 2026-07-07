// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// A structured error raised by the MLX backend.
///
/// Replaces the message-only `MLXError.caught(String)`. The `code` lets callers
/// react programmatically — e.g. evict the GPU cache and retry on ``Code/outOfMemory``,
/// or surface a validation message on ``Code/invalidArgument`` — while `message`
/// preserves the original C++ `what()` text (with the originating `file:line`).
public struct MLXError: LocalizedError, Sendable, Equatable, CustomStringConvertible {

    /// Classification of the failure, mirroring `mlx_error_code` in mlx-c.
    public enum Code: Sendable, Equatable {
        /// Shape / dtype / axis mismatch (`std::invalid_argument`). Usually a
        /// programmer error, but recoverable when inputs are externally sourced.
        case invalidArgument
        /// Out-of-range index (`std::out_of_range`).
        case outOfRange
        /// Allocation failure, including Metal `[metal::malloc]` (`std::bad_alloc`).
        /// Typically recoverable: free buffers / shrink the batch and retry.
        case outOfMemory
        /// Load / save / format failure — corrupt safetensors, missing file, bad GGUF.
        case io
        /// Any other `std::exception` / `std::runtime_error`.
        case runtime
        /// A non-`std::exception` throw crossed the boundary (`catch (...)`).
        case unknown

        init(_ raw: mlx_error_code) {
            switch raw {
            case MLX_ERROR_INVALID_ARGUMENT: self = .invalidArgument
            case MLX_ERROR_OUT_OF_RANGE: self = .outOfRange
            case MLX_ERROR_OUT_OF_MEMORY: self = .outOfMemory
            case MLX_ERROR_IO: self = .io
            case MLX_ERROR_RUNTIME: self = .runtime
            default: self = .unknown
            }
        }
    }

    public let code: Code
    public let message: String

    public var errorDescription: String? { description }
    public var description: String { "MLX \(code): \(message)" }
}

/// Consume the calling thread's mlx-c error slot and throw it natively.
///
/// This is the pull side of the exception boundary. mlx-c stores classified
/// error state in thread-local storage *before* returning a non-zero status;
/// here — where a real Swift frame exists — we read that state and `throw`,
/// which a C callback never could. Because the slot is per-thread, an error
/// raised while evaluating on a Metal completion thread or a `DispatchQueue`
/// worker is reported on *that* thread and surfaces through the status code to
/// whoever synchronizes on it, closing the task-local gap in the old design.
///
/// - Parameter status: the `Int32` returned by any `mlx_*` C function.
@inline(__always)
func checkStatus(_ status: Int32, file: StaticString = #fileID, line: UInt = #line) throws {
    guard status != 0 else { return }

    let code = MLXError.Code(mlx_last_error_code())
    let message =
        mlx_last_error_message().map { String(cString: $0) } ?? "unknown MLX error"
    mlx_clear_last_error()

    throw MLXError(code: code, message: message)
}

/// Non-throwing bridge for code paths that are not yet `throws` (e.g. operator
/// overloads). Records the error into the given ``MLXArray`` as a *poison*
/// value so the first use — including the next `try eval()` touching it —
/// rethrows the original error, killing the zombie-value cascade. See
/// ``MLXArray/poison(_:)``.
@inline(__always)
func checkStatus(_ status: Int32, poisoning array: MLXArray) {
    guard status != 0 else { return }

    let code = MLXError.Code(mlx_last_error_code())
    let message =
        mlx_last_error_message().map { String(cString: $0) } ?? "unknown MLX error"
    mlx_clear_last_error()

    array.poison(MLXError(code: code, message: message))
}
