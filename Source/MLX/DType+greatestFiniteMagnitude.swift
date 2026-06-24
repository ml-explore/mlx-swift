// Copyright © 2026 Apple Inc.

import Foundation
import MLX

extension DType {
    /// Largest finite value of this float dtype.
    ///
    /// ```swift
    /// MLXArray(-scores.dtype.greatestFiniteMagnitude).asType(scores.dtype)
    /// ```
    ///
    public var greatestFiniteMagnitude: Float {
        switch self {
        case .float16: return Float(Float16.greatestFiniteMagnitude)
        case .bfloat16: return Float(bitPattern: 0x7F7F_0000)
        default: return .greatestFiniteMagnitude
        }
    }
}
