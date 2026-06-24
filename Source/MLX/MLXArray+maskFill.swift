// Copyright © 2026 Apple Inc.

import Foundation
import MLX

extension MLXArray {
    /// Scalar masked-score fill (`-finfo(dtype).max`) constructed directly in
    /// `dtype`, so masked positions vanish under softmax — no `asType` to forget.
    public static func maskFill(for dtype: DType) -> MLXArray {
        MLXArray(-dtype.greatestFiniteMagnitude, dtype: dtype)
    }
}
