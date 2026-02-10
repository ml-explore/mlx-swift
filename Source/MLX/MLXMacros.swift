// Copyright © 2026 Apple Inc.

/// Construct an ``MLXArray`` from a nested numeric literal.
///
/// Examples:
///
/// ```swift
/// let a = #mlx([[1, 2, 3], [4, 5, 6]])
/// let b = #mlx([[1, 2, 3], [4, 5, 6]], dtype: .int16)
/// let c = #mlx([[0.1, 0.2], [0.3, 0.4]], dtype: .float16)
/// ```
@freestanding(expression)
public macro mlx(_ literal: Any) -> MLXArray =
    #externalMacro(
        module: "MLXMacrosPlugin", type: "MLXLiteralMacro")

/// Construct an ``MLXArray`` from a nested numeric literal and cast to `dtype`.
@freestanding(expression)
public macro mlx(_ literal: Any, dtype: DType) -> MLXArray =
    #externalMacro(
        module: "MLXMacrosPlugin", type: "MLXLiteralMacro")
