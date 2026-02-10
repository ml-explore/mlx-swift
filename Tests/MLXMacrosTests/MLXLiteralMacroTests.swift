// Copyright © 2026 Apple Inc.

import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

@testable import MLXMacrosPlugin

private let testMacros: [String: Macro.Type] = [
    "mlx": MLXLiteralMacro.self
]

final class MLXLiteralMacroTests: XCTestCase {
    func testExpandsIntegerLiteral() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]])",
            expandedSource: "MLXArray([1, 2, 3, 4], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsFloatLiteral() {
        assertMacroExpansion(
            "#mlx([[0.1, 0.2], [0.3, 0.4]])",
            expandedSource: "MLXArray(converting: [0.1, 0.2, 0.3, 0.4], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsWithDtypeCast() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]], dtype: .int16)",
            expandedSource: "MLXArray([Int16(1), Int16(2), Int16(3), Int16(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithInt64Dtype() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]], dtype: .int64)",
            expandedSource: "MLXArray([Int64(1), Int64(2), Int64(3), Int64(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithUInt8Dtype() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]], dtype: .uint8)",
            expandedSource: "MLXArray([UInt8(1), UInt8(2), UInt8(3), UInt8(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithFloat32Dtype() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]], dtype: .float32)",
            expandedSource: "MLXArray([Float(1), Float(2), Float(3), Float(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testFallsBackToAsTypeForFloat64Dtype() {
        assertMacroExpansion(
            "#mlx([[1.0, 2.0], [3.0, 4.0]], dtype: .float64)",
            expandedSource: "MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [2, 2]).asType(.float64)",
            macros: testMacros
        )
    }

    func testFallsBackToAsTypeForDynamicDtypeExpression() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4]], dtype: dtypeValue)",
            expandedSource: "MLXArray([1, 2, 3, 4], [2, 2]).asType(dtypeValue)",
            macros: testMacros
        )
    }

    func testExpandsThreeDimensionalIntegerLiteral() {
        assertMacroExpansion(
            "#mlx([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])",
            expandedSource: "MLXArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])",
            macros: testMacros
        )
    }

    func testExpandsFourDimensionalFloatLiteral() {
        assertMacroExpansion(
            "#mlx([[[[0.1, 0.2]], [[0.3, 0.4]]], [[[0.5, 0.6]], [[0.7, 0.8]]]])",
            expandedSource:
                "MLXArray(converting: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 2, 1, 2])",
            macros: testMacros
        )
    }

    func testExpandsMixedIntegerFloatLiteralAsFloat() {
        assertMacroExpansion(
            "#mlx([[1, 2.5], [3, 4.5]])",
            expandedSource: "MLXArray(converting: [1, 2.5, 3, 4.5], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsSingleFloatElementAsFloatLiteral() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3, 4.5]])",
            expandedSource: "MLXArray(converting: [1, 2, 3, 4.5], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsDeepLiteralWithFloat16Dtype() {
        assertMacroExpansion(
            "#mlx([[[1, 2], [3, 4]]], dtype: .float16)",
            expandedSource: "MLXArray([1, 2, 3, 4], [1, 2, 2]).asType(.float16)",
            macros: testMacros
        )
    }

    func testExpandsMixedLiteralWithInt8Dtype() {
        assertMacroExpansion(
            "#mlx([[1.25, 2], [3.5, 4]], dtype: .int8)",
            expandedSource: "MLXArray(converting: [1.25, 2, 3.5, 4], [2, 2]).asType(.int8)",
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "#mlx integer dtype with floating-point literal(s) may truncate values during conversion.",
                    line: 1,
                    column: 8,
                    severity: .warning)
            ],
            macros: testMacros
        )
    }

    func testWarnsOnIntegerDtypeWithFloatLiteral() {
        assertMacroExpansion(
            "#mlx([[1, 2.5], [3, 4]], dtype: .int16)",
            expandedSource: "MLXArray(converting: [1, 2.5, 3, 4], [2, 2]).asType(.int16)",
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "#mlx integer dtype with floating-point literal(s) may truncate values during conversion.",
                    line: 1,
                    column: 11,
                    severity: .warning)
            ],
            macros: testMacros
        )
    }

    func testRaggedLiteralDiagnostics() {
        assertMacroExpansion(
            "#mlx([[1, 2], [3]])",
            expandedSource: "MLXArray([])",
            diagnostics: [
                DiagnosticSpec(
                    message: "#mlx does not support ragged nested arrays.", line: 1, column: 6)
            ],
            macros: testMacros
        )
    }
}
