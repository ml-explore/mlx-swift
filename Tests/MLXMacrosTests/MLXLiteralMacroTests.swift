// Copyright © 2026 Apple Inc.

import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

@testable import MLXMacrosPlugin

private let testMacros: [String: Macro.Type] = [
    "MLXArray": MLXLiteralMacro.self,
]

final class MLXLiteralMacroTests: XCTestCase {
    func testExpandsIntegerLiteral() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]])",
            expandedSource: "MLXArray([1, 2, 3, 4], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsFloatLiteral() {
        assertMacroExpansion(
            "#MLXArray([[0.1, 0.2], [0.3, 0.4]])",
            expandedSource: "MLXArray(converting: [0.1, 0.2, 0.3, 0.4], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsBooleanLiteral() {
        assertMacroExpansion(
            "#MLXArray([[true, false], [false, true]])",
            expandedSource: "MLXArray([true, false, false, true], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsWithDtypeCast() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]], dtype: .int16)",
            expandedSource: "MLXArray([Int16(1), Int16(2), Int16(3), Int16(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithInt64Dtype() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]], dtype: .int64)",
            expandedSource: "MLXArray([Int64(1), Int64(2), Int64(3), Int64(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithUInt8Dtype() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]], dtype: .uint8)",
            expandedSource: "MLXArray([UInt8(1), UInt8(2), UInt8(3), UInt8(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsIntegerLiteralWithFloat32Dtype() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]], dtype: .float32)",
            expandedSource: "MLXArray([Float(1), Float(2), Float(3), Float(4)], [2, 2])",
            macros: testMacros
        )
    }

    func testFallsBackToAsTypeForFloat64Dtype() {
        assertMacroExpansion(
            "#MLXArray([[1.0, 2.0], [3.0, 4.0]], dtype: .float64)",
            expandedSource: "MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [2, 2]).asType(.float64)",
            macros: testMacros
        )
    }

    func testExpandsBoolDtypeFromZeroOneLiterals() {
        assertMacroExpansion(
            "#MLXArray([[0, 1], [1, 0]], dtype: .bool)",
            expandedSource: "MLXArray([false, true, true, false], [2, 2])",
            macros: testMacros
        )
    }

    func testFallsBackToAsTypeForDynamicDtypeExpression() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4]], dtype: dtypeValue)",
            expandedSource: "MLXArray([1, 2, 3, 4], [2, 2]).asType(dtypeValue)",
            macros: testMacros
        )
    }

    func testExpandsThreeDimensionalIntegerLiteral() {
        assertMacroExpansion(
            "#MLXArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])",
            expandedSource: "MLXArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])",
            macros: testMacros
        )
    }

    func testExpandsFourDimensionalFloatLiteral() {
        assertMacroExpansion(
            "#MLXArray([[[[0.1, 0.2]], [[0.3, 0.4]]], [[[0.5, 0.6]], [[0.7, 0.8]]]])",
            expandedSource:
                "MLXArray(converting: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 2, 1, 2])",
            macros: testMacros
        )
    }

    func testExpandsMixedIntegerFloatLiteralAsFloat() {
        assertMacroExpansion(
            "#MLXArray([[1, 2.5], [3, 4.5]])",
            expandedSource: "MLXArray(converting: [1, 2.5, 3, 4.5], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsSingleFloatElementAsFloatLiteral() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3, 4.5]])",
            expandedSource: "MLXArray(converting: [1, 2, 3, 4.5], [2, 2])",
            macros: testMacros
        )
    }

    func testExpandsDeepLiteralWithFloat16Dtype() {
        assertMacroExpansion(
            "#MLXArray([[[1, 2], [3, 4]]], dtype: .float16)",
            expandedSource: "MLXArray([1, 2, 3, 4], [1, 2, 2]).asType(.float16)",
            macros: testMacros
        )
    }

    func testExpandsMixedLiteralWithInt8Dtype() {
        assertMacroExpansion(
            "#MLXArray([[1.25, 2], [3.5, 4]], dtype: .int8)",
            expandedSource: "MLXArray(converting: [1.25, 2, 3.5, 4], [2, 2]).asType(.int8)",
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "#MLXArray integer dtype with floating-point literal(s) may truncate values during conversion.",
                    line: 1,
                    column: 13,
                    severity: .warning)
            ],
            macros: testMacros
        )
    }

    func testWarnsOnIntegerDtypeWithFloatLiteral() {
        assertMacroExpansion(
            "#MLXArray([[1, 2.5], [3, 4]], dtype: .int16)",
            expandedSource: "MLXArray(converting: [1, 2.5, 3, 4], [2, 2]).asType(.int16)",
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "#MLXArray integer dtype with floating-point literal(s) may truncate values during conversion.",
                    line: 1,
                    column: 16,
                    severity: .warning)
            ],
            macros: testMacros
        )
    }

    func testRejectsBoolDtypeWithOutOfRangeIntegerLiterals() {
        assertMacroExpansion(
            "#MLXArray([[0, 2], [1, 0]], dtype: .bool)",
            expandedSource: "MLXArray([])",
            diagnostics: [
                DiagnosticSpec(
                    message: "#MLXArray dtype .bool only supports integer literals 0 or 1.",
                    line: 1,
                    column: 16)
            ],
            macros: testMacros
        )
    }

    func testRejectsBoolDtypeWithFloatLiterals() {
        assertMacroExpansion(
            "#MLXArray([[0.0, 1.0]], dtype: .bool)",
            expandedSource: "MLXArray([])",
            diagnostics: [
                DiagnosticSpec(
                    message: "#MLXArray dtype .bool only supports true/false literals or integer 0/1.",
                    line: 1,
                    column: 32)
            ],
            macros: testMacros
        )
    }

    func testRaggedLiteralDiagnostics() {
        assertMacroExpansion(
            "#MLXArray([[1, 2], [3]])",
            expandedSource: "MLXArray([])",
            diagnostics: [
                DiagnosticSpec(
                    message: "#MLXArray does not support ragged nested arrays.", line: 1, column: 11)
            ],
            macros: testMacros
        )
    }
}
