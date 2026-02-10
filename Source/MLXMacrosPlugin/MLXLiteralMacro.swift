// Copyright © 2026 Apple Inc.

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

private enum ScalarKind {
    case int
    case float

    static func merge(_ lhs: ScalarKind, _ rhs: ScalarKind) -> ScalarKind {
        if lhs == .float || rhs == .float {
            return .float
        }
        return .int
    }
}

private struct ParsedLiteral {
    var flat: [ExprSyntax]
    var shape: [Int]
    var kind: ScalarKind
}

private struct MacroError: Error {}

private struct MacroMessage: DiagnosticMessage {
    let message: String
    let diagnosticID: MessageID
    let severity: DiagnosticSeverity

    init(_ message: String, severity: DiagnosticSeverity = .error) {
        self.message = message
        self.severity = severity
        self.diagnosticID = MessageID(domain: "MLXMacros", id: "mlx_literal")
    }
}

private enum KnownDType: String {
    case bool
    case uint8
    case uint16
    case uint32
    case uint64
    case int8
    case int16
    case int32
    case int64
    case float16
    case float32
    case bfloat16
    case complex64
    case float64
}

public struct MLXLiteralMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let args = Array(node.arguments)
        guard let literalArg = args.first else {
            diagnose("#mlx requires a nested numeric array literal.", at: Syntax(node), in: context)
            return "MLXArray([])"
        }

        let dtypeExpr: ExprSyntax?
        if args.count == 1 {
            dtypeExpr = nil
        } else if args.count == 2 {
            guard args[1].label?.text == "dtype" else {
                diagnose(
                    "#mlx second argument must be labeled 'dtype:'.",
                    at: Syntax(args[1]), in: context)
                return "MLXArray([])"
            }
            dtypeExpr = args[1].expression
        } else {
            diagnose(
                "#mlx accepts one literal argument and optional dtype:.", at: Syntax(node),
                in: context)
            return "MLXArray([])"
        }

        let parsed: ParsedLiteral
        do {
            parsed = try parseLiteral(literalArg.expression, context: context)
        } catch {
            return "MLXArray([])"
        }

        let flatSource = parsed.flat.map { $0.description }.joined(separator: ", ")
        let shapeSource = parsed.shape.map(String.init).joined(separator: ", ")
        let baseExpr: ExprSyntax =
            switch parsed.kind {
            case .int:
                "MLXArray([\(raw: flatSource)], [\(raw: shapeSource)])"
            case .float:
                "MLXArray(converting: [\(raw: flatSource)], [\(raw: shapeSource)])"
            }

        if let dtypeExpr {
            if let knownDType = parseKnownDType(dtypeExpr) {
                if isIntegerDType(knownDType), let floatExpr = parsed.flat.first(where: isFloat) {
                    diagnose(
                        "#mlx integer dtype with floating-point literal(s) may truncate values during conversion.",
                        at: Syntax(floatExpr),
                        severity: .warning,
                        in: context)
                }
                if let typedExpr = makeTypedExpression(parsed: parsed, dtype: knownDType) {
                    return typedExpr
                }
            }
            return "\(baseExpr).asType(\(dtypeExpr))"
        } else {
            return baseExpr
        }
    }

    private static func parseKnownDType(_ expr: ExprSyntax) -> KnownDType? {
        guard let member = expr.as(MemberAccessExprSyntax.self) else {
            return nil
        }
        return KnownDType(rawValue: member.declName.baseName.text)
    }

    private static func makeTypedExpression(parsed: ParsedLiteral, dtype: KnownDType) -> ExprSyntax?
    {
        let shapeSource = parsed.shape.map(String.init).joined(separator: ", ")
        let typedFlat: String

        switch dtype {
        case .int8:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "Int8")
        case .int16:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "Int16")
        case .int32:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "Int32")
        case .int64:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "Int64")
        case .uint8:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "UInt8")
        case .uint16:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "UInt16")
        case .uint32:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "UInt32")
        case .uint64:
            guard parsed.kind == .int else { return nil }
            typedFlat = wrap(parsed.flat, with: "UInt64")
        case .float32:
            if parsed.kind == .int {
                typedFlat = wrap(parsed.flat, with: "Float")
            } else {
                typedFlat = wrap(parsed.flat, with: "Float")
            }
        case .bool, .float16, .bfloat16, .complex64, .float64:
            return nil
        }

        return "MLXArray([\(raw: typedFlat)], [\(raw: shapeSource)])"
    }

    private static func wrap(_ values: [ExprSyntax], with typeName: String) -> String {
        values.map { "\(typeName)(\($0))" }.joined(separator: ", ")
    }

    private static func isIntegerDType(_ dtype: KnownDType) -> Bool {
        switch dtype {
        case .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64:
            return true
        default:
            return false
        }
    }

    private static func parseLiteral(
        _ expr: ExprSyntax, context: some MacroExpansionContext
    ) throws -> ParsedLiteral {
        if let arrayExpr = expr.as(ArrayExprSyntax.self) {
            if arrayExpr.elements.isEmpty {
                return ParsedLiteral(flat: [], shape: [0], kind: .int)
            }

            var children: [ParsedLiteral] = []
            children.reserveCapacity(arrayExpr.elements.count)

            for element in arrayExpr.elements {
                children.append(try parseLiteral(element.expression, context: context))
            }

            let firstShape = children[0].shape
            if children.dropFirst().contains(where: { $0.shape != firstShape }) {
                diagnose(
                    "#mlx does not support ragged nested arrays.", at: Syntax(expr), in: context)
                throw MacroError()
            }

            let kind = children.dropFirst().reduce(children[0].kind) {
                ScalarKind.merge($0, $1.kind)
            }

            return ParsedLiteral(
                flat: children.flatMap(\.flat), shape: [children.count] + firstShape, kind: kind)
        }

        if isInteger(expr) {
            return ParsedLiteral(flat: [expr], shape: [], kind: .int)
        }
        if isFloat(expr) {
            return ParsedLiteral(flat: [expr], shape: [], kind: .float)
        }

        diagnose(
            "#mlx only supports integer and floating-point literals.", at: Syntax(expr), in: context
        )
        throw MacroError()
    }

    private static func isInteger(_ expr: ExprSyntax) -> Bool {
        if expr.as(IntegerLiteralExprSyntax.self) != nil {
            return true
        }
        if let prefix = expr.as(PrefixOperatorExprSyntax.self) {
            return isInteger(prefix.expression)
        }
        return false
    }

    private static func isFloat(_ expr: ExprSyntax) -> Bool {
        if expr.as(FloatLiteralExprSyntax.self) != nil {
            return true
        }
        if let prefix = expr.as(PrefixOperatorExprSyntax.self) {
            return isFloat(prefix.expression)
        }
        return false
    }

    private static func diagnose(
        _ message: String,
        at node: Syntax,
        severity: DiagnosticSeverity = .error,
        in context: some MacroExpansionContext
    ) {
        context.diagnose(Diagnostic(node: node, message: MacroMessage(message, severity: severity)))
    }
}
