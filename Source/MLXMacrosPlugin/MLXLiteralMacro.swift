// Copyright © 2026 Apple Inc.

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

private enum ScalarKind {
    case bool
    case int
    case float

    static func merge(_ lhs: ScalarKind, _ rhs: ScalarKind) -> ScalarKind? {
        switch (lhs, rhs) {
        case (.bool, .bool):
            return .bool
        case (.int, .int):
            return .int
        case (.float, .float):
            return .float
        case (.int, .float), (.float, .int):
            return .float
        case (.bool, .int), (.int, .bool), (.bool, .float), (.float, .bool):
            return nil
        }
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
            diagnose("#MLXArray requires a nested numeric array literal.", at: Syntax(node), in: context)
            return "MLXArray([])"
        }

        let dtypeExpr: ExprSyntax?
        if args.count == 1 {
            dtypeExpr = nil
        } else if args.count == 2 {
            guard args[1].label?.text == "dtype" else {
                diagnose(
                    "#MLXArray second argument must be labeled 'dtype:'.",
                    at: Syntax(args[1]), in: context)
                return "MLXArray([])"
            }
            dtypeExpr = args[1].expression
        } else {
            diagnose(
                "#MLXArray accepts one literal argument and optional dtype:.", at: Syntax(node),
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
        // Default lowering path:
        // - integer-only literals use MLXArray([Int...], shape)
        // - any float literal promotes the whole literal to converting:[Double...]
        let baseExpr: ExprSyntax =
            switch parsed.kind {
            case .bool:
                "MLXArray([\(raw: flatSource)], [\(raw: shapeSource)])"
            case .int:
                "MLXArray([\(raw: flatSource)], [\(raw: shapeSource)])"
            case .float:
                "MLXArray(converting: [\(raw: flatSource)], [\(raw: shapeSource)])"
            }

        if let dtypeExpr {
            if let knownDType = parseKnownDType(dtypeExpr) {
                if knownDType == .bool {
                    switch parsed.kind {
                    case .bool:
                        return baseExpr
                    case .int:
                        var boolValues: [String] = []
                        boolValues.reserveCapacity(parsed.flat.count)
                        for element in parsed.flat {
                            guard let value = integerLiteralValue(element), value == 0 || value == 1
                            else {
                                diagnose(
                                    "#MLXArray dtype .bool only supports integer literals 0 or 1.",
                                    at: Syntax(element),
                                    in: context)
                                return "MLXArray([])"
                            }
                            boolValues.append(value == 1 ? "true" : "false")
                        }
                        let boolSource = boolValues.joined(separator: ", ")
                        return "MLXArray([\(raw: boolSource)], [\(raw: shapeSource)])"
                    case .float:
                        diagnose(
                            "#MLXArray dtype .bool only supports true/false literals or integer 0/1.",
                            at: Syntax(dtypeExpr),
                            in: context)
                        return "MLXArray([])"
                    }
                }
                // Keep explicit integer dtypes permissive but visible:
                // if callers write float literals with an integer dtype, emit a warning
                // since runtime conversion may truncate.
                if isIntegerDType(knownDType), let floatExpr = parsed.flat.first(where: isFloat) {
                    diagnose(
                        "#MLXArray integer dtype with floating-point literal(s) may truncate values during conversion.",
                        at: Syntax(floatExpr),
                        severity: .warning,
                        in: context)
                }
                // Fast path for dtypes we can materialize directly as Swift literals.
                // This avoids emitting a trailing `.asType(...)` cast op.
                if let typedExpr = makeTypedExpression(parsed: parsed, dtype: knownDType) {
                    return typedExpr
                }
            }
            // Fallback for dynamic dtype expressions and dtypes that do not map cleanly
            // to a concrete Swift literal representation.
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
        case .bool:
            return nil
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
            // Float32 has a stable, direct Swift literal representation.
            // Emit typed elements instead of base+cast for lower graph overhead.
            typedFlat = wrap(parsed.flat, with: "Float")
        case .float16, .bfloat16, .complex64, .float64:
            // These currently rely on base+cast to keep expansion predictable
            // across targets and avoid lossy/ambiguous literal synthesis.
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
                // Keep empty arrays legal and representable at compile time.
                return ParsedLiteral(flat: [], shape: [0], kind: .int)
            }

            var children: [ParsedLiteral] = []
            children.reserveCapacity(arrayExpr.elements.count)

            for element in arrayExpr.elements {
                children.append(try parseLiteral(element.expression, context: context))
            }

            let firstShape = children[0].shape
            if children.dropFirst().contains(where: { $0.shape != firstShape }) {
                // MLXArray construction here assumes rectangular nested literals.
                // Ragged arrays are rejected early with a macro diagnostic.
                diagnose(
                    "#MLXArray does not support ragged nested arrays.", at: Syntax(expr), in: context)
                throw MacroError()
            }

            guard
                let kind = children.dropFirst().reduce(
                    Optional(children[0].kind),
                    {
                        partial, next in
                        guard let partial else { return nil }
                        return ScalarKind.merge(partial, next.kind)
                    })
            else {
                diagnose(
                    "#MLXArray does not support mixing boolean and numeric literals in the same array.",
                    at: Syntax(expr),
                    in: context)
                throw MacroError()
            }

            return ParsedLiteral(
                flat: children.flatMap(\.flat), shape: [children.count] + firstShape, kind: kind)
        }

        if isBool(expr) {
            return ParsedLiteral(flat: [expr], shape: [], kind: .bool)
        }
        if isInteger(expr) {
            return ParsedLiteral(flat: [expr], shape: [], kind: .int)
        }
        if isFloat(expr) {
            return ParsedLiteral(flat: [expr], shape: [], kind: .float)
        }

        diagnose(
            "#MLXArray only supports boolean, integer, and floating-point literals.", at: Syntax(expr),
            in: context
        )
        throw MacroError()
    }

    private static func isBool(_ expr: ExprSyntax) -> Bool {
        expr.as(BooleanLiteralExprSyntax.self) != nil
    }

    private static func isInteger(_ expr: ExprSyntax) -> Bool {
        if expr.as(IntegerLiteralExprSyntax.self) != nil {
            return true
        }
        if let prefix = expr.as(PrefixOperatorExprSyntax.self) {
            // Accept signed integer literals like -3 / +7.
            return isInteger(prefix.expression)
        }
        return false
    }

    private static func isFloat(_ expr: ExprSyntax) -> Bool {
        if expr.as(FloatLiteralExprSyntax.self) != nil {
            return true
        }
        if let prefix = expr.as(PrefixOperatorExprSyntax.self) {
            // Accept signed float literals like -3.5 / +1.0e-3.
            return isFloat(prefix.expression)
        }
        return false
    }

    private static func integerLiteralValue(_ expr: ExprSyntax) -> Int? {
        if let literal = expr.as(IntegerLiteralExprSyntax.self) {
            return parseIntegerToken(literal.literal.text)
        }
        if let prefix = expr.as(PrefixOperatorExprSyntax.self) {
            guard let value = integerLiteralValue(prefix.expression) else { return nil }
            switch prefix.operator.text {
            case "+":
                return value
            case "-":
                return -value
            default:
                return nil
            }
        }
        return nil
    }

    private static func parseIntegerToken(_ token: String) -> Int? {
        let text = String(token.filter { $0 != "_" })
        if text.hasPrefix("0x") || text.hasPrefix("0X") {
            return Int(text.dropFirst(2), radix: 16)
        }
        if text.hasPrefix("0b") || text.hasPrefix("0B") {
            return Int(text.dropFirst(2), radix: 2)
        }
        if text.hasPrefix("0o") || text.hasPrefix("0O") {
            return Int(text.dropFirst(2), radix: 8)
        }
        return Int(text)
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
