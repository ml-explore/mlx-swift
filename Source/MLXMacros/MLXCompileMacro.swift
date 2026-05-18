// Copyright © 2024-2025 Apple Inc.

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

public struct MLXCompileMacro {}

// MARK: - Return type classification

/// How the function's return type maps to [MLXArray] for compile().
enum ReturnForm {
    case single  // MLXArray
    case tuple(Int)  // (MLXArray, MLXArray, ...) with count
    case array  // [MLXArray]
}

/// Classify a function's return type.
func classifyReturn(_ type: TypeSyntax) -> ReturnForm? {
    let text = type.trimmedDescription
    if text == "MLXArray" {
        return .single
    }
    if text == "[MLXArray]" {
        return .array
    }
    // Check for tuple of MLXArray
    if let tuple = type.as(TupleTypeSyntax.self) {
        let elements = Array(tuple.elements)
        guard !elements.isEmpty else { return nil }
        for elem in elements {
            if elem.type.trimmedDescription != "MLXArray" {
                return nil
            }
        }
        return .tuple(elements.count)
    }
    return nil
}

// MARK: - PeerMacro

extension MLXCompileMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        guard let funcDecl = declaration.as(FunctionDeclSyntax.self) else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.notAFunction))
            return []
        }

        if funcDecl.body != nil {
            context.diagnose(.init(node: Syntax(funcDecl), message: Diagnostic.hasBody))
            return []
        }

        if funcDecl.signature.effectSpecifiers?.asyncSpecifier != nil {
            context.diagnose(.init(node: Syntax(funcDecl), message: Diagnostic.asyncNotSupported))
            return []
        }
        if funcDecl.signature.effectSpecifiers?.throwsClause != nil {
            context.diagnose(.init(node: Syntax(funcDecl), message: Diagnostic.throwsNotSupported))
            return []
        }

        if funcDecl.genericParameterClause != nil {
            context.diagnose(.init(node: Syntax(funcDecl), message: Diagnostic.genericNotSupported))
            return []
        }

        let funcName = funcDecl.name.text
        let params = funcDecl.signature.parameterClause.parameters

        for param in params {
            if param.type.trimmedDescription != "MLXArray" {
                context.diagnose(
                    .init(node: Syntax(param.type), message: Diagnostic.nonMLXArrayParam))
                return []
            }
        }

        guard let returnClause = funcDecl.signature.returnClause else {
            context.diagnose(.init(node: Syntax(funcDecl), message: Diagnostic.unsupportedReturn))
            return []
        }

        guard let returnForm = classifyReturn(returnClause.type) else {
            context.diagnose(
                .init(node: Syntax(returnClause.type), message: Diagnostic.unsupportedReturn))
            return []
        }

        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return []
        }

        let (compileParams, closureExpr) = extractArguments(arguments)

        guard let closureExpr else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return []
        }

        // Build the wrapping closure for compile().
        // Always uses ([MLXArray]) -> [MLXArray] form.
        let paramCount = params.count
        let closureText = closureExpr.trimmedDescription
        let argsList = (0 ..< paramCount).map { "args[\($0)]" }.joined(separator: ", ")

        let wrappedBody: String
        switch returnForm {
        case .single:
            wrappedBody = "[(\(closureText))(\(argsList))]"
        case .tuple(let count):
            let tupleAccess = (0 ..< count).map { "r.\($0)" }.joined(separator: ", ")
            wrappedBody = "{ let r = (\(closureText))(\(argsList)); return [\(tupleAccess)] }()"
        case .array:
            wrappedBody = "(\(closureText))(\(argsList))"
        }

        let compileCall: String
        if compileParams.isEmpty {
            compileCall = "compile { args in \(wrappedBody) }"
        } else {
            compileCall =
                "compile(\(compileParams.joined(separator: ", "))) { args in \(wrappedBody) }"
        }

        let isStatic = funcDecl.modifiers.contains { modifier in
            modifier.name.tokenKind == .keyword(.static)
        }
        let staticPrefix = isStatic ? "static " : ""

        let peerDecl: DeclSyntax =
            """
            \(raw: staticPrefix)private let _mlxc_\(raw: funcName): @Sendable ([MLXArray]) -> [MLXArray] =
                \(raw: compileCall)
            """

        return [peerDecl]
    }
}

// MARK: - BodyMacro

extension MLXCompileMacro: BodyMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingBodyFor declaration: some DeclSyntaxProtocol & WithOptionalCodeBlockSyntax,
        in context: some MacroExpansionContext
    ) throws -> [CodeBlockItemSyntax] {
        guard let funcDecl = declaration.as(FunctionDeclSyntax.self) else {
            return []
        }

        // Skip body generation if validation would fail (peer emits diagnostics)
        guard funcDecl.body == nil,
            funcDecl.signature.effectSpecifiers?.asyncSpecifier == nil,
            funcDecl.signature.effectSpecifiers?.throwsClause == nil,
            funcDecl.genericParameterClause == nil
        else {
            return []
        }

        let params = funcDecl.signature.parameterClause.parameters
        for param in params {
            if param.type.trimmedDescription != "MLXArray" {
                return []
            }
        }

        guard let returnClause = funcDecl.signature.returnClause else {
            return []
        }
        guard let returnForm = classifyReturn(returnClause.type) else {
            return []
        }

        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
            return []
        }
        let (_, closureExpr) = extractArguments(arguments)
        guard closureExpr != nil else {
            return []
        }

        let funcName = funcDecl.name.text
        let paramNames = params.map { ($0.secondName ?? $0.firstName).text }
        let arrayArgs = "[\(paramNames.joined(separator: ", "))]"

        let callExpr: String
        switch returnForm {
        case .single:
            callExpr = "_mlxc_\(funcName)(\(arrayArgs))[0]"
        case .tuple(let count):
            // let r = _mlxc_f([a, b]); return (r[0], r[1])
            let tupleElements = (0 ..< count).map { "r[\($0)]" }.joined(separator: ", ")
            let stmts: CodeBlockItemSyntax =
                """
                let r = _mlxc_\(raw: funcName)(\(raw: arrayArgs))
                return (\(raw: tupleElements))
                """
            return [stmts]
        case .array:
            callExpr = "_mlxc_\(funcName)(\(arrayArgs))"
        }

        let stmt: CodeBlockItemSyntax = "return \(raw: callExpr)"
        return [stmt]
    }
}

// MARK: - ExpressionMacro (freestanding #MLXCompile)

extension MLXCompileMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        // Extract compile params and closure from arguments
        let (compileParams, closureFromArgs) = extractArguments(node.arguments)

        // Closure can be in arguments or as trailing closure
        let closure: ClosureExprSyntax
        if let closureFromArgs {
            closure = closureFromArgs
        } else if let trailing = node.trailingClosure {
            closure = trailing
        } else if !compileParams.isEmpty {
            // No closure literal — pass through to compile() as-is.
            // This handles function references like #MLXCompile(gelu).
            let allArgs = compileParams.joined(separator: ", ")
            return "compile(\(raw: allArgs))"
        } else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return "{ fatalError() }()"
        }

        // Parse closure signature — require explicit types
        guard let signature = closure.signature,
            case .parameterClause(let paramClause) = signature.parameterClause
        else {
            context.diagnose(
                .init(node: Syntax(closure), message: Diagnostic.closureNeedsExplicitTypes))
            return "{ fatalError() }()"
        }

        let params = Array(paramClause.parameters)
        let paramCount = params.count

        guard let returnClause = signature.returnClause else {
            context.diagnose(
                .init(node: Syntax(closure), message: Diagnostic.closureNeedsExplicitTypes))
            return "{ fatalError() }()"
        }

        guard let returnForm = classifyReturn(returnClause.type) else {
            context.diagnose(
                .init(node: Syntax(returnClause.type), message: Diagnostic.unsupportedReturn))
            return "{ fatalError() }()"
        }

        // Build the inner compile wrapper (same wrapping logic as PeerMacro)
        let closureText = closure.trimmedDescription
        let argsList = (0 ..< paramCount).map { "args[\($0)]" }.joined(separator: ", ")

        let wrappedBody: String
        switch returnForm {
        case .single:
            wrappedBody = "[(\(closureText))(\(argsList))]"
        case .tuple(let count):
            let tupleAccess = (0 ..< count).map { "r.\($0)" }.joined(separator: ", ")
            wrappedBody = "{ let r = (\(closureText))(\(argsList)); return [\(tupleAccess)] }()"
        case .array:
            wrappedBody = "(\(closureText))(\(argsList))"
        }

        let compileCall: String
        if compileParams.isEmpty {
            compileCall = "compile { args in \(wrappedBody) }"
        } else {
            compileCall =
                "compile(\(compileParams.joined(separator: ", "))) { args in \(wrappedBody) }"
        }

        // Build the outer wrapper that returns a function with the user's signature
        let paramNames = params.map { $0.firstName.text }
        let paramDecls = params.map {
            "\($0.firstName.text): \($0.type?.trimmedDescription ?? "MLXArray")"
        }.joined(separator: ", ")
        let argPack = "[\(paramNames.joined(separator: ", "))]"
        let returnTypeText = returnClause.type.trimmedDescription

        let outerBody: String
        switch returnForm {
        case .single:
            outerBody = "_c(\(argPack))[0]"
        case .tuple(let count):
            let tupleUnpack = (0 ..< count).map { "r[\($0)]" }.joined(separator: ", ")
            outerBody = "{ let r = _c(\(argPack)); return (\(tupleUnpack)) }()"
        case .array:
            outerBody = "_c(\(argPack))"
        }

        let expr: ExprSyntax = """
            {
                let _c: @Sendable ([MLXArray]) -> [MLXArray] = \(raw: compileCall)
                return { (\(raw: paramDecls)) -> \(raw: returnTypeText) in \(raw: outerBody) }
            }()
            """

        return expr
    }
}

// MARK: - Argument extraction

extension MLXCompileMacro {
    static func extractArguments(_ arguments: LabeledExprListSyntax) -> (
        [String], ClosureExprSyntax?
    ) {
        var compileParams: [String] = []
        var closureExpr: ClosureExprSyntax?

        let argList = Array(arguments)

        for (index, arg) in argList.enumerated() {
            let isLast = index == argList.count - 1

            if isLast, arg.label == nil,
                let closure = arg.expression.as(ClosureExprSyntax.self)
            {
                closureExpr = closure
            } else {
                let label = arg.label?.text
                let expr = arg.expression.trimmedDescription
                if let label {
                    compileParams.append("\(label): \(expr)")
                } else {
                    compileParams.append(expr)
                }
            }
        }

        return (compileParams, closureExpr)
    }
}

// MARK: - Diagnostics

extension MLXCompileMacro {
    enum Diagnostic: String, DiagnosticMessage {
        case notAFunction
        case hasBody
        case missingClosure
        case closureNeedsExplicitTypes
        case nonMLXArrayParam
        case unsupportedReturn
        case asyncNotSupported
        case throwsNotSupported
        case genericNotSupported

        var severity: DiagnosticSeverity { .error }

        var message: String {
            switch self {
            case .notAFunction:
                return "@MLXCompile can only be applied to functions"
            case .hasBody:
                return "@MLXCompile requires a function without a body"
            case .missingClosure:
                return "@MLXCompile requires an implementation closure as the last argument"
            case .closureNeedsExplicitTypes:
                return
                    "#MLXCompile requires a closure with explicit parameter types and return type"
            case .nonMLXArrayParam:
                return "@MLXCompile requires all parameters to be of type MLXArray"
            case .unsupportedReturn:
                return
                    "@MLXCompile requires return type to be MLXArray, (MLXArray, ...), or [MLXArray]"
            case .asyncNotSupported:
                return "@MLXCompile does not support async functions"
            case .throwsNotSupported:
                return "@MLXCompile does not support throwing functions"
            case .genericNotSupported:
                return "@MLXCompile does not support generic functions"
            }
        }

        var diagnosticID: MessageID {
            MessageID(domain: "MLXMacros", id: rawValue)
        }
    }
}
