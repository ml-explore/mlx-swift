// Copyright © 2024-2025 Apple Inc.

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

public struct MLXValueAndGradMacro {}

// MARK: - Return type classification for valueAndGrad

/// How the loss closure's return type maps to [MLXArray].
private enum VGReturnForm {
    case single  // MLXArray
    case array  // [MLXArray]
}

private func classifyVGReturn(_ type: TypeSyntax) -> VGReturnForm? {
    let text = type.trimmedDescription
    if text == "MLXArray" {
        return .single
    }
    if text == "[MLXArray]" {
        return .array
    }
    return nil
}

// MARK: - Argument extraction

extension MLXValueAndGradMacro {
    /// Extract the `model:` expression and trailing closure from macro arguments.
    static func extractArguments(_ arguments: LabeledExprListSyntax) -> (
        modelExpr: ExprSyntax?, ClosureExprSyntax?
    ) {
        var modelExpr: ExprSyntax?
        var closureExpr: ClosureExprSyntax?

        let argList = Array(arguments)

        for (index, arg) in argList.enumerated() {
            let isLast = index == argList.count - 1

            if arg.label?.text == "model" {
                modelExpr = arg.expression
            } else if isLast, arg.label == nil,
                let closure = arg.expression.as(ClosureExprSyntax.self)
            {
                closureExpr = closure
            }
        }

        return (modelExpr, closureExpr)
    }
}

// MARK: - Closure parameter parsing

private struct ParsedClosure {
    let modelName: String
    let modelType: String
    let arrayParams: [(name: String, type: String)]  // remaining params after model
    let returnForm: VGReturnForm
    let returnTypeText: String
    let closureText: String
}

extension MLXValueAndGradMacro {
    fileprivate static func parseClosure(
        _ closure: ClosureExprSyntax,
        in context: some MacroExpansionContext
    ) -> ParsedClosure? {
        guard let signature = closure.signature,
            case .parameterClause(let paramClause) = signature.parameterClause
        else {
            context.diagnose(
                .init(node: Syntax(closure), message: Diagnostic.closureNeedsExplicitTypes))
            return nil
        }

        let params = Array(paramClause.parameters)
        guard params.count >= 1 else {
            context.diagnose(
                .init(node: Syntax(closure), message: Diagnostic.closureNeedsExplicitTypes))
            return nil
        }

        guard let returnClause = signature.returnClause else {
            context.diagnose(
                .init(node: Syntax(closure), message: Diagnostic.closureNeedsExplicitTypes))
            return nil
        }

        guard let returnForm = classifyVGReturn(returnClause.type) else {
            context.diagnose(
                .init(node: Syntax(returnClause.type), message: Diagnostic.unsupportedReturn))
            return nil
        }

        // First param is the model
        let modelParam = params[0]
        guard let modelType = modelParam.type?.trimmedDescription else {
            context.diagnose(
                .init(node: Syntax(modelParam), message: Diagnostic.closureNeedsExplicitTypes))
            return nil
        }
        let modelName = modelParam.firstName.text

        // Remaining params must be MLXArray
        var arrayParams: [(name: String, type: String)] = []
        for param in params.dropFirst() {
            let typeText = param.type?.trimmedDescription ?? ""
            if typeText != "MLXArray" {
                context.diagnose(
                    .init(node: Syntax(param), message: Diagnostic.nonMLXArrayParam))
                return nil
            }
            arrayParams.append((name: param.firstName.text, type: typeText))
        }

        return ParsedClosure(
            modelName: modelName,
            modelType: modelType,
            arrayParams: arrayParams,
            returnForm: returnForm,
            returnTypeText: returnClause.type.trimmedDescription,
            closureText: closure.trimmedDescription
        )
    }
}

// MARK: - ExpressionMacro (freestanding #MLXValueAndGrad)

extension MLXValueAndGradMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let (modelExpr, closureFromArgs) = extractArguments(node.arguments)

        guard let modelExpr else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingModel))
            return "{ fatalError() }()"
        }

        let closure: ClosureExprSyntax
        if let closureFromArgs {
            closure = closureFromArgs
        } else if let trailing = node.trailingClosure {
            closure = trailing
        } else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return "{ fatalError() }()"
        }

        guard let parsed = parseClosure(closure, in: context) else {
            return "{ fatalError() }()"
        }

        let modelExprText = modelExpr.trimmedDescription
        let arrayCount = parsed.arrayParams.count

        // Inner: valueAndGrad(model: m) { _model, _arrays -> [MLXArray] in ... }
        let argsUnpack = (0 ..< arrayCount).map { "_arrays[\($0)]" }.joined(separator: ", ")
        let closureCall =
            "(\(parsed.closureText))(_model\(arrayCount > 0 ? ", " : "")\(argsUnpack))"

        let innerBody: String
        switch parsed.returnForm {
        case .single:
            innerBody = "[\(closureCall)]"
        case .array:
            innerBody = closureCall
        }

        // Outer function signature
        let outerParams =
            "\(parsed.modelName): \(parsed.modelType)"
            + (parsed.arrayParams.isEmpty
                ? ""
                : ", "
                    + parsed.arrayParams.map { "\($0.name): \($0.type)" }.joined(separator: ", "))
        let arrayPack =
            "[\(parsed.arrayParams.map { $0.name }.joined(separator: ", "))]"

        let outerReturnType: String
        let outerBody: String
        switch parsed.returnForm {
        case .single:
            outerReturnType = "(MLXArray, ModuleParameters)"
            outerBody =
                "let (_v, _g) = _vg(\(parsed.modelName), \(arrayPack))\nreturn (_v[0], _g)"
        case .array:
            outerReturnType = "([MLXArray], ModuleParameters)"
            outerBody = "_vg(\(parsed.modelName), \(arrayPack))"
        }

        let expr: ExprSyntax = """
            {
                let _vg = valueAndGrad(model: \(raw: modelExprText)) { _model, _arrays -> [MLXArray] in
                    \(raw: innerBody)
                }
                return { (\(raw: outerParams)) -> \(raw: outerReturnType) in
                    \(raw: outerBody)
                }
            }()
            """

        return expr
    }
}

// MARK: - PeerMacro

extension MLXValueAndGradMacro: PeerMacro {
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
            context.diagnose(
                .init(node: Syntax(funcDecl), message: Diagnostic.genericNotSupported))
            return []
        }

        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return []
        }

        let (modelExpr, closureExpr) = extractArguments(arguments)

        guard let modelExpr else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingModel))
            return []
        }

        guard let closureExpr else {
            context.diagnose(.init(node: Syntax(node), message: Diagnostic.missingClosure))
            return []
        }

        guard let parsed = parseClosure(closureExpr, in: context) else {
            return []
        }

        // Validate function params match closure
        let params = funcDecl.signature.parameterClause.parameters
        let paramList = Array(params)

        // First param should match model type
        guard paramList.count >= 1 else {
            context.diagnose(
                .init(node: Syntax(funcDecl), message: Diagnostic.closureNeedsExplicitTypes))
            return []
        }

        // Remaining params should be MLXArray
        for param in paramList.dropFirst() {
            if param.type.trimmedDescription != "MLXArray" {
                context.diagnose(
                    .init(node: Syntax(param.type), message: Diagnostic.nonMLXArrayParam))
                return []
            }
        }

        let funcName = funcDecl.name.text
        let modelExprText = modelExpr.trimmedDescription
        let arrayCount = parsed.arrayParams.count
        let argsUnpack = (0 ..< arrayCount).map { "_arrays[\($0)]" }.joined(separator: ", ")
        let closureCall =
            "(\(parsed.closureText))(_model\(arrayCount > 0 ? ", " : "")\(argsUnpack))"

        let innerBody: String
        switch parsed.returnForm {
        case .single:
            innerBody = "[\(closureCall)]"
        case .array:
            innerBody = closureCall
        }

        let isStatic = funcDecl.modifiers.contains { modifier in
            modifier.name.tokenKind == .keyword(.static)
        }
        let staticPrefix = isStatic ? "static " : ""

        let peerDecl: DeclSyntax =
            """
            \(raw: staticPrefix)private let _mlxvg_\(raw: funcName) =
                valueAndGrad(model: \(raw: modelExprText)) { _model, _arrays -> [MLXArray] in
                    \(raw: innerBody)
                }
            """

        return [peerDecl]
    }
}

// MARK: - BodyMacro

extension MLXValueAndGradMacro: BodyMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingBodyFor declaration: some DeclSyntaxProtocol & WithOptionalCodeBlockSyntax,
        in context: some MacroExpansionContext
    ) throws -> [CodeBlockItemSyntax] {
        guard let funcDecl = declaration.as(FunctionDeclSyntax.self) else {
            return []
        }

        // Skip body generation if validation would fail
        guard funcDecl.body == nil,
            funcDecl.signature.effectSpecifiers?.asyncSpecifier == nil,
            funcDecl.signature.effectSpecifiers?.throwsClause == nil,
            funcDecl.genericParameterClause == nil
        else {
            return []
        }

        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
            return []
        }

        let (_, closureExpr) = extractArguments(arguments)
        guard let closureExpr else {
            return []
        }

        guard let parsed = parseClosure(closureExpr, in: context) else {
            return []
        }

        let funcName = funcDecl.name.text
        let params = Array(funcDecl.signature.parameterClause.parameters)

        // First param is model name
        let modelParamName = (params[0].secondName ?? params[0].firstName).text
        let arrayParamNames = params.dropFirst().map { ($0.secondName ?? $0.firstName).text }
        let arrayPack = "[\(arrayParamNames.joined(separator: ", "))]"

        switch parsed.returnForm {
        case .single:
            let stmts: CodeBlockItemSyntax =
                """
                let (_v, _g) = _mlxvg_\(raw: funcName)(\(raw: modelParamName), \(raw: arrayPack))
                return (_v[0], _g)
                """
            return [stmts]
        case .array:
            let stmt: CodeBlockItemSyntax =
                "return _mlxvg_\(raw: funcName)(\(raw: modelParamName), \(raw: arrayPack))"
            return [stmt]
        }
    }
}

// MARK: - Diagnostics

extension MLXValueAndGradMacro {
    enum Diagnostic: String, DiagnosticMessage {
        case notAFunction
        case hasBody
        case missingClosure
        case missingModel
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
                return "@MLXValueAndGrad can only be applied to functions"
            case .hasBody:
                return "@MLXValueAndGrad requires a function without a body"
            case .missingClosure:
                return
                    "@MLXValueAndGrad requires an implementation closure as the last argument"
            case .missingModel:
                return "@MLXValueAndGrad requires a model: argument"
            case .closureNeedsExplicitTypes:
                return
                    "#MLXValueAndGrad requires a closure with explicit parameter types and return type"
            case .nonMLXArrayParam:
                return
                    "@MLXValueAndGrad requires all parameters after the model to be of type MLXArray"
            case .unsupportedReturn:
                return
                    "@MLXValueAndGrad requires return type to be MLXArray or [MLXArray]"
            case .asyncNotSupported:
                return "@MLXValueAndGrad does not support async functions"
            case .throwsNotSupported:
                return "@MLXValueAndGrad does not support throwing functions"
            case .genericNotSupported:
                return "@MLXValueAndGrad does not support generic functions"
            }
        }

        var diagnosticID: MessageID {
            MessageID(domain: "MLXMacros", id: rawValue)
        }
    }
}
