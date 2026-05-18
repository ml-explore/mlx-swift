// Copyright © 2024-2025 Apple Inc.

import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct MLXMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        MLXCompileMacro.self,
        MLXValueAndGradMacro.self,
    ]
}
