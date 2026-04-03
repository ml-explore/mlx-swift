// Copyright © 2026 Apple Inc.

import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct MLXMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        MLXLiteralMacro.self
    ]
}
