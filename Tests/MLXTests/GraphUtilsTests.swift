// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct GraphUtilsTests {

    @Test func exportToDotIncludesPrimitives() {
        let x = MLXArray([1, 2, 3])
        let dot = exportToDot([x * 2 + 1])

        #expect(!dot.isEmpty)
        #expect(dot.contains("digraph"), "expected a digraph, got:\n\(dot)")

        // the graph should mention the primitives that produce the output
        #expect(dot.contains("Add"), "expected Add in:\n\(dot)")
        #expect(dot.contains("Multiply"), "expected Multiply in:\n\(dot)")
    }

    @Test func graphDescriptionIncludesPrimitives() {
        let x = MLXArray([1, 2, 3])
        let description = graphDescription([x + x])

        #expect(!description.isEmpty)
        #expect(description.contains("Add"), "expected Add in:\n\(description)")
    }

    @Test func nodeNamer() {
        let x = MLXArray([1, 2, 3])
        let y = MLXArray([4, 5, 6])

        let namer = NodeNamer()
        namer.setName("lhs", for: x)
        namer.setName("rhs", for: y)

        #expect(namer.name(for: x) == "lhs")
        #expect(namer.name(for: y) == "rhs")

        let dot = exportToDot([x + y], namer: namer)
        #expect(dot.contains("lhs"), "expected lhs in:\n\(dot)")
        #expect(dot.contains("rhs"), "expected rhs in:\n\(dot)")
    }

    @Test func nodeNamerGeneratesNames() {
        // unnamed arrays get a generated name, and it is stable
        let x = MLXArray([1, 2, 3])
        let namer = NodeNamer()

        let first = namer.name(for: x)
        #expect(first != nil)
        #expect(namer.name(for: x) == first)
    }

    @Test func graphIsNotEvaluated() {
        // describing the graph should not force evaluation
        let x = MLXArray([1, 2, 3])
        let y = x * 2

        _ = exportToDot([y])

        // still usable afterwards
        expectEqual(y, MLXArray([2, 4, 6]))
    }
}
