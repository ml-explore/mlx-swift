// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class GraphUtilsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testExportToDot() {
        let x = MLXArray([1, 2, 3])
        let dot = exportToDot([x * 2 + 1])

        XCTAssertFalse(dot.isEmpty)
        XCTAssertTrue(dot.contains("digraph"), "expected a digraph, got:\n\(dot)")

        // the graph should mention the primitives that produce the output
        XCTAssertTrue(dot.contains("Add"), "expected Add in:\n\(dot)")
        XCTAssertTrue(dot.contains("Multiply"), "expected Multiply in:\n\(dot)")
    }

    func testGraphDescription() {
        let x = MLXArray([1, 2, 3])
        let description = graphDescription([x + x])

        XCTAssertFalse(description.isEmpty)
        XCTAssertTrue(description.contains("Add"), "expected Add in:\n\(description)")
    }

    func testNodeNamer() {
        let x = MLXArray([1, 2, 3])
        let y = MLXArray([4, 5, 6])

        let namer = NodeNamer()
        namer.setName("lhs", for: x)
        namer.setName("rhs", for: y)

        XCTAssertEqual(namer.name(for: x), "lhs")
        XCTAssertEqual(namer.name(for: y), "rhs")

        let dot = exportToDot([x + y], namer: namer)
        XCTAssertTrue(dot.contains("lhs"), "expected lhs in:\n\(dot)")
        XCTAssertTrue(dot.contains("rhs"), "expected rhs in:\n\(dot)")
    }

    func testNodeNamerGeneratesNames() {
        // unnamed arrays get a generated name, and it is stable
        let x = MLXArray([1, 2, 3])
        let namer = NodeNamer()

        let first = namer.name(for: x)
        XCTAssertNotNil(first)
        XCTAssertEqual(namer.name(for: x), first)
    }

    func testGraphIsNotEvaluated() {
        // describing the graph should not force evaluation
        let x = MLXArray([1, 2, 3])
        let y = x * 2

        _ = exportToDot([y])

        // still usable afterwards
        assertEqual(y, MLXArray([2, 4, 6]))
    }
}
