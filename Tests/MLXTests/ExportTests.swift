// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class ExportTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    func testExportFunction() throws {
        let url = temporaryPath.appending(path: "fn.mlxfn")

        func f(arrays: [MLXArray]) -> [MLXArray] {
            [arrays[0] * arrays[1]]
        }

        let x = MLXArray(1)
        let y = MLXArray([1, 2, 3])

        try exportFunction(to: url, f)(x, y: y)

        // load it back in
        let f2 = try importFunction(from: url)

        let a = MLXArray(10)
        let b = MLXArray([5, 10, 20])

        // call it -- the shapes and labels have to match
        let r = try f2(a, y: b)[0]
        assertEqual(r, MLXArray([50, 100, 200]))
    }

    func testExportFunctions() throws {
        let url = temporaryPath.appending(path: "fn.mlxfn")

        func f(_ arrays: [MLXArray]) -> [MLXArray] {
            [arrays.dropFirst().reduce(arrays[0], +)]
        }

        let x = MLXArray([1])

        try exportFunctions(to: url, shapeless: true, f) { export in
            try export(x)
            try export(x, x)
            try export(x, x, x)
        }

        // load it back in
        let f2 = try importFunction(from: url)

        let a = MLXArray([10, 10, 10])
        let b = MLXArray([5, 10, 20])
        let c = MLXArray([1, 2, 3])

        let r1 = try f2(a)[0]
        assertEqual(r1, a)

        let r2 = try f2(a, b)[0]
        assertEqual(r2, a + b)

        let r3 = try f2(a, b, c)[0]
        assertEqual(r3, a + b + c)
    }

    func testExportError() {
        func f(arrays: [MLXArray]) -> [MLXArray] {
            [arrays[0] * arrays[1]]
        }

        let x = MLXArray(1)
        let y = MLXArray([1, 2, 3])

        do {
            try exportFunction(to: URL(fileURLWithPath: "/does/not/exist"), f)(x, y: y)
            XCTFail("should throw")
        } catch {
            // expected
        }
    }

}
