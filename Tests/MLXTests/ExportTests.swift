// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

/// A class rather than a struct so that `deinit` can clean up -- Swift Testing
/// creates a new instance per test, so `init` / `deinit` take the place of
/// XCTest's `setUpWithError` / `tearDownWithError`.
@Suite
final class ExportTests {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    init() throws {
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    deinit {
        // deinit cannot throw -- a leftover temporary directory is not a test failure
        try? FileManager.default.removeItem(at: temporaryPath)
    }

    @Test func exportAndImportFunction() throws {
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
        expectEqual(r, MLXArray([50, 100, 200]))
    }

    @Test func exportAndImportShapelessFunctions() throws {
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
        expectEqual(r1, a)

        let r2 = try f2(a, b)[0]
        expectEqual(r2, a + b)

        let r3 = try f2(a, b, c)[0]
        expectEqual(r3, a + b + c)
    }

    @Test func exportToInvalidPathThrows() {
        func f(arrays: [MLXArray]) -> [MLXArray] {
            [arrays[0] * arrays[1]]
        }

        let x = MLXArray(1)
        let y = MLXArray([1, 2, 3])

        #expect(throws: MLXError.self) {
            try exportFunction(to: URL(fileURLWithPath: "/does/not/exist"), f)(x, y: y)
        }
    }

}
