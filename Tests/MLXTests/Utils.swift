// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

/// assert two arrays have the same shape and contents
func assertEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(array1.shape, array2.shape, "shapes differ: \(array1.shape) != \(array2.shape)")
    XCTAssertTrue(
        array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self),
        "contents differ:\n\(array1)\n\(array2)")
}

func assertEqual(
    _ array1: [MLXArray], _ array2: [MLXArray], rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(array1.count, array2.count, file: file, line: line)
    for (e1, e2) in zip(array1, array2) {
        assertEqual(e1, e2, rtol: rtol, atol: atol, file: file, line: line)
    }
}

func assertNotEqual(
    _ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(array1.shape, array2.shape, "shapes differ: \(array1.shape) != \(array2.shape)")
    XCTAssertFalse(
        array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self),
        "contents same:\n\(array1)\n\(array2)")
}

class DeviceScopedTestCase: XCTestCase {
    class var testDevice: Device { .gpu }

    override func invokeTest() {
        Device.withDefaultDevice(type(of: self).testDevice) {
            super.invokeTest()
        }
    }
}

class CPUDeviceScopedTestCase: DeviceScopedTestCase {
    override class var testDevice: Device { .cpu }
}

func findBuiltExecutable(named name: String, for testCase: XCTestCase) -> URL? {
    for directory in builtProductSearchDirectories(for: testCase) {
        let candidate = directory.appendingPathComponent(name)
        if FileManager.default.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
    }

    return nil
}

func builtExecutableNotFoundMessage(named name: String, for testCase: XCTestCase) -> String {
    let paths = builtProductSearchDirectories(for: testCase).map(\.path).joined(separator: ", ")
    return "\(name) binary not found in build products. Searched: \(paths)"
}

private func builtProductSearchDirectories(for testCase: XCTestCase) -> [URL] {
    var directories: [URL] = []

    func appendUnique(_ url: URL?) {
        guard let url else { return }
        let normalized = url.standardizedFileURL
        if !directories.contains(normalized) {
            directories.append(normalized)
        }
    }

    let bundleProducts = Bundle(for: type(of: testCase)).bundleURL.deletingLastPathComponent()
    appendUnique(bundleProducts)

    if let builtProductsDir = ProcessInfo.processInfo.environment["BUILT_PRODUCTS_DIR"] {
        appendUnique(URL(fileURLWithPath: builtProductsDir, isDirectory: true))
    }

    let executableDirectory = URL(fileURLWithPath: CommandLine.arguments[0])
        .deletingLastPathComponent()
    appendUnique(executableDirectory)

    return directories
}
