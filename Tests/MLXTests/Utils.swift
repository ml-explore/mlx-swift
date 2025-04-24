// Copyright © 2024 Apple Inc.

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

func setDefaultDevice() {
    MLX.Device.setDefault(device: .gpu)
}
