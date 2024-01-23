import XCTest
import MLX

/// assert two arrays have the same shape and contents
func assertEqual(_ array1: MLXArray, _ array2: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-8, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(array1.shape, array2.shape, "shapes differ: \(array1.shape) != \(array2.shape)")
    XCTAssertTrue(array1.allClose(array2, rtol: rtol, atol: atol).item(Bool.self), "contents differ:\n\(array1)\n\(array2))))")
}

