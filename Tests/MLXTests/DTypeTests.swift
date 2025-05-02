// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class DTypeTests: XCTestCase {

    func testScalarAsMLXArrayInt() {
        // Int (which is really a 64 bit value) produces .int32 to match
        // python mlx
        let i = 10
        let intArray = i.asMLXArray(dtype: nil)
        XCTAssertEqual(intArray.dtype, .int32)

        // but using Int64 directly will give .int64
        let i64 = Int64(10)
        let int64Array = i64.asMLXArray(dtype: nil)
        XCTAssertEqual(int64Array.dtype, .int64)
    }

    func testScalarAsMLXArrayPromotionInt() {
        let i = 10

        // does not promote arrays to larger type,
        // e.g. ([1, 2, 3] dtype: .int8) + scalar(10)
        // won't promote the whole thing to int32.
        XCTAssertEqual(i.asMLXArray(dtype: .int8).dtype, .int8)
        XCTAssertEqual(i.asMLXArray(dtype: .uint32).dtype, .uint32)
        XCTAssertEqual(i.asMLXArray(dtype: .int64).dtype, .int64)

        XCTAssertEqual(i.asMLXArray(dtype: .bool).dtype, .int32)

        XCTAssertEqual(i.asMLXArray(dtype: .float16).dtype, .float16)
    }

    func testScalarAsMLXArrayPromotionFloat() {
        let f: Float = 1.5

        // floats do promote to a matching float type
        XCTAssertEqual(f.asMLXArray(dtype: .int8).dtype, .float32)
        XCTAssertEqual(f.asMLXArray(dtype: .uint32).dtype, .float32)
        XCTAssertEqual(f.asMLXArray(dtype: .int64).dtype, .float32)

        XCTAssertEqual(f.asMLXArray(dtype: .bool).dtype, .float32)

        // but ([1.5, 2.5, 3.5] dtype: .float16) + scalar(1.5)
        // doesn't promote the array to float32
        XCTAssertEqual(f.asMLXArray(dtype: .float16).dtype, .float16)
    }

}
