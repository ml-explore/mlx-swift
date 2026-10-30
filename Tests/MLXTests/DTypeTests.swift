// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct DTypeTests {

    @Test func scalarAsMLXArrayInt() {
        // Int (which is really a 64 bit value) produces .int32 to match
        // python mlx
        let i = 10
        let intArray = i.asMLXArray(dtype: nil)
        #expect(intArray.dtype == .int32)

        // but using Int64 directly will give .int64
        let i64 = Int64(10)
        let int64Array = i64.asMLXArray(dtype: nil)
        #expect(int64Array.dtype == .int64)
    }

    @Test func scalarAsMLXArrayPromotionInt() {
        let i = 10

        // does not promote arrays to larger type,
        // e.g. ([1, 2, 3] dtype: .int8) + scalar(10)
        // won't promote the whole thing to int32.
        #expect(i.asMLXArray(dtype: .int8).dtype == .int8)
        #expect(i.asMLXArray(dtype: .uint32).dtype == .uint32)
        #expect(i.asMLXArray(dtype: .int64).dtype == .int64)

        #expect(i.asMLXArray(dtype: .bool).dtype == .int32)

        #expect(i.asMLXArray(dtype: .float16).dtype == .float16)
    }

    @Test func scalarAsMLXArrayPromotionFloat() {
        let f: Float = 1.5

        // floats do promote to a matching float type
        #expect(f.asMLXArray(dtype: .int8).dtype == .float32)
        #expect(f.asMLXArray(dtype: .uint32).dtype == .float32)
        #expect(f.asMLXArray(dtype: .int64).dtype == .float32)

        #expect(f.asMLXArray(dtype: .bool).dtype == .float32)

        // but ([1.5, 2.5, 3.5] dtype: .float16) + scalar(1.5)
        // doesn't promote the array to float32
        #expect(f.asMLXArray(dtype: .float16).dtype == .float16)
    }

}
