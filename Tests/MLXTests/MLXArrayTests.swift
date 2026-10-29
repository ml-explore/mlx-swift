// Copyright © 2024 Apple Inc.

import Foundation
import Testing

@testable import MLX

@Suite
struct MLXArrayTests {

    @Test func arrayProperties() {
        let a = MLXArray(converting: [3.5, 4.5, 5.5, 7.0, 9.4, 10.0], [2, 3, 1])

        #expect(a.itemSize == 4)
        #expect(a.size == 6)
        #expect(a.count == 2)
        #expect(a.nbytes == 6 * 4)
        #expect(a.ndim == 3)
        #expect(a.dtype == .float32)
        #expect(a.shape == [2, 3, 1])
    }

    @Test func arrayRead() {
        let a = MLXArray(0 ..< 12, [4, 3])

        #expect(a.asArray(Int32.self) == Array(0 ..< 12))
        #expect(a[1][2].item(Int.self) == 5)
    }

    @Test func asArrayContiguous() {
        // read array from contiguous memory
        let a = MLXArray(0 ..< 12, [4, 3])
        let b = a.asArray(Int.self)
        #expect(b == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

    @Test func asArrayEmpty() {
        // an empty slice has size 0 and no backing pointer; it should return
        // an empty array rather than crashing (see issue #480)
        let a = MLXArray.ones([1, 200])
        let b = a[128...]
        #expect(b.shape == [0, 200])
        #expect(b.size == 0)
        #expect(b.asArray(Int.self) == [])
        #expect(b.asArray(Float.self) == [])

        // a directly-constructed empty array behaves the same
        let empty = MLXArray([Int32](), [0])
        #expect(empty.asArray(Int32.self) == [])

        // a non-contiguous empty array goes through the strided copy path
        let t = b.T
        #expect(t.shape == [200, 0])
        #expect(t.asArray(Float.self) == [])
    }

    @Test func asDataEmpty() {
        // asData shares physicalSize with asArray, so it also returns empty
        // data for an empty array (see issue #480)
        let b = MLXArray.ones([1, 200])[128...]
        for access in [MLXArray.AccessMethod.copy, .noCopyIfContiguous, .noCopy] {
            let data = b.asData(access: access)
            #expect(data.data.count == 0)
            #expect(data.shape == [0, 200])
        }
        #expect(b.T.asData(access: .copy).data.count == 0)
    }

    @Test func asArrayScalar() {
        // a scalar (0-dimensional) array yields a single-element array
        let a = MLXArray(7)
        #expect(a.shape == [])
        #expect(a.size == 1)
        #expect(a.asArray(Int32.self) == [7])
    }

    @Test func asArrayNonContiguous1() {
        // skipping elements via slicing
        let a = MLXArray(0 ..< 9, [3, 3])

        let s = a[0 ..< 2, 1 ..< 3]
        expectEqual(s, MLXArray([1, 2, 4, 5], [2, 2]))

        #expect(s.shape == [2, 2])

        // size and nbytes are the logical size
        #expect(s.size == 2 * 2)
        #expect(s.nbytes == 2 * 2 * s.itemSize)

        // internal property for counting the physical size of the backing.
        // note that the physical size doesn't include the row that is
        // sliced out
        #expect(s.physicalSize == 3 * 2)

        // evaluating s (the comparison above) will realize the strides.
        // if we examine these before they might be [2, 1] which are the
        // "logical" strides
        #expect(s.internalStrides == [3, 1])

        let s_arr = s.asArray(Int32.self)
        #expect(s_arr == [1, 2, 4, 5])
    }

    @Test func asArrayNonContiguous2() {
        // a transpose via strides
        let a = MLXArray(0 ..< 12, [4, 3])

        let s = asStrided(a, [3, 4], strides: [1, 3])

        let expected: [Int32] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        expectEqual(s, MLXArray(expected, [3, 4]))

        // Note: be careful to use the matching type -- if we transcode
        // to a different type it will be converted to contiguous
        let s_arr = s.asArray(Int32.self)
        #expect(s_arr == expected)
    }

    @Test func asArrayNonContiguous4() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        let expected: [Int32] = [0, 2, 4, 6, 8, 10, 12, 14]
        expectEqual(s, MLXArray(expected, [4, 2]))

        #expect(s.internalStrides == [4, 2])

        let s_arr = s.asArray(Int32.self)
        #expect(s_arr == expected)
    }

    @Test func contiguousStridesForShapes() {
        #expect(contiguousStrides(shape: [1, 1, 1]) == [1, 1, 1])
        #expect(contiguousStrides(shape: [4, 4]) == [4, 1])
        #expect(contiguousStrides(shape: [4, 2]) == [2, 1])
        #expect(contiguousStrides(shape: [3, 2, 1, 5]) == [10, 5, 5, 1])
    }

    @Test func asDataContiguous() {
        // contiguous source
        let a = MLXArray(0 ..< 16, [4, 4])

        let result = a.asData()
        #expect(result.shape == [4, 4])
        #expect(result.strides == [4, 1])
        #expect(result.dType == .int32)
    }

    @Test func asDataContiguousNoCopy() {
        // contiguous source
        let a = MLXArray(0 ..< 16, [4, 4])

        do {
            let result = a.asData(access: .noCopy)
            #expect(result.shape == [4, 4])
            #expect(result.strides == [4, 1])
            #expect(result.dType == .int32)
        }
        do {
            let result = a.asData(access: .noCopyIfContiguous)
            #expect(result.shape == [4, 4])
            #expect(result.strides == [4, 1])
            #expect(result.dType == .int32)
        }
    }

    @Test func asDataRoundTrip() {
        let a = MLXArray(0 ..< 16, [4, 4])
        let arrayData = a.asData(access: .copy)
        let result = MLXArray(arrayData.data, arrayData.shape, dtype: arrayData.dType)
        expectEqual(a, result)
    }

    @Test func asDataNonContiguous() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        let result = s.asData()
        #expect(result.shape == [4, 2])
        #expect(result.strides == [2, 1])
        #expect(result.dType == .int32)
    }

    @Test func asDataNonContiguousNoCopy() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        do {
            // the strides will match the strided array
            let result = s.asData(access: .noCopy)
            #expect(result.shape == [4, 2])
            #expect(result.strides == [4, 2])
            #expect(result.dType == .int32)
        }
        do {
            // it isn't contiguous so we will get a contiguous copy
            let result = s.asData(access: .noCopyIfContiguous)
            #expect(result.shape == [4, 2])
            #expect(result.strides == [2, 1])
            #expect(result.dType == .int32)
        }
    }

    @Test func updateInternalWithItself() {
        // `Module.update(parameters:)` can hand an array its own value back
        let a = MLXArray([1, 2, 3] as [Float])
        a._updateInternal(a)
        #expect(a.asArray(Float.self) == [1, 2, 3])
    }

}
