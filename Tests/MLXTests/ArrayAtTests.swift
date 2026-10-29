// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct ArrayAtTests {

    @Test func arrayAt() {
        // from example at https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html#mlx.core.array.at

        // this references each index twice
        let idx = MLXArray([0, 1, 0, 1])

        // assign through index -- we can only observe the last assignment to a location
        let a0 = MLXArray([0, 0])
        a0[idx] = MLXArray(2)
        expectEqual(a0, MLXArray([2, 2]))

        // similar to above -- we can only observe one assignment, so we just get a +1
        // note: there was a bug in the += operator where the lhs was not inout and
        // this was producing [0, 0]
        let a1 = MLXArray([0, 0])
        a1[idx] += 1
        expectEqual(a1, MLXArray([1, 1]))

        // the bare add produces a value for each index including the duplicates
        let a2 = MLXArray([0, 0])
        expectEqual(a2[idx] + 1, MLXArray([1, 1, 1, 1]))

        // but the assign back through the index will collapse the values down
        // into the same location -- this is the same as a2[idx] += 1
        a2[idx] = a2[idx] + 1
        expectEqual(a2, MLXArray([1, 1]))

        // this will update 0 and 1 twice
        let a3 = MLXArray([0, 0])
        expectEqual(a3.at[idx].add(1), MLXArray([2, 2]))
    }

    // MARK: - slice updates
    //
    // pure slice indices route through the mlx_slice_update_* family rather than a
    // scatter -- these check that path produces the same answers.

    @Test func sliceAdd() {
        let a = MLXArray(0 ..< 6)
        expectEqual(a.at[1 ..< 4].add(10), MLXArray([0, 11, 12, 13, 4, 5]))
    }

    @Test func sliceSubtract() {
        let a = MLXArray(0 ..< 6)
        expectEqual(a.at[1 ..< 4].subtract(1), MLXArray([0, 0, 1, 2, 4, 5]))
    }

    @Test func sliceMultiply() {
        let a = MLXArray(1 ..< 7)
        expectEqual(a.at[2 ..< 5].multiply(10), MLXArray([1, 2, 30, 40, 50, 6]))
    }

    @Test func sliceDivide() {
        let a = MLXArray(converting: [2.0, 4.0, 6.0, 8.0])
        expectEqual(
            a.at[1 ..< 3].divide(2), MLXArray(converting: [2.0, 2.0, 3.0, 8.0]), atol: 1e-6)
    }

    @Test func sliceMinimumMaximum() {
        let a = MLXArray([5, 5, 5, 5])
        expectEqual(a.at[0 ..< 2].minimum(3), MLXArray([3, 3, 5, 5]))
        expectEqual(a.at[0 ..< 2].maximum(7), MLXArray([7, 7, 5, 5]))
    }

    @Test func sliceIndexAdd() {
        // a single integer index is also expressible as a slice update
        let a = MLXArray(0 ..< 4)
        expectEqual(a.at[1].add(100), MLXArray([0, 101, 2, 3]))
    }

    @Test func multiDimensionalSliceAdd() {
        let a = MLXArray(0 ..< 6, [2, 3])
        expectEqual(
            a.at[0 ..< 1, 1 ..< 3].add(10),
            MLXArray([0, 11, 12, 3, 4, 5], [2, 3]))
    }

    @Test func stridedSliceAdd() {
        let a = MLXArray(0 ..< 6)
        expectEqual(a.at[.stride(by: 2)].add(10), MLXArray([10, 1, 12, 3, 14, 5]))
    }

    @Test func sliceAndScatterAgree() {
        // the slice path and the scatter path should produce the same result --
        // indexing with an array forces the scatter fallback
        let a = MLXArray(0 ..< 6)

        let viaSlice = a.at[1 ..< 4].add(10)
        let viaScatter = a.at[MLXArray([1, 2, 3])].add(10)

        expectEqual(viaSlice, viaScatter)
    }
}
