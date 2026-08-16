// Copyright © 2026 Apple Inc.

import MLX
import XCTest

final class BatchInvariantTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testMatmulMatchesSingleRow() {
        let previous = GPU.batchInvariantLimit
        GPU.batchInvariantLimit = 4
        defer { GPU.batchInvariantLimit = previous }

        let m = 4
        let k = 4096
        let n = 4096
        let x = sin(arange(0, m * k).asType(.float32) * 0.013)
            .reshaped(m, k).asType(.bfloat16)
        let weight = cos(arange(0, n * k).asType(.float32) * 0.009)
            .reshaped(n, k).asType(.bfloat16)
        let xLast = x[(m - 1) ..< m]

        let block = matmul(x, weight.T)
        let single = matmul(xLast, weight.T)
        eval(block, single)

        XCTAssertTrue(block[(m - 1) ..< m].arrayEqual(single).item())
    }

    func testSDPAMatchesSingleQuery() {
        let previous = GPU.batchInvariantLimit
        GPU.batchInvariantLimit = 4
        defer { GPU.batchInvariantLimit = previous }

        let queryHeads = 8
        let kvHeads = 4
        let queryLength = 4
        let keyLength = 16_384
        let headDimension = 128
        MLXRandom.seed(42)
        let queries = MLXRandom.normal([1, queryHeads, queryLength, headDimension])
            .asType(.bfloat16)
        let keys = MLXRandom.normal([1, kvHeads, keyLength, headDimension])
            .asType(.bfloat16)
        let values = MLXRandom.normal([1, kvHeads, keyLength, headDimension])
            .asType(.bfloat16)
        let lastQuery = queries[
            0 ..< 1, 0..., (queryLength - 1) ..< queryLength, 0...]

        let block = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(headDimension)), mask: .causal)
        let single = MLXFast.scaledDotProductAttention(
            queries: lastQuery, keys: keys, values: values,
            scale: 1 / sqrt(Float(headDimension)), mask: .none)
        eval(block, single)

        let blockLast = block[
            0 ..< 1, 0..., (queryLength - 1) ..< queryLength, 0...]
        XCTAssertTrue(blockLast.arrayEqual(single).item())
    }
}
