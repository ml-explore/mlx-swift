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
        GPU.batchInvariantLimit = 8
        defer { GPU.batchInvariantLimit = previous }

        let queryHeads = 2
        let kvHeads = 1
        let queryLength = 8
        let headDimension = 64
        let queries = sin(
            arange(0, queryHeads * queryLength * headDimension).asType(.float32) * 0.017
        )
        .reshaped(1, queryHeads, queryLength, headDimension)
        .asType(.bfloat16)

        for keyLength in [1_028, 8_196, 16_388, 65_540] {
            let keys = cos(
                arange(0, kvHeads * keyLength * headDimension).asType(.float32) * 0.011
            )
            .reshaped(1, kvHeads, keyLength, headDimension)
            .asType(.bfloat16)
            let values = sin(
                arange(0, kvHeads * keyLength * headDimension).asType(.float32) * 0.007
            )
            .reshaped(1, kvHeads, keyLength, headDimension)
            .asType(.bfloat16)
            let block = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: 1 / sqrt(Float(headDimension)), mask: .causal)

            for row in 0 ..< queryLength {
                let prefix = keyLength - queryLength + row + 1
                let single = MLXFast.scaledDotProductAttention(
                    queries: queries[0 ..< 1, 0..., row ..< (row + 1), 0...],
                    keys: keys[0 ..< 1, 0..., 0 ..< prefix, 0...],
                    values: values[0 ..< 1, 0..., 0 ..< prefix, 0...],
                    scale: 1 / sqrt(Float(headDimension)), mask: .none)
                let blockRow = block[0 ..< 1, 0..., row ..< (row + 1), 0...]
                eval(blockRow, single)
                XCTAssertTrue(
                    blockRow.arrayEqual(single).item(),
                    "keyLength=\(keyLength), row=\(row), prefix=\(prefix)")
            }
        }
    }

    func testQuantizedMatmulMatchesSingleRows() {
        let previous = GPU.batchInvariantLimit
        let limit = 12
        GPU.batchInvariantLimit = limit
        defer { GPU.batchInvariantLimit = previous }

        let k = 4_096
        let n = 512
        let x = sin(arange(0, limit * k).asType(.float32) * 0.013)
            .reshaped(limit, k)
        let weight = cos(arange(0, n * k).asType(.float32) * 0.009)
            .reshaped(n, k)
        let q = quantized(weight, groupSize: 32, bits: 4)

        for m in 1 ... limit {
            let block = quantizedMM(
                x[0 ..< m], q.wq, scales: q.scales, biases: q.biases,
                groupSize: 32, bits: 4)
            for row in 0 ..< m {
                let single = quantizedMM(
                    x[row ..< (row + 1)], q.wq, scales: q.scales, biases: q.biases,
                    groupSize: 32, bits: 4)
                let blockRow = block[row ..< (row + 1)]
                eval(blockRow, single)
                XCTAssertTrue(blockRow.arrayEqual(single).item(), "m=\(m), row=\(row)")
            }
        }
    }
}
