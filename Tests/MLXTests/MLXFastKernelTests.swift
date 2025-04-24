// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class MLXFastKernelTests: XCTestCase {

    func testCustomKernelBasic() {
        // based on def test_custom_kernel_basic
        MLXRandom.seed(7)
        let a = normal([2, 2])
        let kernel = MLXFast.metalKernel(
            name: "basic",
            inputNames: ["a"],
            outputNames: ["out1"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    out1[elem] = a[elem];
                """)

        let out = kernel(
            [a],
            grid: (4, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2]],
            outputDTypes: [.float32])

        XCTAssertTrue(allClose(out[0], a).all().item())
    }

    func testCustomKernelArgs() {
        // based on def test_custom_kernel_args
        MLXRandom.seed(7)
        let a = normal([3, 6])
        let c = normal([2, 2]).asType(.bfloat16)

        let kernel = MLXFast.metalKernel(
            name: "arg_test",
            inputNames: ["a", "b", "c", "d"],
            outputNames: ["out1", "out2"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    T tmp = a[0];
                    if (e) {
                        out1[elem] = a[1] + b[2] + c[3] + d + f;
                    } else {
                        out1[elem] = 1;
                    }
                    out2[elem] = a[1] + b[2] + c[1] - d;
                """)

        let out = kernel(
            [
                a,
                MLXArray([3, 4, 5]),
                c,
                7.3,
            ],
            template: [
                ("e", true),
                ("f", 3),
                ("T", DType.float16),
            ],
            grid: (6, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2], [3, 2]],
            outputDTypes: [.float32, .int32])

        XCTAssertTrue(allClose(out[0], full([2, 2], values: 14.0484)).all().item())
        XCTAssertTrue(allClose(out[1], full([3, 2], values: -2)).all().item())
    }

    func testFastSDPA() {
        // https://github.com/ml-explore/mlx-swift/issues/172
        // this will just make sure the MLXFast.scaled_dot_product_attention is
        // callable in the various cases, based on
        // https://github.com/ml-explore/mlx/blob/main/python/tests/test_fast_sdpa.py#L65-L87

        let Dk = 64
        let scale = 1.0 / sqrt(Float(Dk))
        let dTypes = [DType.float32, DType.float16]
        for SEQUENCE_LENGTH in [63, 129, 400] {
            for dtype in dTypes {
                let B = 2
                let H = 24
                let q = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let k = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let v = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)

                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil,
                    memoryEfficientThreshold: 2)

                eval(result)
            }
        }
    }
}
