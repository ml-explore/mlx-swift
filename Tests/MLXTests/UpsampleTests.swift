// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

class UpsampleTests: XCTestCase {

    func testNearest() {
        // BHWC
        let input = MLXArray([1, 2, 3, 4], [1, 2, 2, 1])

        let up = Upsample(scaleFactor: 2.0, mode: .nearest)
        let result = up(input).squeezed()

        XCTAssertEqual(result.shape, [4, 4])

        // array([[1, 1, 2, 2],
        //        [1, 1, 2, 2],
        //        [3, 3, 4, 4],
        //        [3, 3, 4, 4]], dtype=int32)
        let expected = MLXArray([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4], [4, 4])
            .asType(.int32)
        assertEqual(result, expected)
    }

    func testLinear() {
        // BHWC
        let input = MLXArray([1, 2, 3, 4], [1, 2, 2, 1])

        let up = Upsample(scaleFactor: 2.0, mode: .linear())
        let result = up(input).squeezed()

        XCTAssertEqual(result.shape, [4, 4])

        // array([[1, 1.25, 1.75, 2],
        //        [1.5, 1.75, 2.25, 2.5],
        //        [2.5, 2.75, 3.25, 3.5],
        //        [3, 3.25, 3.75, 4]], dtype=float32)
        let expected = MLXArray(
            converting: [
                1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5,
                2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
            ], [4, 4])
        assertEqual(result, expected)
    }

}
