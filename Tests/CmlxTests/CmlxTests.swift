// Copyright © 2024 Apple Inc.

//
//  Copyright © 2023 Apple. All rights reserved.
//

import Foundation
import XCTest

@testable import Cmlx

class CmlxTests: XCTestCase {

    func testMinimal() throws {
        // smoke test making sure we can build, link & call C api
        //
        // note: there are convenience wrappers in MLX + the entire
        // wrapping of the API in swift

        var data: [Float] = [1, 2, 3, 4, 5, 6]
        var shape: [Int32] = [2, 3]

        let arr = mlx_array_from_data(&data, &shape, 2, MLX_FLOAT32)!

        let str = mlx_tostring(UnsafeMutableRawPointer(arr))!
        defer { mlx_free(UnsafeMutableRawPointer(str)) }
        let description = String(cString: mlx_string_data(str))

        print(description)

        mlx_free(UnsafeMutableRawPointer(arr))
    }

}
