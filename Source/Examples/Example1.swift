// Copyright © 2024 Apple Inc.

import Foundation
import MLX

@main
struct Example1 {
    static func main() {
        // create from stride (sequence of 2, 4, 6, 8)
        let arr = MLXArray(stride(from: Int32(2), through: 8, by: 2), [2, 2])

        print(arr)
        print(arr.dtype)
        print(arr.shape)
        print(arr.ndim)
        print(arr.asType(.int64))

        // print a row
        print(arr[1])

        // print a value
        print(arr[0, 1].item(Int32.self))
    }
}
