// Copyright © 2024 Apple Inc.

import Foundation
import MLX

@main
struct Example1 {
    static func main() {
        func getDeviceFromArgs() -> Device? {
            guard let index = CommandLine.arguments.firstIndex(of: "--device") else {
                return nil
            }

            let valueIndex = index + 1
            guard valueIndex < CommandLine.arguments.count else {
                print("Error: Missing value for option '--device'.")
                exit(1)
            }

            let value = CommandLine.arguments[valueIndex]
            switch value.lowercased() {
            case "cpu":
                return .cpu
            case "gpu":
                return .gpu
            default:
                print("Error: Invalid device: '\(value)'. Please use 'cpu' or 'gpu'.")
                exit(1)
            }
        }

        let specifiedDevice = getDeviceFromArgs()

        let defaultDevice: Device
        #if os(Linux)
            defaultDevice = .cpu
        #else
            defaultDevice = .gpu
        #endif

        let selectedDevice = specifiedDevice ?? defaultDevice

        print("Using device: \(selectedDevice).")

        Stream.withNewDefaultStream(device: selectedDevice) {
            let arr = MLXArray(stride(from: Int32(2), through: 8, by: 2), [2, 2])

            print(arr)
            print(arr.dtype)
            print(arr.shape)
            print(arr.ndim)
            print(arr.asType(.int64))

            print(arr[1])
            print(arr[0, 1].item(Int32.self))
        }
    }
}
