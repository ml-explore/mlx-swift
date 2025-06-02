// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class StreamTests: XCTestCase {

    func testEquatableDevice() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        // equality ignores index
        XCTAssertEqual(s1, s2)

        XCTAssertNotEqual(s1, s3)
        XCTAssertNotEqual(s2, s3)
    }

    func testDeviceType() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        XCTAssertEqual(s1.deviceType, .gpu)
        XCTAssertEqual(s2.deviceType, .gpu)
        XCTAssertEqual(s3.deviceType, .cpu)
    }

    func testUsingDevice() {
        let defaultDevice = Device.defaultDevice()

        Device.withDefaultDevice(.cpu) {
            // these _should_ be the same
            XCTAssertTrue(Device.defaultDevice().description.contains("cpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("cpu"))
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())

        Device.withDefaultDevice(.gpu) {
            XCTAssertTrue(Device.defaultDevice().description.contains("gpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
        }
        XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
    }

    func testSetUnsetDefaultDevice() {
        // Issue #237 -- setting an unsetting the default device in a loop
        // exhausts many resources
        for _ in 1 ..< 10000 {
            let defaultDevice = MLX.Device.defaultDevice()
            MLX.Device.setDefault(device: .cpu)
            defer {
                MLX.Device.setDefault(device: defaultDevice)
            }

            let x = MLXArray(1)
            let _ = x * x
        }
        print("here")
    }

    func testWithDefaultDevice() {
        // Issue #237 -- scoped variant
        for _ in 1 ..< 10000 {
            Device.withDefaultDevice(.cpu) {
                Device.withDefaultDevice(.gpu) {
                    let x = MLXArray(1)
                    let _ = x * x
                }
            }
        }
        print("here")
    }

    func disabledTestCreateStream() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            let _ = Stream(.cpu)
        }
        print("here")
    }

    func disabledTestCreateStreamScoped() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            Stream.withNewDefaultStream(device: .cpu) {
                let x = MLXArray(1)
                let _ = x * x
            }
        }
    }

}
