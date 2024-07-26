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

    func testEquatableStreamOrDevice() {
        let s1 = StreamOrDevice.gpu
        let s2 = StreamOrDevice.stream(.init(index: 0, .gpu))
        let s3 = StreamOrDevice.cpu

        XCTAssertEqual(s1, s2)
        XCTAssertNotEqual(s1, s3)
        XCTAssertNotEqual(s2, s3)
    }

    func testUsingDevice() {
        let defaultDevice = Device.defaultDevice()

        using(device: .cpu) {
            XCTAssertEqual(Device.defaultDevice(), .cpu)
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())

        using(device: .gpu) {
            XCTAssertEqual(Device.defaultDevice(), .gpu)
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())
    }
}
