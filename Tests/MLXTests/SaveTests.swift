//
//  SaveTests.swift
//
//
//  Created by Rounak Jain on 4/2/24.
//

import MLX
import XCTest

final class SaveTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        setDefaultDevice()
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    public func testSaveArrays() throws {
        let safetensorsPath = temporaryPath.appending(
            path: "arrays.safetensors",
            directoryHint: .notDirectory
        )

        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        try MLX.save(arrays: arrays, url: safetensorsPath)

        let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    public func testSaveArray() throws {
        // single array npy file
        let path = temporaryPath.appending(
            path: "array.npy",
            directoryHint: .notDirectory
        )

        let array = MLX.ones([2, 4])

        try MLX.save(array: array, url: path)

        let loaded = try MLX.loadArray(url: path)

        assertEqual(array, loaded)
    }

    public func testSaveArraysData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        let data = try saveToData(arrays: arrays)
        let loadedArrays = try loadArrays(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
    }

    public func testSaveArraysMetadataData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]
        let metadata = [
            "key": "value",
            "key2": "value2",
        ]

        let data = try saveToData(arrays: arrays, metadata: metadata)
        let (loadedArrays, loadedMetadata) = try loadArraysAndMetadata(data: data)
        XCTAssertEqual(loadedArrays.keys.sorted(), arrays.keys.sorted())

        assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
        assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))
        XCTAssertEqual(loadedMetadata, metadata)
    }

}
