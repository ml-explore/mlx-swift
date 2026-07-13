//
//  SaveTests.swift
//
//
//  Created by Rounak Jain on 4/2/24.
//

import MLX
import XCTest
import os

private final class ProgressRecorder: Sendable {
    private let fractions = OSAllocatedUnfairLock(initialState: [Double]())

    func record(_ progress: LoadProgress) {
        fractions.withLock { values in
            values.append(progress.fractionCompleted)
        }
    }

    var values: [Double] {
        fractions.withLock { values in
            values
        }
    }
}

final class SaveTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    public func testSaveArrays() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
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
    }

    public func testLoadArraysProgressReportsThroughEvaluation() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "arrays.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            let recorder = ProgressRecorder()
            let loadedArrays = try MLX.loadArrays(
                url: safetensorsPath
            ) { @Sendable progress in
                recorder.record(progress)
            }

            assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))
            assertEqual(try XCTUnwrap(loadedArrays["bar"]), try XCTUnwrap(arrays["bar"]))

            let fractions = recorder.values
            XCTAssertGreaterThan(fractions.count, 1)
            XCTAssertEqual(fractions.first, 0)
            XCTAssertEqual(fractions.last, 1)
            XCTAssertEqual(fractions, fractions.sorted())
        }
    }

    public func testLoadArraysProgressFailsOnTruncatedTensorData() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "truncated.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            var data = try Data(contentsOf: safetensorsPath)
            data.removeLast(32)
            try data.write(to: safetensorsPath)

            let loadedArrays = try MLX.loadArrays(url: safetensorsPath) { _ in }

            XCTAssertThrowsError(
                try checkedEval(Array(loadedArrays.values) as [Any])
            )
        }
    }

    public func testSaveArray() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
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
    }

    public func testSaveArraysData() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
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
    }

    public func testSaveArraysMetadataData() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
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

}
