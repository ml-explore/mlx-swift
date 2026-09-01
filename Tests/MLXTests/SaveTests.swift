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
    private let progress = OSAllocatedUnfairLock(initialState: [LoadProgress]())

    func record(_ progress: LoadProgress) {
        self.progress.withLock { values in
            values.append(progress)
        }
    }

    var reported: [LoadProgress] {
        progress.withLock { values in
            values
        }
    }

    var values: [Double] {
        reported.map { $0.fractionCompleted }
    }

    /// Fractions reported for a single file, in order.
    func values(for url: URL) -> [Double] {
        reported.filter { $0.url == url }.map { $0.fractionCompleted }
    }

    /// Aggregate fraction across every file seen, by bytes.
    var aggregateFraction: Double {
        var completed = [URL: Int64]()
        var total = [URL: Int64]()
        for progress in reported {
            completed[progress.url] = progress.completedUnitCount
            total[progress.url] = progress.totalUnitCount
        }
        let totalBytes = total.values.reduce(0, +)
        guard totalBytes > 0 else { return 0 }
        return Double(completed.values.reduce(0, +)) / Double(totalBytes)
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

            // A truncated file has to be reported either eagerly, while the header is
            // parsed (mlx >= 0.32.1 validates the tensor data offsets against the size of
            // the file), or lazily, when the arrays are evaluated and the read fails
            // (ml-explore/mlx#3742 + ml-explore/mlx-c#126).
            var thrownError: Error?
            do {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath) { _ in }
                try checkedEval(Array(loadedArrays.values) as [Any])
            } catch {
                thrownError = error
            }

            if thrownError == nil {
                throw XCTSkip(
                    """
                    the vendored mlx/mlx-c silently ignores a failed read from a custom \
                    io reader -- requires mlx >= 0.32.1 (ml-explore/mlx#3742) and \
                    ml-explore/mlx-c#126
                    """)
            }
        }
    }

    public func testScopedLoadProgressHandler() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "scoped.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            let recorder = ProgressRecorder()

            // note: the plain loadArrays(url:) -- no progress handler passed at the call site
            let loadedArrays = try withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
                MLX.eval(Array(loadedArrays.values))
                return loadedArrays
            }

            assertEqual(try XCTUnwrap(loadedArrays["foo"]), try XCTUnwrap(arrays["foo"]))

            let fractions = recorder.values
            XCTAssertGreaterThan(fractions.count, 1)
            XCTAssertEqual(fractions.first, 0)
            XCTAssertEqual(fractions.last, 1)
            XCTAssertEqual(fractions, fractions.sorted())
            XCTAssertEqual(Set(recorder.reported.map(\.url)), [safetensorsPath])
        }
    }

    public func testScopedLoadProgressHandlerIsScoped() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "unscoped.safetensors",
                directoryHint: .notDirectory
            )
            try MLX.save(arrays: ["foo": MLX.ones([128, 128])], url: safetensorsPath)

            let recorder = ProgressRecorder()
            withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
            }

            // outside the scope nothing is reported
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath)
            MLX.eval(Array(loadedArrays.values))

            XCTAssertTrue(recorder.reported.isEmpty)
        }
    }

    public func testScopedLoadProgressAggregatesAcrossFiles() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let shards = try (0 ..< 3).map { index -> URL in
                let url = temporaryPath.appending(
                    path: "shard-\(index).safetensors",
                    directoryHint: .notDirectory
                )
                try MLX.save(arrays: ["w\(index)": MLX.ones([64, 128])], url: url)
                return url
            }

            let recorder = ProgressRecorder()
            try withLoadProgressHandler({ @Sendable in recorder.record($0) }) {
                // this mimics a model loader: several shards loaded lazily, then evaluated
                var weights = [String: MLXArray]()
                for url in shards {
                    let (w, _) = try MLX.loadArraysAndMetadata(url: url)
                    weights.merge(w) { _, new in new }
                }
                MLX.eval(Array(weights.values))
            }

            XCTAssertEqual(Set(recorder.reported.map(\.url)), Set(shards))
            for url in shards {
                XCTAssertEqual(recorder.values(for: url).last, 1)
            }
            XCTAssertEqual(recorder.aggregateFraction, 1, accuracy: 1e-9)
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
