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
            // the file), or lazily, when the arrays are evaluated and the short read is
            // turned into an error (ml-explore/mlx-c#130).
            do {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath) { _ in }
                try checkedEval(Array(loadedArrays.values) as [Any])
                XCTFail("a truncated safetensors file must not load successfully")
            } catch {
                // expected
            }
        }
    }

    /// mlx validates the tensor data offsets against the size of the file while it parses
    /// the header, so a file that is _already_ truncated fails eagerly. A file that is
    /// truncated after the header is parsed can only be caught when the lazy arrays are
    /// evaluated and the read comes up short -- `mlx_io_reader` turns that into an error
    /// rather than leaving the destination buffer uninitialized (ml-explore/mlx-c#130).
    public func testLoadFailsWhenFileIsTruncatedBeforeEvaluation() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let safetensorsPath = temporaryPath.appending(
                path: "truncated-late.safetensors",
                directoryHint: .notDirectory
            )

            let arrays: [String: MLXArray] = [
                "foo": MLX.ones([128, 128]),
                "bar": MLX.zeros([64, 256]),
            ]
            try MLX.save(arrays: arrays, url: safetensorsPath)

            // the header is parsed and validated against the size of the file here, but
            // the tensor data is only read once the lazy arrays are evaluated
            let recorder = ProgressRecorder()
            let loadedArrays = try MLX.loadArrays(url: safetensorsPath) {
                @Sendable in recorder.record($0)
            }
            let size = try XCTUnwrap(recorder.reported.first?.totalUnitCount)

            // ... so truncate the file out from under them. `truncate()` shortens the
            // inode the reader already has open, unlike a rewrite which may replace it.
            XCTAssertEqual(
                truncate(safetensorsPath.path(percentEncoded: false), off_t(size - 32)), 0)

            do {
                try checkedEval(Array(loadedArrays.values) as [Any])
                XCTFail("evaluating arrays read from a truncated file must fail")
            } catch {
                // expected
            }

            // the bytes that were read are still accounted for, monotonically, and the
            // aggregate stops short of the original size of the file
            let fractions = recorder.values
            XCTAssertEqual(fractions, fractions.sorted())
            XCTAssertEqual(fractions.first, 0)
            XCTAssertLessThan(try XCTUnwrap(fractions.last), 1)
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

    /// `loadArrays(data:)` seeks to the end to size its input, so the in-memory IO
    /// stream has to report the size of the whole buffer rather than an offset
    /// relative to the read position.
    public func testLoadFromDataLargerThanHeader() throws {
        // enough data that a SEEK_END mistake shows up as a truncated size
        let arrays: [String: MLXArray] = [
            "big": MLX.ones([64, 64])
        ]

        let data = try saveToData(arrays: arrays)
        XCTAssertGreaterThan(data.count, 8)

        let loaded = try loadArrays(data: data)
        assertEqual(try XCTUnwrap(loaded["big"]), try XCTUnwrap(arrays["big"]))
    }

    /// A truncated in-memory buffer must be reported as an error. The in-memory reader
    /// reports a read that runs past the end of the buffer as zero bytes so that
    /// `mlx_io_reader` throws instead of leaving the destination uninitialized
    /// (ml-explore/mlx-c#130).
    public func testLoadFromTruncatedDataFails() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let arrays: [String: MLXArray] = [
                "big": MLX.ones([64, 64])
            ]

            var data = try saveToData(arrays: arrays)
            data.removeLast(32)

            do {
                let loaded = try loadArrays(data: data)
                try checkedEval(Array(loaded.values) as [Any])
                XCTFail("a truncated safetensors buffer must not load successfully")
            } catch {
                // expected
            }
        }
    }

}
