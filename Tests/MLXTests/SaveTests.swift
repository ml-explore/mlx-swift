//
//  SaveTests.swift
//
//
//  Created by Rounak Jain on 4/2/24.
//

import Foundation
import MLX
import Testing

// All access to the recorded callbacks is protected by the lock.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var progress = [LoadProgress]()

    func record(_ progress: LoadProgress) {
        lock.withLock {
            self.progress.append(progress)
        }
    }

    var reported: [LoadProgress] {
        lock.withLock { progress }
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

@Suite
final class SaveTests {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    init() throws {
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    deinit {
        // deinit cannot throw -- a leftover temporary directory is not a test failure
        try? FileManager.default.removeItem(at: temporaryPath)
    }

    @Test func saveArrays() throws {
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
        #expect(loadedArrays.keys.sorted() == arrays.keys.sorted())

        expectEqual(try #require(loadedArrays["foo"]), try #require(arrays["foo"]))
        expectEqual(try #require(loadedArrays["bar"]), try #require(arrays["bar"]))
    }

    @Test func loadArraysProgressReportsThroughEvaluation() throws {
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

            expectEqual(try #require(loadedArrays["foo"]), try #require(arrays["foo"]))
            expectEqual(try #require(loadedArrays["bar"]), try #require(arrays["bar"]))

            let fractions = recorder.values
            #expect(fractions.count > 1)
            #expect(fractions.first == 0)
            #expect(fractions.last == 1)
            #expect(fractions == fractions.sorted())
        }
    }

    @Test func loadArraysProgressFailsOnTruncatedTensorData() throws {
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
            #expect(
                throws: MLXError.self,
                "a truncated safetensors file must not load successfully"
            ) {
                let loadedArrays = try MLX.loadArrays(url: safetensorsPath) { _ in }
                try checkedEval(Array(loadedArrays.values) as [Any])
            }
        }
    }

    /// mlx validates the tensor data offsets against the size of the file while it parses
    /// the header, so a file that is _already_ truncated fails eagerly. A file that is
    /// truncated after the header is parsed can only be caught when the lazy arrays are
    /// evaluated and the read comes up short -- `mlx_io_reader` turns that into an error
    /// rather than leaving the destination buffer uninitialized (ml-explore/mlx-c#130).
    @Test func loadFailsWhenFileIsTruncatedBeforeEvaluation() throws {
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
            let size = try #require(recorder.reported.first?.totalUnitCount)

            // ... so truncate the file out from under them. `truncate()` shortens the
            // inode the reader already has open, unlike a rewrite which may replace it.
            #expect(
                truncate(safetensorsPath.path(percentEncoded: false), off_t(size - 32)) == 0)

            #expect(
                throws: MLXError.self, "evaluating arrays read from a truncated file must fail"
            ) {
                try checkedEval(Array(loadedArrays.values) as [Any])
            }

            // the bytes that were read are still accounted for, monotonically, and the
            // aggregate stops short of the original size of the file
            let fractions = recorder.values
            #expect(fractions == fractions.sorted())
            #expect(fractions.first == 0)
            #expect(try #require(fractions.last) < 1)
        }
    }

    @Test func scopedLoadProgressHandler() throws {
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

            expectEqual(try #require(loadedArrays["foo"]), try #require(arrays["foo"]))

            let fractions = recorder.values
            #expect(fractions.count > 1)
            #expect(fractions.first == 0)
            #expect(fractions.last == 1)
            #expect(fractions == fractions.sorted())
            #expect(Set(recorder.reported.map(\.url)) == [safetensorsPath])
        }
    }

    @Test func scopedLoadProgressHandlerIsScoped() throws {
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

            #expect(recorder.reported.isEmpty)
        }
    }

    @Test func scopedLoadProgressAggregatesAcrossFiles() throws {
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

            #expect(Set(recorder.reported.map(\.url)) == Set(shards))
            for url in shards {
                #expect(recorder.values(for: url).last == 1)
            }
            expectEqual(recorder.aggregateFraction, 1, accuracy: 1e-9)
        }
    }

    @Test func progressChunksLargeTensorReads() throws {
        let url = temporaryPath.appending(path: "large.safetensors")
        let array = MLXArray(Array(repeating: Float(3), count: 3 * 1024 * 1024))
        try MLX.save(arrays: ["large": array], metadata: ["format": "test"], url: url)

        let recorder = ProgressRecorder()
        let (loaded, metadata) = try MLX.loadArraysAndMetadata(url: url) {
            recorder.record($0)
        }
        let headerBytes = try #require(recorder.reported.last?.completedUnitCount)
        #expect(headerBytes < Int64(array.nbytes))
        #expect(metadata == ["format": "test"])
        #expect(
            try #require(loaded["large"]).asArray(Float.self) == array.asArray(Float.self))

        let reports = recorder.reported
        #expect(reports.first?.completedUnitCount == 0)
        #expect(reports.last?.fractionCompleted == 1)
        #expect(
            reports.map(\.completedUnitCount) == reports.map(\.completedUnitCount).sorted())
        let tensorReports = reports.filter { $0.completedUnitCount > headerBytes }
        #expect(tensorReports.count >= 3)
        for (previous, current) in zip(reports, reports.dropFirst()) {
            #expect(
                current.completedUnitCount - previous.completedUnitCount <= 4 * 1024 * 1024)
        }
    }

    @Test func progressRetainsHandlerAndSkipsUnevaluatedArrays() throws {
        let url = temporaryPath.appending(path: "partial.safetensors")
        let values = Array(repeating: Float(7), count: 1024)
        try MLX.save(arrays: ["used": MLXArray(values), "unused": MLXArray(values)], url: url)

        let recorder = ProgressRecorder()
        let arrays = try withLoadProgressHandler({ recorder.record($0) }) {
            try MLX.loadArrays(url: url)
        }
        let headerBytes = try #require(recorder.reported.last?.completedUnitCount)
        #expect(try #require(arrays["used"]).asArray(Float.self) == values)
        let partial = try #require(recorder.reported.last)
        #expect(
            partial.completedUnitCount
                == headerBytes + Int64(values.count * MemoryLayout<Float>.size)
        )
        #expect(partial.fractionCompleted < 1)
        #expect(try #require(arrays["unused"]).asArray(Float.self) == values)
        #expect(recorder.reported.last?.fractionCompleted == 1)
    }

    @Test func nestedProgressHandlersRestoreAfterThrowAndExplicitHandlerWins() throws {
        enum TestError: Error { case expected }
        let url = temporaryPath.appending(path: "nested.safetensors")
        try MLX.save(arrays: ["weight": MLXArray([Float(1), 2])], url: url)
        let outer = ProgressRecorder()
        let inner = ProgressRecorder()
        let explicit = ProgressRecorder()

        try withLoadProgressHandler({ outer.record($0) }) {
            #expect(throws: TestError.expected) {
                try withLoadProgressHandler({ inner.record($0) }) {
                    let arrays = try MLX.loadArrays(url: url)
                    try checkedEval(Array(arrays.values) as [Any])
                    throw TestError.expected
                }
            }
            #expect(outer.reported.isEmpty)
            #expect(inner.reported.last?.fractionCompleted == 1)

            let explicitlyLoaded = try MLX.loadArrays(url: url) { explicit.record($0) }
            try checkedEval(Array(explicitlyLoaded.values) as [Any])
            #expect(outer.reported.isEmpty)
            #expect(explicit.reported.last?.fractionCompleted == 1)

            let arrays = try MLX.loadArrays(url: url)
            try checkedEval(Array(arrays.values) as [Any])
        }
        #expect(outer.reported.last?.fractionCompleted == 1)
    }

    @Test func asyncProgressScopeInheritsHandler() async throws {
        let url = temporaryPath.appending(path: "async.safetensors")
        try MLX.save(arrays: ["weight": MLXArray([Float(1), 2])], url: url)
        let recorder = ProgressRecorder()
        try await withLoadProgressHandler({ recorder.record($0) }) {
            try await Task {
                let arrays = try MLX.loadArrays(url: url)
                try checkedEval(Array(arrays.values) as [Any])
            }.value
        }
        #expect(recorder.reported.first?.fractionCompleted == 0)
        #expect(recorder.reported.last?.fractionCompleted == 1)
    }

    @Test func saveArray() throws {
        // single array npy file
        let path = temporaryPath.appending(
            path: "array.npy",
            directoryHint: .notDirectory
        )

        let array = MLX.ones([2, 4])

        try MLX.save(array: array, url: path)

        let loaded = try MLX.loadArray(url: path)

        expectEqual(array, loaded)
    }

    @Test func saveArraysData() throws {
        let arrays: [String: MLXArray] = [
            "foo": MLX.ones([1, 2]),
            "bar": MLX.zeros([2, 1]),
        ]

        let data = try saveToData(arrays: arrays)
        let loadedArrays = try loadArrays(data: data)
        #expect(loadedArrays.keys.sorted() == arrays.keys.sorted())

        expectEqual(try #require(loadedArrays["foo"]), try #require(arrays["foo"]))
        expectEqual(try #require(loadedArrays["bar"]), try #require(arrays["bar"]))
    }

    @Test func saveArraysMetadataData() throws {
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
        #expect(loadedArrays.keys.sorted() == arrays.keys.sorted())

        expectEqual(try #require(loadedArrays["foo"]), try #require(arrays["foo"]))
        expectEqual(try #require(loadedArrays["bar"]), try #require(arrays["bar"]))
        #expect(loadedMetadata == metadata)
    }

    /// `loadArrays(data:)` seeks to the end to size its input, so the in-memory IO
    /// stream has to report the size of the whole buffer rather than an offset
    /// relative to the read position.
    @Test func loadFromDataLargerThanHeader() throws {
        // enough data that a SEEK_END mistake shows up as a truncated size
        let arrays: [String: MLXArray] = [
            "big": MLX.ones([64, 64])
        ]

        let data = try saveToData(arrays: arrays)
        #expect(data.count > 8)

        let loaded = try loadArrays(data: data)
        expectEqual(try #require(loaded["big"]), try #require(arrays["big"]))
    }

    /// A truncated in-memory buffer must be reported as an error. The in-memory reader
    /// reports a read that runs past the end of the buffer as zero bytes so that
    /// `mlx_io_reader` throws instead of leaving the destination uninitialized
    /// (ml-explore/mlx-c#130).
    @Test func loadFromTruncatedDataFails() throws {
        try MLX.Device.withDefaultDevice(.cpu) {
            let arrays: [String: MLXArray] = [
                "big": MLX.ones([64, 64])
            ]

            var data = try saveToData(arrays: arrays)
            data.removeLast(32)

            #expect(
                throws: MLXError.self,
                "a truncated safetensors buffer must not load successfully"
            ) {
                let loaded = try loadArrays(data: data)
                try checkedEval(Array(loaded.values) as [Any])
            }
        }
    }

}
