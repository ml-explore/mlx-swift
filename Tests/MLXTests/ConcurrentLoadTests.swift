// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

@Suite
final class ConcurrentLoadTests {

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

    private func write(
        arrays: [String: MLXArray], metadata: [String: String] = [:], name: String
    ) throws -> URL {
        let url = temporaryPath.appending(path: name, directoryHint: .notDirectory)
        try save(arrays: arrays, metadata: metadata, url: url)
        return url
    }

    private func writeRawSafetensor(
        header: String, payload: Data = Data(), name: String
    ) throws -> URL {
        let url = temporaryPath.appending(path: name, directoryHint: .notDirectory)
        var headerLength = UInt64(header.utf8.count).littleEndian
        var data = withUnsafeBytes(of: &headerLength) { Data($0) }
        data.append(contentsOf: header.utf8)
        data.append(payload)
        try data.write(to: url)
        return url
    }

    @Test func loadArraysMatchesSerialLoads() throws {
        var urls = [URL]()
        var expected = [String: MLXArray]()
        for file in 0 ..< 3 {
            var arrays = [String: MLXArray]()
            // More roots than MLX's default BFS width exercises a larger evaluation tape.
            for tensor in 0 ..< 32 {
                let name = "file\(file).tensor\(tensor)"
                arrays[name] = MLXArray(Int32(0) ..< 1024).reshaped(32, 32) + Int32(tensor)
            }
            urls.append(try write(arrays: arrays, name: "shard\(file).safetensors"))
            expected.merge(arrays) { _, new in new }
        }

        let loaded = try loadArrays(urls: urls)

        #expect(loaded.keys.sorted() == expected.keys.sorted())
        #expect(loaded.values.allSatisfy { isEvaluated($0) })
        for (name, array) in expected {
            expectEqual(try #require(loaded[name]), array)
        }
    }

    @Test func loadArraysSingleFile() throws {
        let arrays = [
            "a": MLXArray(converting: [1.5, 2.5, 3.5]),
            "b": MLX.ones([8, 8]),
        ]
        let url = try write(arrays: arrays, name: "single.safetensors")

        let loaded = try loadArrays(urls: [url])
        #expect(loaded.keys.sorted() == arrays.keys.sorted())
        for (name, array) in arrays {
            expectEqual(try #require(loaded[name]), array)
        }
    }

    @Test func loadArraysMaterializesBeforeReturning() throws {
        let firstExpected = MLXArray(Int32(0) ..< 4096).reshaped(64, 64)
        let secondExpected = MLXArray(Int32(4096) ..< 8192).reshaped(64, 64)
        let firstURL = try write(
            arrays: ["first": firstExpected], name: "first-materialized.safetensors")
        let secondURL = try write(
            arrays: ["second": secondExpected], name: "second-materialized.safetensors")

        let loaded = try loadArrays(urls: [firstURL, secondURL])
        for url in [firstURL, secondURL] {
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: 0)
            try handle.close()
        }

        // The checkedEval barrier covers every file. Neither returned array may attempt
        // deferred I/O after both are truncated.
        expectEqual(try #require(loaded["first"]), firstExpected)
        expectEqual(try #require(loaded["second"]), secondExpected)
    }

    @Test func loadArraysEmpty() throws {
        let loaded = try loadArrays(urls: [])
        #expect(loaded.isEmpty)
    }

    @Test func loadArraysMetadataOnlyFile() throws {
        let url = try writeRawSafetensor(
            header: #"{"__metadata__":{"format":"test"}}"#,
            name: "metadata-only.safetensors")

        let (arrays, metadata) = try loadArraysAndMetadata(urls: [url])
        #expect(arrays.isEmpty)
        #expect(metadata == ["format": "test"])
    }

    @Test func loadArraysFallbackMaterializesEveryTensor() throws {
        // MLX accepts gaps, but our ordering parser rejects them and falls back.
        let url = try writeRawSafetensor(
            header:
                #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
            payload: Data([7, 0, 9]),
            name: "fallback.safetensors")
        #expect(throws: (any Error).self) { try safetensorNamesInFileOrder(url: url) }

        let loaded = try loadArrays(urls: [url])
        #expect(loaded.values.allSatisfy { isEvaluated($0) })
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 0)
        try handle.close()

        #expect(try #require(loaded["a"]).item(UInt8.self) == 7)
        #expect(try #require(loaded["b"]).item(UInt8.self) == 9)
    }

    @Test func duplicateNameLaterFileWins() throws {
        let first = try write(arrays: ["x": MLX.zeros([4])], name: "first.safetensors")
        let second = try write(arrays: ["x": MLX.ones([4])], name: "second.safetensors")

        let loaded = try loadArrays(urls: [first, second])
        expectEqual(try #require(loaded["x"]), MLX.ones([4]))
    }

    @Test func metadataMergedInFileOrder() throws {
        let first = try write(
            arrays: ["a": MLX.zeros([2])],
            metadata: ["shared": "first", "only-first": "1"],
            name: "first.safetensors")
        let second = try write(
            arrays: ["b": MLX.ones([2])],
            metadata: ["shared": "second", "only-second": "2"],
            name: "second.safetensors")

        let (arrays, metadata) = try loadArraysAndMetadata(urls: [first, second])
        #expect(arrays.keys.sorted() == ["a", "b"])
        #expect(metadata == ["shared": "second", "only-first": "1", "only-second": "2"])
    }

    @Test func missingFileThrows() throws {
        let good = try write(arrays: ["a": MLX.zeros([2])], name: "good.safetensors")
        let missing = temporaryPath.appending(
            path: "missing.safetensors", directoryHint: .notDirectory)

        #expect(throws: MLXError.self) { try loadArrays(urls: [good, missing]) }
    }

    @Test func unknownExtensionThrows() throws {
        let bogus = temporaryPath.appending(path: "weights.bin", directoryHint: .notDirectory)
        try Data([0, 1, 2, 3]).write(to: bogus)

        #expect(throws: LoadSaveError.self) { try loadArrays(urls: [bogus]) }
    }

    // MARK: - internals

    @Test func safetensorNamesInFileOrderListsEveryTensor() throws {
        let arrays = [
            "big": MLX.zeros([64, 64]),
            "small": MLX.zeros([4]),
            "medium": MLX.zeros([16, 16]),
        ]
        let url = try write(arrays: arrays, name: "spans.safetensors")

        let names = try safetensorNamesInFileOrder(url: url)
        #expect(names.sorted() == ["big", "medium", "small"])
    }

    @Test func safetensorNamesFollowOffsetsRatherThanNameOrHeaderOrder() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"middle":{"dtype":"U8","shape":[2],"data_offsets":[1,3]},"alpha":{"dtype":"U8","shape":[1],"data_offsets":[3,4]},"zeta":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payload: Data([1, 2, 3, 4]),
            name: "offset-order.safetensors")

        #expect(try safetensorNamesInFileOrder(url: url) == ["zeta", "middle", "alpha"])
    }

    @Test func safetensorNamesRejectsMalformedFile() throws {
        let url = temporaryPath.appending(path: "bad.safetensors", directoryHint: .notDirectory)
        try Data([1, 2, 3]).write(to: url)
        #expect(throws: (any Error).self) { try safetensorNamesInFileOrder(url: url) }
    }

    @Test func safetensorNamesRejectsOverflowingOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[-9223372036854775808,9223372036854775807]}}"#,
            name: "overflow.safetensors")

        #expect(throws: (any Error).self) { try safetensorNamesInFileOrder(url: url) }
    }

    @Test func safetensorNamesRejectsNonintegerOffsets() throws {
        for (name, offset) in [("fractional", "1.5"), ("boolean", "true")] {
            let url = try writeRawSafetensor(
                header:
                    #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[0,"#
                    + offset + #"]}}"#,
                name: "\(name).safetensors")

            #expect(throws: (any Error).self) { try safetensorNamesInFileOrder(url: url) }
        }
    }

    @Test func safetensorNamesRejectsNoncontiguousOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
            payload: Data([0, 0, 0]),
            name: "noncontiguous.safetensors")

        #expect(throws: (any Error).self) { try safetensorNamesInFileOrder(url: url) }
    }

    @Test func safetensorNamesOrdersEmptyTensorBeforeSharedOffset() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
            payload: Data([42]),
            name: "empty.safetensors")

        #expect(try safetensorNamesInFileOrder(url: url) == ["empty", "value"])
    }

    @Test func evaluationOrderReversesFilesAndOffsetsWithoutDroppingDuplicateNames() {
        let first = ["shared": MLXArray(Int32(1)), "last": MLXArray(Int32(2))]
        let second = ["shared": MLXArray(Int32(3)), "last": MLXArray(Int32(4))]
        let roots = safetensorArraysInEvaluationOrder(
            arrays: [first, second], names: [["shared", "last"], ["shared", "last"]])

        // eval_impl pops its tape from the back: all four independent Load roots must
        // be retained in reverse order, including the name overwritten by the merge.
        #expect(roots.map { $0.item(Int32.self) } == [4, 3, 2, 1])
    }

    @Test func evaluationOrderIncludesEveryArrayWhenPlanningFailsOrNamesDisagree() {
        let arrays = ["a": MLXArray(Int32(1)), "b": MLXArray(Int32(2))]
        for names: [String]? in [nil, [], ["a"], ["a", "missing"]] {
            let roots = safetensorArraysInEvaluationOrder(arrays: [arrays], names: [names])
            #expect(roots.map { $0.item(Int32.self) }.sorted() == [1, 2])
        }
    }
}
