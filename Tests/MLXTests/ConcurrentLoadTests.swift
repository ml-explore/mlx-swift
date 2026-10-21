// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX

final class ConcurrentLoadTests: XCTestCase {

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

    func testLoadArraysMatchesSerialLoads() throws {
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

        XCTAssertEqual(loaded.keys.sorted(), expected.keys.sorted())
        XCTAssertTrue(loaded.values.allSatisfy { isEvaluated($0) })
        for (name, array) in expected {
            assertEqual(try XCTUnwrap(loaded[name]), array)
        }
    }

    func testLoadArraysSingleFile() throws {
        let arrays = [
            "a": MLXArray(converting: [1.5, 2.5, 3.5]),
            "b": MLX.ones([8, 8]),
        ]
        let url = try write(arrays: arrays, name: "single.safetensors")

        let loaded = try loadArrays(urls: [url])
        XCTAssertEqual(loaded.keys.sorted(), arrays.keys.sorted())
        for (name, array) in arrays {
            assertEqual(try XCTUnwrap(loaded[name]), array)
        }
    }

    func testLoadArraysMaterializesBeforeReturning() throws {
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
        assertEqual(try XCTUnwrap(loaded["first"]), firstExpected)
        assertEqual(try XCTUnwrap(loaded["second"]), secondExpected)
    }

    func testLoadArraysEmpty() throws {
        let loaded = try loadArrays(urls: [])
        XCTAssertTrue(loaded.isEmpty)
    }

    func testLoadArraysMetadataOnlyFile() throws {
        let url = try writeRawSafetensor(
            header: #"{"__metadata__":{"format":"test"}}"#,
            name: "metadata-only.safetensors")

        let (arrays, metadata) = try loadArraysAndMetadata(urls: [url])
        XCTAssertTrue(arrays.isEmpty)
        XCTAssertEqual(metadata, ["format": "test"])
    }

    func testLoadArraysFallbackMaterializesEveryTensor() throws {
        // MLX accepts gaps, but our ordering parser rejects them and falls back.
        let url = try writeRawSafetensor(
            header:
                #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
            payload: Data([7, 0, 9]),
            name: "fallback.safetensors")
        XCTAssertThrowsError(try safetensorNamesInFileOrder(url: url))

        let loaded = try loadArrays(urls: [url])
        XCTAssertTrue(loaded.values.allSatisfy { isEvaluated($0) })
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 0)
        try handle.close()

        XCTAssertEqual(try XCTUnwrap(loaded["a"]).item(UInt8.self), 7)
        XCTAssertEqual(try XCTUnwrap(loaded["b"]).item(UInt8.self), 9)
    }

    func testDuplicateNameLaterFileWins() throws {
        let first = try write(arrays: ["x": MLX.zeros([4])], name: "first.safetensors")
        let second = try write(arrays: ["x": MLX.ones([4])], name: "second.safetensors")

        let loaded = try loadArrays(urls: [first, second])
        assertEqual(try XCTUnwrap(loaded["x"]), MLX.ones([4]))
    }

    func testMetadataMergedInFileOrder() throws {
        let first = try write(
            arrays: ["a": MLX.zeros([2])],
            metadata: ["shared": "first", "only-first": "1"],
            name: "first.safetensors")
        let second = try write(
            arrays: ["b": MLX.ones([2])],
            metadata: ["shared": "second", "only-second": "2"],
            name: "second.safetensors")

        let (arrays, metadata) = try loadArraysAndMetadata(urls: [first, second])
        XCTAssertEqual(arrays.keys.sorted(), ["a", "b"])
        XCTAssertEqual(
            metadata, ["shared": "second", "only-first": "1", "only-second": "2"])
    }

    func testMissingFileThrows() throws {
        let good = try write(arrays: ["a": MLX.zeros([2])], name: "good.safetensors")
        let missing = temporaryPath.appending(
            path: "missing.safetensors", directoryHint: .notDirectory)

        XCTAssertThrowsError(try loadArrays(urls: [good, missing]))
    }

    func testUnknownExtensionThrows() throws {
        let bogus = temporaryPath.appending(path: "weights.bin", directoryHint: .notDirectory)
        try Data([0, 1, 2, 3]).write(to: bogus)

        XCTAssertThrowsError(try loadArrays(urls: [bogus]))
    }

    // MARK: - internals

    func testSafetensorNamesInFileOrder() throws {
        let arrays = [
            "big": MLX.zeros([64, 64]),
            "small": MLX.zeros([4]),
            "medium": MLX.zeros([16, 16]),
        ]
        let url = try write(arrays: arrays, name: "spans.safetensors")

        let names = try safetensorNamesInFileOrder(url: url)
        XCTAssertEqual(names.sorted(), ["big", "medium", "small"])
    }

    func testSafetensorNamesFollowOffsetsRatherThanNameOrHeaderOrder() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"middle":{"dtype":"U8","shape":[2],"data_offsets":[1,3]},"alpha":{"dtype":"U8","shape":[1],"data_offsets":[3,4]},"zeta":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            payload: Data([1, 2, 3, 4]),
            name: "offset-order.safetensors")

        XCTAssertEqual(try safetensorNamesInFileOrder(url: url), ["zeta", "middle", "alpha"])
    }

    func testSafetensorNamesRejectsMalformedFile() throws {
        let url = temporaryPath.appending(path: "bad.safetensors", directoryHint: .notDirectory)
        try Data([1, 2, 3]).write(to: url)
        XCTAssertThrowsError(try safetensorNamesInFileOrder(url: url))
    }

    func testSafetensorNamesRejectsOverflowingOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[-9223372036854775808,9223372036854775807]}}"#,
            name: "overflow.safetensors")

        XCTAssertThrowsError(try safetensorNamesInFileOrder(url: url))
    }

    func testSafetensorNamesRejectsNonintegerOffsets() throws {
        for (name, offset) in [("fractional", "1.5"), ("boolean", "true")] {
            let url = try writeRawSafetensor(
                header:
                    #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[0,"#
                    + offset + #"]}}"#,
                name: "\(name).safetensors")

            XCTAssertThrowsError(try safetensorNamesInFileOrder(url: url))
        }
    }

    func testSafetensorNamesRejectsNoncontiguousOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
            payload: Data([0, 0, 0]),
            name: "noncontiguous.safetensors")

        XCTAssertThrowsError(try safetensorNamesInFileOrder(url: url))
    }

    func testSafetensorNamesOrdersEmptyTensorBeforeSharedOffset() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
            payload: Data([42]),
            name: "empty.safetensors")

        XCTAssertEqual(try safetensorNamesInFileOrder(url: url), ["empty", "value"])
    }

    func testEvaluationOrderReversesFilesAndOffsetsWithoutDroppingDuplicateNames() {
        let first = ["shared": MLXArray(Int32(1)), "last": MLXArray(Int32(2))]
        let second = ["shared": MLXArray(Int32(3)), "last": MLXArray(Int32(4))]
        let roots = safetensorArraysInEvaluationOrder(
            arrays: [first, second], names: [["shared", "last"], ["shared", "last"]])

        // eval_impl pops its tape from the back: all four independent Load roots must
        // be retained in reverse order, including the name overwritten by the merge.
        XCTAssertEqual(roots.map { $0.item(Int32.self) }, [4, 3, 2, 1])
    }

    func testEvaluationOrderIncludesEveryArrayWhenPlanningFailsOrNamesDisagree() {
        let arrays = ["a": MLXArray(Int32(1)), "b": MLXArray(Int32(2))]
        for names: [String]? in [nil, [], ["a"], ["a", "missing"]] {
            let roots = safetensorArraysInEvaluationOrder(arrays: [arrays], names: [names])
            XCTAssertEqual(roots.map { $0.item(Int32.self) }.sorted(), [1, 2])
        }
    }
}
