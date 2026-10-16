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
        setDefaultDevice()
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
            for tensor in 0 ..< 4 {
                let name = "file\(file).tensor\(tensor)"
                arrays[name] = MLXArray(Int32(0) ..< 1024).reshaped(32, 32) + Int32(tensor)
            }
            urls.append(try write(arrays: arrays, name: "shard\(file).safetensors"))
            expected.merge(arrays) { _, new in new }
        }

        let loaded = try loadArrays(urls: urls)

        XCTAssertEqual(loaded.keys.sorted(), expected.keys.sorted())
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

        // The first file is submitted with asyncEval and the second is the checkedEval
        // barrier. Neither returned array may attempt deferred I/O after both are truncated.
        assertEqual(try XCTUnwrap(loaded["first"]), firstExpected)
        assertEqual(try XCTUnwrap(loaded["second"]), secondExpected)
    }

    func testLoadArraysEmpty() throws {
        let loaded = try loadArrays(urls: [])
        XCTAssertTrue(loaded.isEmpty)
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

    func testSafetensorSpansInFileOrder() throws {
        let arrays = [
            "big": MLX.zeros([64, 64]),
            "small": MLX.zeros([4]),
            "medium": MLX.zeros([16, 16]),
        ]
        let url = try write(arrays: arrays, name: "spans.safetensors")

        let spans = try safetensorSpansInFileOrder(url: url)
        XCTAssertEqual(spans.map(\.name).sorted(), ["big", "medium", "small"])
        XCTAssertEqual(
            spans.first { $0.name == "big" }?.byteCount, 64 * 64 * 4)
        XCTAssertEqual(
            spans.first { $0.name == "small" }?.byteCount, 4 * 4)

        // total spans bytes never exceed the file size
        let fileSize = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64)
        XCTAssertLessThanOrEqual(spans.reduce(0) { $0 + $1.byteCount }, fileSize)
    }

    func testSafetensorSpansRejectsMalformedFile() throws {
        let url = temporaryPath.appending(path: "bad.safetensors", directoryHint: .notDirectory)
        try Data([1, 2, 3]).write(to: url)
        XCTAssertThrowsError(try safetensorSpansInFileOrder(url: url))
    }

    func testSafetensorSpansRejectsOverflowingOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[-9223372036854775808,9223372036854775807]}}"#,
            name: "overflow.safetensors")

        XCTAssertThrowsError(try safetensorSpansInFileOrder(url: url))
    }

    func testSafetensorSpansRejectsNonintegerOffsets() throws {
        for (name, offset) in [("fractional", "1.5"), ("boolean", "true")] {
            let url = try writeRawSafetensor(
                header:
                    #"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[0,"#
                    + offset + #"]}}"#,
                name: "\(name).safetensors")

            XCTAssertThrowsError(try safetensorSpansInFileOrder(url: url))
        }
    }

    func testSafetensorSpansRejectsNoncontiguousOffsets() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
            payload: Data([0, 0, 0]),
            name: "noncontiguous.safetensors")

        XCTAssertThrowsError(try safetensorSpansInFileOrder(url: url))
    }

    func testSafetensorSpansOrdersEmptyTensorBeforeSharedOffset() throws {
        let url = try writeRawSafetensor(
            header:
                #"{"value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#,
            payload: Data([42]),
            name: "empty.safetensors")

        let spans = try safetensorSpansInFileOrder(url: url)
        XCTAssertEqual(spans.map(\.name), ["empty", "value"])
        XCTAssertEqual(spans.map(\.byteCount), [0, 1])
    }

    func testContiguousLoadGroups() {
        // empty
        XCTAssertEqual(contiguousLoadGroups(byteCounts: [], groupCount: 4), [])

        // single group
        XCTAssertEqual(
            contiguousLoadGroups(byteCounts: [1, 2, 3], groupCount: 1), [0 ..< 3])

        // balanced split preserving order, covering every index exactly once
        let byteCounts: [Int64] = [10, 10, 10, 10, 10, 10, 10, 10]
        let groups = contiguousLoadGroups(byteCounts: byteCounts, groupCount: 4)
        XCTAssertEqual(groups, [0 ..< 2, 2 ..< 4, 4 ..< 6, 6 ..< 8])

        // more groups than elements: no empty ranges, still covers everything
        let tiny = contiguousLoadGroups(byteCounts: [5, 5], groupCount: 8)
        XCTAssertEqual(tiny.flatMap { Array($0) }, [0, 1])

        // skewed sizes still cover every index in order
        let skewed = contiguousLoadGroups(
            byteCounts: [1000, 1, 1, 1, 1000, 1, 1, 1000], groupCount: 3)
        XCTAssertEqual(skewed.flatMap { Array($0) }, Array(0 ..< 8))

        // Computing proportional boundaries must not overflow for very large files.
        let huge = contiguousLoadGroups(
            byteCounts: [.max - 1, 1], groupCount: 16)
        XCTAssertEqual(huge.flatMap { Array($0) }, [0, 1])
    }

    func testConcurrentLoadGroupCount() {
        XCTAssertEqual(concurrentLoadGroupCount(processorCount: 2), 4)
        XCTAssertEqual(concurrentLoadGroupCount(processorCount: 10), 10)
        XCTAssertEqual(concurrentLoadGroupCount(processorCount: 32), 16)
    }
}
