// Copyright © 2026 Apple Inc.

import CoreFoundation
import Foundation

// MARK: - Concurrent safetensors loading

// Safetensors payload reads are lazy. Submit tensors in reverse file-offset order
// because MLX builds its evaluation tape in input order and consumes it from the back.
// Each Load primitive queues its read on MLX's I/O pool, which owns read concurrency;
// one checkedEval schedules the complete checkpoint and waits for all reads to finish.

/// Match the limit enforced by MLX and the safetensors reference implementation.
private let maximumSafetensorHeaderBytes: UInt64 = 100_000_000

/// Decode a JSON integer without accepting booleans, floating-point values, or values
/// outside the supported signed 64-bit offset range.
private func safetensorOffset(_ value: Any) -> Int64? {
    guard let number = value as? NSNumber else { return nil }
    let numberType = String(cString: number.objCType)
    guard
        CFGetTypeID(number) != CFBooleanGetTypeID(),
        numberType != "f",
        numberType != "d",
        numberType != "D"
    else {
        return nil
    }
    let offset = number.int64Value
    return offset >= 0 ? offset : nil
}

/// The tensor names in the safetensors file at `url`, ordered by position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorNamesInFileOrder(url: URL) throws -> [String] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    guard headerLength > 0, headerLength < maximumSafetensorHeaderBytes else {
        throw Malformed()
    }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, end: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = safetensorOffset(offsets[0]),
            let end = safetensorOffset(offsets[1]),
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end))
    }
    // Safetensors requires the data buffer to be covered contiguously. Sorting by end as
    // well handles zero-byte tensors that share the following tensor's begin offset.
    spans.sort {
        $0.begin == $1.begin ? $0.end < $1.end : $0.begin < $1.begin
    }
    var expectedBegin: Int64 = 0
    for span in spans {
        guard span.begin == expectedBegin else { throw Malformed() }
        expectedBegin = span.end
    }
    return spans.map(\.name)
}

/// Roots in the order MLX must receive them to schedule reads in file-offset order.
func safetensorArraysInEvaluationOrder(
    arrays: [[String: MLXArray]], names: [[String]?]
) -> [MLXArray] {
    var orderedArrays = [MLXArray]()
    for (fileArrays, fileNames) in zip(arrays, names) {
        if let fileNames,
            fileNames.count == fileArrays.count,
            fileNames.allSatisfy({ fileArrays[$0] != nil })
        {
            orderedArrays.append(contentsOf: fileNames.compactMap { fileArrays[$0] })
        } else {
            // A planning failure or disagreement with MLX's header read must not leave
            // any arrays out of the completion barrier. Their read order is unspecified.
            orderedArrays.append(contentsOf: fileArrays.values)
        }
    }
    // Reverse the whole checkpoint so MLX visits files in URL order and tensors within
    // each file in ascending offset order. Keep duplicate names from different files.
    return orderedArrays.reversed()
}

/// Lock-guarded results of the parallel, header-only preparation pass.
private final class ConcurrentLoadPreparation: @unchecked Sendable {
    private let lock = NSLock()
    private var perFileArrays: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var namesPerFile: [[String]?]
    private var firstError: Error?

    init(fileCount: Int) {
        perFileArrays = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
        namesPerFile = Array(repeating: nil, count: fileCount)
    }

    func set(
        file: Int,
        arrays: [String: MLXArray],
        metadata: [String: String],
        names: [String]?
    ) {
        lock.lock()
        defer { lock.unlock() }
        perFileArrays[file] = arrays
        perFileMetadata[file] = metadata
        namesPerFile[file] = names
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    func result() throws -> (
        arrays: [[String: MLXArray]],
        metadata: [[String: String]],
        names: [[String]?]
    ) {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        return (perFileArrays, perFileMetadata, namesPerFile)
    }
}

/// Load a dictionary of ``MLXArray`` from several `safetensors` files at once,
/// scheduling payload reads in file-offset order and waiting for every array.
///
/// The dictionaries are merged in file order: on a duplicate name the later file wins,
/// matching a serial load-and-merge loop over the same `urls`. Passing a single URL
/// also submits its tensors in file-offset order. MLX handles read concurrency.
///
/// This function is synchronous and blocks until every file is loaded. From async code,
/// call it via a continuation on a `DispatchQueue` rather than directly on a Swift
/// concurrency cooperative thread.
///
/// - Parameters:
///     - urls: URLs of the `safetensors` files to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``loadArraysAndMetadata(urls:stream:)``
public func loadArrays(urls: [URL], stream: StreamOrDevice = .cpu) throws -> [String: MLXArray] {
    try loadArraysAndMetadata(urls: urls, stream: stream).0
}

/// Load a dictionary of ``MLXArray`` and merged metadata `[String: String]` from several
/// `safetensors` files at once, scheduling payload reads in file-offset order.
///
/// See ``loadArrays(urls:stream:)``. The metadata dictionaries are merged with the same
/// rule as the arrays: on a duplicate key the later file wins.
///
/// - Parameters:
///     - urls: URLs of the `safetensors` files to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(urls:stream:)``
/// - ``loadArraysAndMetadata(url:stream:)``
public func loadArraysAndMetadata(urls: [URL], stream: StreamOrDevice = .cpu) throws -> (
    [String: MLXArray], [String: String]
) {
    // Only header parsing happens on Dispatch worker threads. Payload reads stay lazy
    // until checkedEval below, where MLX owns their concurrency.
    let preparation = ConcurrentLoadPreparation(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: urls.count) { file in
        let url = urls[file]
        let names = try? safetensorNamesInFileOrder(url: url)
        do {
            let (arrays, metadata) = try loadArraysAndMetadata(url: url, stream: stream)
            preparation.set(
                file: file, arrays: arrays, metadata: metadata, names: names)
        } catch {
            preparation.record(error: error)
        }
    }

    let prepared = try preparation.result()
    let perFileArrays = prepared.arrays
    let perFileMetadata = prepared.metadata
    let evaluationOrder = safetensorArraysInEvaluationOrder(
        arrays: perFileArrays, names: prepared.names)

    // checkedEval schedules the ordered roots under evalLock and waits outside it.
    // Keep every root, including overwritten names, through this checked I/O barrier.
    try checkedEval(evaluationOrder)

    var arrays = [String: MLXArray]()
    var metadata = [String: String]()
    for fileArrays in perFileArrays {
        arrays.merge(fileArrays) { _, new in new }
    }
    for fileMetadata in perFileMetadata {
        metadata.merge(fileMetadata) { _, new in new }
    }
    return (arrays, metadata)
}
