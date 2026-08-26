// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

// MARK: - Concurrent safetensors loading

// The safetensors loader is lazy: ``loadArraysAndMetadata(url:stream:)`` reads only the
// header, and a tensor's bytes are read when its array is evaluated. Evaluating a whole
// checkpoint with a single `eval` visits the tensors in dictionary order -- effectively
// random file offsets -- and serializes the reads and the copies into unified memory.
//
// ``loadArrays(urls:stream:)`` instead splits every file into contiguous byte-balanced
// ranges of tensors, in file-offset order, and evaluates the ranges from concurrent work
// items. The reads become sequential and the header parsing, I/O, and copies of separate
// ranges overlap. Measured on an M4 Pro (14 cores, NVMe at ~6.1 GB/s sequential): loading
// an 18 GB checkpoint cold goes from 4.5 to 5.6 GB/s (-20% wall time) and warm loads of a
// 10 GB checkpoint are 5-8% faster; when the single-eval loader is already disk-bound the
// two are equal. The same technique measured larger wins against older mlx cores, whose
// serial loader did not yet read in file order (ml-explore/mlx-swift-lm#575).
//
// This function is synchronous and blocks its thread until every file is loaded. Do not
// call it from a Swift concurrency cooperative thread; hop to a `DispatchQueue` first.

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// The tensors of the safetensors file at `url`, ordered by their position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorSpansInFileOrder(url: URL) throws -> [SafetensorSpan] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    // a header bigger than this is not a header
    guard headerLength > 0, headerLength <= 512 * 1024 * 1024 else { throw Malformed() }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, byteCount: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = (offsets[0] as? NSNumber)?.int64Value,
            let end = (offsets[1] as? NSNumber)?.int64Value,
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end - begin))
    }
    spans.sort { $0.begin < $1.begin }
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.byteCount) }
}

/// Contiguous index ranges of `byteCounts` whose byte totals are balanced around
/// `total / groupCount`, preserving order.
func contiguousLoadGroups(byteCounts: [Int64], groupCount: Int) -> [Range<Int>] {
    guard !byteCounts.isEmpty else { return [] }
    let total = byteCounts.reduce(0, +)
    guard groupCount > 1, total > 0 else { return [0 ..< byteCounts.count] }

    let groups = Int64(groupCount)
    var ranges = [Range<Int>]()
    var start = 0
    var cumulative: Int64 = 0
    var boundary: Int64 = 1
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= total * boundary / groups {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// How many concurrent evaluations to spread the loading across.
///
/// Throughput rises with concurrent readers until the disk (cold) or the memory system
/// (warm) saturates -- around 8-16 in-flight readers on Apple silicon. More workers than
/// cores only adds contention.
func concurrentLoadWorkerCount(
    processorCount: Int = ProcessInfo.processInfo.activeProcessorCount
) -> Int {
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Lock-guarded shared state for the concurrent load.
private final class ConcurrentLoadState: @unchecked Sendable {
    private let lock = NSLock()
    private var perFile: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var firstError: Error?

    init(fileCount: Int) {
        perFile = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
    }

    func merge(file: Int, arrays: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        perFile[file].merge(arrays) { _, new in new }
        perFileMetadata[file] = metadata
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    /// Arrays and metadata merged in file order -- on a duplicate key the later
    /// file wins, matching a serial load-and-merge loop over the same urls.
    func result() throws -> (arrays: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        var arrays = [String: MLXArray]()
        var metadata = [String: String]()
        for fileArrays in perFile {
            arrays.merge(fileArrays) { _, new in new }
        }
        for fileMetadata in perFileMetadata {
            metadata.merge(fileMetadata) { _, new in new }
        }
        return (arrays, metadata)
    }
}

/// Load a dictionary of ``MLXArray`` from several `safetensors` files at once,
/// evaluating contiguous byte ranges of each file concurrently.
///
/// The dictionaries are merged in file order: on a duplicate name the later file wins,
/// matching a serial load-and-merge loop over the same `urls`. Passing a single url is
/// also worthwhile for a large file -- the file is still split into byte-balanced
/// ranges that are read sequentially and evaluated concurrently.
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
/// `safetensors` files at once, evaluating contiguous byte ranges of each file concurrently.
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
    struct WorkItem {
        let file: Int
        let url: URL
        /// tensors this item evaluates; nil evaluates the whole file
        let names: [String]?
    }

    let items: [WorkItem] = {
        var spansPerFile = [[SafetensorSpan]?]()
        var totalBytes: Int64 = 0
        for url in urls {
            let spans = try? safetensorSpansInFileOrder(url: url)
            spansPerFile.append(spans)
            totalBytes += spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        }

        let workers = concurrentLoadWorkerCount()
        let groupBytes = max(minimumBytesPerLoadGroup, totalBytes / Int64(workers))
        var items = [WorkItem]()
        for (file, url) in urls.enumerated() {
            if let spans = spansPerFile[file], !spans.isEmpty {
                let bytes = spans.reduce(0) { $0 + $1.byteCount }
                let groupCount = max(1, Int(bytes / groupBytes))
                for range in contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount)
                {
                    items.append(
                        WorkItem(file: file, url: url, names: spans[range].map(\.name)))
                }
            } else {
                // header not parseable (or empty) -- let the full loader either load it
                // whole or produce its usual error
                items.append(WorkItem(file: file, url: url, names: nil))
            }
        }
        return items
    }()

    let state = ConcurrentLoadState(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: items.count) { index in
        let item = items[index]
        do {
            let (all, metadata) = try loadArraysAndMetadata(url: item.url, stream: stream)

            var selected = [String: MLXArray]()
            if let names = item.names {
                for name in names {
                    if let array = all[name] { selected[name] = array }
                }
            } else {
                selected = all
            }

            // force this range's I/O here, in file-offset order
            if !selected.isEmpty { eval(selected.values) }
            state.merge(file: item.file, arrays: selected, metadata: metadata)
        } catch {
            state.record(error: error)
        }
    }
    return try state.result()
}
