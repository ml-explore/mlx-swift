// Copyright © 2026 Apple Inc.

import Cmlx
import CoreFoundation
import Foundation

// MARK: - Concurrent safetensors loading

// The safetensors loader is lazy: ``loadArraysAndMetadata(url:stream:)`` reads only the
// header, and a tensor's bytes are read when its array is evaluated. Evaluating a whole
// checkpoint with a single `eval` visits the tensors in dictionary order -- effectively
// random file offsets -- and loses the benefit of sequential reads.
//
// ``loadArrays(urls:stream:)`` instead splits every file into contiguous byte-balanced
// ranges of tensors in file-offset order. It schedules those ranges asynchronously and
// then waits once for the complete checkpoint. MLX therefore owns the read concurrency;
// in particular, all ranges feed its shared reader pipeline (and the adaptive reader pool
// in newer cores) without Swift threads blocking on the global evaluation lock.
// Measured on an M4 Pro (14 cores, NVMe at ~6.1 GB/s sequential), cold throughput for an
// 18 GB checkpoint goes from 4.5 to 5.6 GB/s (-20% wall time), and warm loads of a 10 GB
// checkpoint are 5-8% faster. When the single-eval loader is already disk-bound the two
// are equal. The same technique measured larger wins against older mlx cores, whose
// serial loader did not yet read in file order (ml-explore/mlx-swift-lm#575).
//
// This function is synchronous and blocks its thread until every file is loaded. Do not
// call it from a Swift concurrency cooperative thread; hop to a `DispatchQueue` first.

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// Match the limit enforced by MLX and the safetensors reference implementation.
private let maximumSafetensorHeaderBytes: UInt64 = 100_000_000

/// Decode a JSON integer without accepting booleans, floating-point values, or values
/// outside the range used by the grouping arithmetic below.
private func safetensorOffset(_ value: Any) -> Int64? {
    guard let number = value as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID(),
        !CFNumberIsFloatType(number)
    else {
        return nil
    }
    let offset = number.int64Value
    return offset >= 0 ? offset : nil
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
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.end - $0.begin) }
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
    var threshold = groups.dividingFullWidth(
        total.multipliedFullWidth(by: boundary)
    ).quotient
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= threshold {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
            if boundary < groups {
                threshold =
                    groups.dividingFullWidth(
                        total.multipliedFullWidth(by: boundary)
                    ).quotient
            }
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// Maximum number of independently scheduled, byte-balanced ranges.
///
/// MLX, rather than Swift, owns the threads that execute these ranges and applies its own
/// I/O concurrency cap. Limiting the number of graph submissions here avoids needless
/// scheduler overhead while still exposing enough work to saturate the reader pipeline.
func concurrentLoadGroupCount(
    processorCount: Int = ProcessInfo.processInfo.activeProcessorCount
) -> Int {
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Lock-guarded results of the parallel, header-only preparation pass.
private final class ConcurrentLoadPreparation: @unchecked Sendable {
    private let lock = NSLock()
    private var perFileArrays: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var spansPerFile: [[SafetensorSpan]?]
    private var firstError: Error?

    init(fileCount: Int) {
        perFileArrays = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
        spansPerFile = Array(repeating: nil, count: fileCount)
    }

    func set(
        file: Int,
        arrays: [String: MLXArray],
        metadata: [String: String],
        spans: [SafetensorSpan]?
    ) {
        lock.lock()
        defer { lock.unlock() }
        perFileArrays[file] = arrays
        perFileMetadata[file] = metadata
        spansPerFile[file] = spans
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    func result() throws -> (
        arrays: [[String: MLXArray]],
        metadata: [[String: String]],
        spans: [[SafetensorSpan]?]
    ) {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        return (perFileArrays, perFileMetadata, spansPerFile)
    }
}

/// Load a dictionary of ``MLXArray`` from several `safetensors` files at once,
/// asynchronously scheduling contiguous byte ranges of each file and waiting once.
///
/// The dictionaries are merged in file order: on a duplicate name the later file wins,
/// matching a serial load-and-merge loop over the same `urls`. Passing a single url is
/// also worthwhile for a large file -- the file is still split into byte-balanced
/// ranges that are submitted in file-offset order and read concurrently by MLX.
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
/// `safetensors` files at once, scheduling contiguous byte ranges asynchronously.
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
    // until the ordered asyncEval pass below, where MLX owns their concurrency.
    let preparation = ConcurrentLoadPreparation(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: urls.count) { file in
        let url = urls[file]
        let spans = try? safetensorSpansInFileOrder(url: url)
        do {
            let (arrays, metadata) = try loadArraysAndMetadata(url: url, stream: stream)
            preparation.set(
                file: file, arrays: arrays, metadata: metadata, spans: spans)
        } catch {
            preparation.record(error: error)
        }
    }

    let prepared = try preparation.result()
    let perFileArrays = prepared.arrays
    let perFileMetadata = prepared.metadata
    let spansPerFile = prepared.spans
    let totalBytes = spansPerFile.reduce(Int64(0)) { total, spans in
        let fileBytes = spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        let (sum, overflow) = total.addingReportingOverflow(fileBytes)
        return overflow ? .max : sum
    }
    let groupBytes = max(
        minimumBytesPerLoadGroup,
        totalBytes / Int64(concurrentLoadGroupCount()))

    // Retain every root through the final barrier. Explicit arrays also keep scheduling
    // independent of dictionary iteration order.
    var groups = [[MLXArray]]()

    for file in urls.indices {
        let arrays = perFileArrays[file]
        if let spans = spansPerFile[file], !spans.isEmpty {
            let orderedArrays = spans.compactMap { arrays[$0.name] }
            guard orderedArrays.count == spans.count, spans.count == arrays.count else {
                // The file changed between our planning read and MLX's header read (or the
                // parsers disagreed). Evaluate everything as one group rather than return
                // an array whose I/O was never included in the completion barrier.
                if !arrays.isEmpty { groups.append(Array(arrays.values)) }
                continue
            }
            let bytes = spans.reduce(0) { $0 + $1.byteCount }
            let groupCount = max(1, Int(bytes / groupBytes))
            groups.append(
                contentsOf: contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount
                ).map { range in
                    Array(orderedArrays[range])
                })
        } else {
            // Header not parseable (or empty): the full loader above either produced
            // its usual error or returned arrays that can be scheduled as one group.
            if !arrays.isEmpty {
                groups.append(Array(arrays.values))
            }
        }
    }
    groups.removeAll(where: \.isEmpty)

    // asyncEval takes the global evaluation lock only while submitting work. The last
    // group is evaluated synchronously on the same stream, so it is both the completion
    // barrier for all preceding submissions and the checked Swift error path. Cores that
    // propagate scheduler exceptions also surface deferred I/O errors here. Evaluating
    // only the final group avoids walking every already-scheduled root a second time.
    var submitted = [MLXArray]()
    for group in groups.dropLast() where !group.isEmpty {
        submitted.append(contentsOf: group)
        do {
            try withError {
                asyncEval(group)
            }
        } catch {
            // Drain work submitted before the scheduling error while all readers and roots
            // are still retained, but preserve the first error for the caller.
            if !submitted.isEmpty { try? checkedEval(submitted) }
            throw error
        }
    }
    if let finalGroup = groups.last, !finalGroup.isEmpty {
        try checkedEval(finalGroup)
    }

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
