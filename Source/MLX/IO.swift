// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Byte-level progress for loading arrays from disk.
///
/// `completedUnitCount` and `totalUnitCount` are bytes. Progress callbacks for a
/// single load are delivered in monotonically increasing order.
public struct LoadProgress: Sendable, Equatable {
    public let completedUnitCount: Int64
    public let totalUnitCount: Int64

    public var fractionCompleted: Double {
        guard totalUnitCount > 0 else { return 0 }
        return min(1, max(0, Double(completedUnitCount) / Double(totalUnitCount)))
    }

    public init(completedUnitCount: Int64, totalUnitCount: Int64) {
        self.completedUnitCount = completedUnitCount
        self.totalUnitCount = totalUnitCount
    }
}

public enum LoadSaveError: Error {
    case unableToOpen(URL, String)
    case unknownExtension(String)
}

extension LoadSaveError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unableToOpen(let url, let message):
            return "\(message) \(url)"
        case .unknownExtension(let fileExtension):
            return "Unknown extension \(fileExtension)"
        }
    }
}

/// Save array to a binary file in `.npy` format.
///
/// - Parameters:
///     - array: array to save
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``save(arrays:metadata:url:stream:)``
/// - ``loadArray(url:stream:)``
/// - ``loadArrays(url:stream:)``
public func save(array: MLXArray, url: URL, stream: StreamOrDevice = .default) throws {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)
    switch url.pathExtension {
    case "npy":
        _ = try withError {
            _ = evalLock.withLock {
                mlx_save(path.cString(using: .utf8), array.ctx)
            }
        }

    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Save dictionary of arrays in `safetensors` format.
///
/// - Parameters:
///     - arrays: array to save
///     - metadata: metadata to save
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``save(arrays:metadata:url:stream:)``
/// - ``loadArray(url:stream:)``
/// - ``loadArrays(url:stream:)``
public func save(
    arrays: [String: MLXArray], metadata: [String: String] = [:], url: URL,
    stream: StreamOrDevice = .default
) throws {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    let mlx_arrays = new_mlx_array_map(arrays)
    defer { mlx_map_string_to_array_free(mlx_arrays) }

    let mlx_metadata = new_mlx_string_map(metadata)
    defer { mlx_map_string_to_string_free(mlx_metadata) }

    switch url.pathExtension {
    case "safetensors":
        _ = try withError {
            _ = evalLock.withLock {
                mlx_save_safetensors(path.cString(using: .utf8), mlx_arrays, mlx_metadata)
            }
        }

    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load array from a binary file in `.npy` format.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``save(array:url:stream:)``
/// - ``save(arrays:metadata:url:stream:)``
public func loadArray(url: URL, stream: StreamOrDevice = .cpu) throws -> MLXArray {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    switch url.pathExtension {
    case "npy":
        var result = mlx_array_new()
        _ = try withError {
            mlx_load(&result, path.cString(using: .utf8), stream.ctx)
        }
        return MLXArray(result)

    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load dictionary of ``MLXArray`` from a `safetensors` file.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArray(url:stream:)``
/// - ``loadArraysAndMetadata(url:stream:)``
/// - ``save(array:url:stream:)``
/// - ``save(arrays:metadata:url:stream:)``
public func loadArrays(url: URL, stream: StreamOrDevice = .cpu) throws -> [String: MLXArray] {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    switch url.pathExtension {
    case "safetensors":
        var r0 = mlx_map_string_to_array_new()
        var r1 = mlx_map_string_to_string_new()
        defer { mlx_map_string_to_array_free(r0) }
        defer { mlx_map_string_to_string_free(r1) }

        _ = try withError {
            mlx_load_safetensors(&r0, &r1, path.cString(using: .utf8), stream.ctx)
        }

        return mlx_map_array_values(r0)
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load dictionary of ``MLXArray`` from a `safetensors` file, reporting byte progress as
/// lazy arrays are evaluated.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///     - progressHandler: progress callback. This may be called from MLX worker threads.
///       Progress is reported in byte chunks while the returned lazy arrays are evaluated.
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``loadArraysAndMetadata(url:stream:progressHandler:)``
public func loadArrays(
    url: URL, stream: StreamOrDevice = .cpu,
    progressHandler: @Sendable @escaping (LoadProgress) -> Void
) throws -> [String: MLXArray] {
    let (arrays, _) = try loadArraysAndMetadata(
        url: url, stream: stream, progressHandler: progressHandler)
    return arrays
}

/// Load dictionary of ``MLXArray`` and metadata `[String:String]` from a `safetensors` file.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``loadArrays(url:stream:)``
/// - ``loadArray(url:stream:)``
public func loadArraysAndMetadata(url: URL, stream: StreamOrDevice = .cpu) throws -> (
    [String: MLXArray], [String: String]
) {
    precondition(url.isFileURL)
    let path = url.path(percentEncoded: false)

    switch url.pathExtension {
    case "safetensors":
        var r0 = mlx_map_string_to_array_new()
        var r1 = mlx_map_string_to_string_new()
        defer { mlx_map_string_to_array_free(r0) }
        defer { mlx_map_string_to_string_free(r1) }

        _ = try withError {
            mlx_load_safetensors(&r0, &r1, path.cString(using: .utf8), stream.ctx)
        }

        return (mlx_map_array_values(r0), mlx_map_string_values(r1))
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

/// Load dictionary of ``MLXArray`` and metadata from a `safetensors` file, reporting byte
/// progress as lazy arrays are evaluated.
///
/// - Parameters:
///     - url: URL of file to load
///     - stream: stream or device to evaluate on
///     - progressHandler: progress callback. This may be called from MLX worker threads.
///       Progress is reported in byte chunks while the returned lazy arrays are evaluated.
///
/// ### See Also
/// - ``loadArraysAndMetadata(url:stream:)``
/// - ``loadArrays(url:stream:progressHandler:)``
public func loadArraysAndMetadata(
    url: URL, stream: StreamOrDevice = .cpu,
    progressHandler: @Sendable @escaping (LoadProgress) -> Void
) throws -> ([String: MLXArray], [String: String]) {
    precondition(url.isFileURL)

    switch url.pathExtension {
    case "safetensors":
        var r0 = mlx_map_string_to_array_new()
        var r1 = mlx_map_string_to_string_new()
        defer { mlx_map_string_to_array_free(r0) }
        defer { mlx_map_string_to_string_free(r1) }

        let reader = try new_mlx_io_reader_fileIO(url, progressHandler: progressHandler)
        defer { mlx_io_reader_free(reader) }

        _ = try withError {
            mlx_load_safetensors_reader(&r0, &r1, reader, stream.ctx)
        }

        return (mlx_map_array_values(r0), mlx_map_string_values(r1))
    default:
        throw LoadSaveError.unknownExtension(url.pathExtension)
    }
}

// MARK: - Memory I/O

private class IOState {
    var offset = 0
    var data = Data()

    internal init(offset: Int = 0, data: Data = Data()) {
        self.offset = offset
        self.data = data
    }
}

private final class FileIOState {
    private static let maximumReadChunkSize = 4 * 1024 * 1024

    private let descriptor: CInt
    private let lock = NSLock()
    private let progressLock = NSLock()
    private var offset: Int64 = 0
    private var completedUnitCount: Int64 = 0
    private var readError: String?
    private let progressHandler: @Sendable (LoadProgress) -> Void
    private let labelPointer: UnsafeMutablePointer<CChar>

    let totalUnitCount: Int64

    init(url: URL, progressHandler: @Sendable @escaping (LoadProgress) -> Void) throws {
        let path = url.path(percentEncoded: false)
        let descriptor = path.withCString { open($0, O_RDONLY) }
        guard descriptor >= 0 else {
            throw LoadSaveError.unableToOpen(url, String(cString: strerror(errno)))
        }

        var statBuffer = stat()
        guard fstat(descriptor, &statBuffer) == 0 else {
            let message = String(cString: strerror(errno))
            close(descriptor)
            throw LoadSaveError.unableToOpen(url, message)
        }

        guard let labelPointer = strdup("file \(path)") else {
            close(descriptor)
            throw LoadSaveError.unableToOpen(url, String(cString: strerror(errno)))
        }

        self.descriptor = descriptor
        self.totalUnitCount = max(0, Int64(statBuffer.st_size))
        self.progressHandler = progressHandler
        self.labelPointer = labelPointer

        progressHandler(.init(completedUnitCount: 0, totalUnitCount: totalUnitCount))
    }

    deinit {
        close(descriptor)
        free(labelPointer)
    }

    var isOpen: Bool {
        descriptor >= 0
    }

    var good: Bool {
        lock.withLock {
            readError == nil
        }
    }

    var label: UnsafePointer<CChar> {
        UnsafePointer(labelPointer)
    }

    func tell() -> Int {
        lock.withLock {
            Int(offset)
        }
    }

    func seek(offset newOffset: Int64, whence: Int32) {
        lock.withLock {
            switch whence {
            case SEEK_SET:
                offset = newOffset
            case SEEK_CUR:
                offset += newOffset
            case SEEK_END:
                offset = totalUnitCount + newOffset
            default:
                break
            }
        }
    }

    func read(to data: UnsafeMutablePointer<CChar>?, count: Int) {
        guard let data else { return }

        let readOffset = lock.withLock {
            offset
        }
        let bytesRead = read(to: data, count: count, offset: readOffset)

        lock.withLock {
            offset += Int64(bytesRead)
        }
    }

    func read(to data: UnsafeMutablePointer<CChar>?, count: Int, offset readOffset: Int64) {
        guard let data else { return }

        _ = read(to: data, count: count, offset: readOffset)
    }

    @discardableResult
    private func read(to data: UnsafeMutablePointer<CChar>, count: Int, offset readOffset: Int64)
        -> Int
    {
        var totalRead = 0
        while totalRead < count {
            let chunkSize = min(count - totalRead, Self.maximumReadChunkSize)
            let bytesRead = pread(
                descriptor,
                UnsafeMutableRawPointer(data.advanced(by: totalRead)),
                chunkSize,
                off_t(readOffset + Int64(totalRead)))
            guard bytesRead > 0 else {
                recordReadError(bytesRead: bytesRead, requestedCount: count - totalRead)
                break
            }
            totalRead += bytesRead
            reportProgress(bytesRead: bytesRead)
        }

        return totalRead
    }

    private func recordReadError(bytesRead: Int, requestedCount: Int) {
        let message: String
        if bytesRead < 0 {
            message = String(cString: strerror(errno))
        } else {
            message = "unexpected end of file while reading \(requestedCount) bytes"
        }
        lock.withLock {
            if readError == nil {
                readError = message
            }
        }
    }

    private func reportProgress(bytesRead: Int) {
        guard bytesRead > 0 else { return }

        progressLock.withLock {
            completedUnitCount = min(totalUnitCount, completedUnitCount + Int64(bytesRead))
            let progress = LoadProgress(
                completedUnitCount: completedUnitCount,
                totalUnitCount: totalUnitCount)
            progressHandler(progress)
        }
    }
}

private let label: StaticString = "<memory IO stream>\0"

private func getData(_ writer: mlx_io_writer) -> Data {
    var ptr: UnsafeMutableRawPointer?
    mlx_io_writer_descriptor(&ptr, writer)
    let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()
    return state.data
}

private func new_mlx_io_vtable_dataIO() -> mlx_io_vtable {
    mlx_io_vtable { ptr in
        ptr != nil
    } good: { ptr in
        true
    } tell: { ptr in
        let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()
        return state.offset

    } seek: { ptr, offset, whence in
        let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()

        switch whence {
        case SEEK_SET:
            state.offset = Int(offset)
        case SEEK_CUR:
            state.offset += Int(offset)
        case SEEK_END:
            state.offset = state.offset - Int(offset)
        default:
            break
        }
    } read: { ptr, data, n in
        let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()

        if n + state.offset <= state.data.count {
            guard let data = data else { return }
            _ = state.data.withUnsafeBytes { buffer in
                memcpy(data, buffer.baseAddress!.advanced(by: state.offset), n)
            }
            state.offset += n
        }

    } read_at_offset: { ptr, data, n, offset in
        let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()

        if n + offset <= state.data.count {
            guard let data = data else { return }
            _ = state.data.withUnsafeBytes { buffer in
                memcpy(data, buffer.baseAddress!.advanced(by: offset), n)
            }
            state.offset = offset
        }

    } write: { ptr, data, n in
        let state = Unmanaged<IOState>.fromOpaque(ptr!).takeUnretainedValue()

        let buffer = UnsafeBufferPointer(start: data, count: n)
        state.data.append(buffer)
        state.offset += n

    } label: { ptr in
        UnsafeRawPointer(label.utf8Start).assumingMemoryBound(to: Int8.self)

    } free: { ptr in
        Unmanaged<IOState>.fromOpaque(ptr!).release()
    }
}

private func new_mlx_io_reader_dataIO(_ data: Data) -> mlx_io_reader {
    let ptr = Unmanaged.passRetained(IOState(data: data)).toOpaque()
    return mlx_io_reader_new(ptr, new_mlx_io_vtable_dataIO())
}

private func new_mlx_io_vtable_fileIO() -> mlx_io_vtable {
    mlx_io_vtable { ptr in
        guard let ptr else { return false }
        return Unmanaged<FileIOState>.fromOpaque(ptr).takeUnretainedValue().isOpen
    } good: { ptr in
        guard let ptr else { return false }
        let state = Unmanaged<FileIOState>.fromOpaque(ptr).takeUnretainedValue()
        return state.isOpen && state.good
    } tell: { ptr in
        let state = Unmanaged<FileIOState>.fromOpaque(ptr!).takeUnretainedValue()
        return state.tell()

    } seek: { ptr, offset, whence in
        let state = Unmanaged<FileIOState>.fromOpaque(ptr!).takeUnretainedValue()
        state.seek(offset: Int64(offset), whence: whence)

    } read: { ptr, data, n in
        let state = Unmanaged<FileIOState>.fromOpaque(ptr!).takeUnretainedValue()
        state.read(to: data, count: n)

    } read_at_offset: { ptr, data, n, offset in
        let state = Unmanaged<FileIOState>.fromOpaque(ptr!).takeUnretainedValue()
        state.read(to: data, count: n, offset: Int64(offset))

    } write: { _, _, _ in

    } label: { ptr in
        let state = Unmanaged<FileIOState>.fromOpaque(ptr!).takeUnretainedValue()
        return state.label

    } free: { ptr in
        Unmanaged<FileIOState>.fromOpaque(ptr!).release()
    }
}

private func new_mlx_io_reader_fileIO(
    _ url: URL, progressHandler: @Sendable @escaping (LoadProgress) -> Void
) throws -> mlx_io_reader {
    let ptr = Unmanaged.passRetained(try FileIOState(url: url, progressHandler: progressHandler))
        .toOpaque()
    return mlx_io_reader_new(ptr, new_mlx_io_vtable_fileIO())
}

private func new_mlx_io_writer_dataIO() -> mlx_io_writer {
    let ptr = Unmanaged.passRetained(IOState()).toOpaque()
    return mlx_io_writer_new(ptr, new_mlx_io_vtable_dataIO())
}

/// Save dictionary of arrays in `safetensors` format into `Data`.
///
/// - Parameters:
///     - arrays: arrays to save
///     - metadata: metadata to save
///
/// ### See Also
/// - ``save(arrays:metadata:url:stream:)``
/// - ``loadArrays(data:stream:)``
/// - ``loadArraysAndMetadata(data:stream:)``
public func saveToData(
    arrays: [String: MLXArray], metadata: [String: String] = [:]
) throws -> Data {
    let mlx_arrays = new_mlx_array_map(arrays)
    defer { mlx_map_string_to_array_free(mlx_arrays) }

    let mlx_metadata = new_mlx_string_map(metadata)
    defer { mlx_map_string_to_string_free(mlx_metadata) }

    let writer = new_mlx_io_writer_dataIO()
    defer { mlx_io_writer_free(writer) }

    _ = try withError {
        _ = evalLock.withLock {
            mlx_save_safetensors_writer(writer, mlx_arrays, mlx_metadata)
        }
    }

    return getData(writer)
}

/// Load dictionary of ``MLXArray`` from a `safetensors` `Data`.
///
/// - Parameters:
///     - data: `Data` to load from
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``saveToData(arrays:metadata:)``
/// - ``loadArraysAndMetadata(data:stream:)``
public func loadArrays(data: Data, stream: StreamOrDevice = .cpu) throws -> [String: MLXArray] {
    let reader = new_mlx_io_reader_dataIO(data)
    defer { mlx_io_reader_free(reader) }

    var r0 = mlx_map_string_to_array_new()
    var r1 = mlx_map_string_to_string_new()
    defer { mlx_map_string_to_array_free(r0) }
    defer { mlx_map_string_to_string_free(r1) }

    _ = try withError {
        mlx_load_safetensors_reader(&r0, &r1, reader, stream.ctx)
    }

    return mlx_map_array_values(r0)
}

/// Load dictionary of ``MLXArray`` and metadata from a `safetensors` `Data`.
///
/// - Parameters:
///     - data: `Data` to load from
///     - stream: stream or device to evaluate on
///
/// ### See Also
/// - ``saveToData(arrays:metadata:)``
/// - ``loadArrays(data:stream:)``
public func loadArraysAndMetadata(data: Data, stream: StreamOrDevice = .cpu) throws -> (
    [String: MLXArray], [String: String]
) {
    let reader = new_mlx_io_reader_dataIO(data)
    defer { mlx_io_reader_free(reader) }

    var r0 = mlx_map_string_to_array_new()
    var r1 = mlx_map_string_to_string_new()
    defer { mlx_map_string_to_array_free(r0) }
    defer { mlx_map_string_to_string_free(r1) }

    _ = try withError {
        mlx_load_safetensors_reader(&r0, &r1, reader, stream.ctx)
    }

    return (mlx_map_array_values(r0), mlx_map_string_values(r1))
}
