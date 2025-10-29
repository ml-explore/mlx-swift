// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

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
            mlx_save(path.cString(using: .utf8), array.ctx)
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
            mlx_save_safetensors(path.cString(using: .utf8), mlx_arrays, mlx_metadata)
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

// MARK: - Memory I/O

private class IOState {
    var offset = 0
    var data = Data()

    internal init(offset: Int = 0, data: Data = Data()) {
        self.offset = offset
        self.data = data
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

    mlx_save_safetensors_writer(writer, mlx_arrays, mlx_metadata)

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
