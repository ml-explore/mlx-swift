// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// ```swift
/// // produced on cpu
/// let a = MLXRandom.uniform([100, 100], stream: .cpu)
///
/// // produced on gpu
/// let b = MLXRandom.uniform([100, 100], stream: .gpu)
/// ```
///
/// If omitted it will use the ``default``, which will be ``Device/gpu`` unless
/// set otherwise.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``Stream``
/// - ``Device``
public struct StreamOrDevice: Sendable, CustomStringConvertible, Equatable {

    public let stream: Stream

    private init(_ stream: Stream) {
        self.stream = stream
    }

    /// The default stream on the default device.
    ///
    /// This will be ``Device/gpu`` unless ``Device/setDefault(device:)``
    /// sets it otherwise.
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream.defaultStream ?? Device.defaultStream())
    }

    public static func device(_ device: Device) -> StreamOrDevice {
        StreamOrDevice(Stream.defaultStream(device))
    }

    /// The ``Stream/defaultStream(_:)`` on the ``Device/cpu``
    public static let cpu = device(.cpu)

    /// The ``Stream/defaultStream(_:)`` on the ``Device/gpu``
    ///
    /// ### See Also
    /// - ``GPU``
    public static let gpu = device(.gpu)

    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(Device.defaultStream())
    }

    /// Internal context -- used with Cmlx calls.
    public var ctx: mlx_stream {
        stream.ctx
    }

    public var description: String {
        stream.description
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically this is used via the `stream: ` parameter on a method with a ``StreamOrDevice``:
///
/// ```swift
/// let a: MLXArray ...
/// let result = sqrt(a, stream: .gpu)
/// ```
///
/// Read more at <doc:using-streams>.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``StreamOrDevice``
public final class Stream: @unchecked Sendable, Equatable {

    let ctx: mlx_stream

    public static let gpu = Stream(mlx_default_gpu_stream_new())
    public static let cpu = Stream(mlx_default_cpu_stream_new())

    @TaskLocal static var defaultStream: Stream?

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(device: Device? = nil, _ body: () throws -> R)
        rethrows -> R
    {
        let device = device ?? Device.defaultDevice()
        return try $defaultStream.withValue(Stream(device), operation: body)
    }

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(
        device: Device? = nil, _ body: () async throws -> R
    ) async rethrows -> R {
        let device = device ?? Device.defaultDevice()
        return try await $defaultStream.withValue(Stream(device), operation: body)
    }

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    /// Default stream on the default device.
    public init() {
        let device = Device.defaultDevice()
        var ctx = mlx_stream_new()
        mlx_get_default_stream(&ctx, device.ctx)
        self.ctx = ctx
    }

    @available(*, deprecated, message: "use init(Device) -- index not supported")
    public init(index: Int32, _ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    /// New stream on the given device.
    ///
    /// See also ``withNewDefaultStream(device:_:)-5bwc3``
    public init(_ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    deinit {
        _ = evalLock.withLock {
            mlx_stream_free(ctx)
        }
    }

    /// Synchronize with the given stream
    public func synchronize() {
        _ = evalLock.withLock {
            mlx_synchronize(ctx)
        }
    }

    static public func defaultStream(_ device: Device) -> Stream {
        switch device.deviceType {
        case .cpu: .cpu
        case .gpu: .gpu
        default: fatalError("Unexpected device type: \(device)")
        }
    }

    public static func == (lhs: Stream, rhs: Stream) -> Bool {
        mlx_stream_equal(lhs.ctx, rhs.ctx)
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        mlx_stream_tostring(&s, ctx)
        defer { mlx_string_free(s) }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
