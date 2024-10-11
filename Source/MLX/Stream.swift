// Copyright © 2024 Apple Inc.

import Cmlx

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

    private let stream: Stream

    private init(_ stream: Stream) {
        self.stream = stream
    }

    /// The default stream on the default device.
    ///
    /// This will be ``Device/gpu`` unless ``Device/setDefault(device:)``
    /// sets it otherwise.
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream())
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
        StreamOrDevice(stream)
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

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    public init() {
        let device = mlx_default_device()
        ctx = mlx_default_stream(device)
        mlx_device_free(device)
    }

    public init(index: Int32, _ device: Device) {
        ctx = mlx_stream_new(index, device.ctx)
    }

    deinit {
        mlx_stream_free(ctx)
    }

    /// Synchronize with the given stream
    public func synchronize() {
        mlx_synchronize(ctx)
    }

    static public func defaultStream(_ device: Device) -> Stream {
        return Stream(mlx_default_stream(device.ctx))
    }

    public static func == (lhs: Stream, rhs: Stream) -> Bool {
        mlx_stream_equal(lhs.ctx, rhs.ctx)
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        ""
        // TODO
        //        let s = mlx_stream_tostring(ctx)
        //        defer { mlx_string_free(s) }
        //        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
