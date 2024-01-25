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
/// - <doc:Using-Streams>
/// - ``Stream``
/// - ``Device``
public struct StreamOrDevice: CustomStringConvertible {

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
    public static let gpu = device(.gpu)

    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(stream)
    }

    /// Internal context -- used with Cmlx calls.
    public var ctx: OpaquePointer {
        stream.ctx
    }

    public var description: String {
        stream.description
    }
}

/// ### See Also
/// - <doc:Using-Streams>
/// - ``StreamOrDevice``
public final class Stream {
    public let ctx: OpaquePointer!
    init(_ ctx_: mlx_stream) {
        ctx = ctx_
    }

    public init() {
        let dDev = mlx_default_device()!
        ctx = mlx_default_stream(dDev)
        mlx_free(dDev)
    }

    public init(index: Int32, _ device: Device) {
        ctx = mlx_stream_new(index, device.ctx)
    }

    deinit {
        mlx_free(ctx)
    }

    static public func defaultStream(_ device: Device) -> Stream {
        return Stream(mlx_default_stream(device.ctx))
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        mlx_describe(ctx) ?? String(describing: type(of: self))
    }
}
