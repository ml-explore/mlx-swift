import Cmlx

public enum StreamOrDevice {
    case `default`
    case device(Device)
    case stream(Stream)
    
    public var stream: Stream {
        switch self {
        case .default:
            Stream()
        case .device(let device):
            Stream.defaultStream(device)
        case .stream(let stream):
            stream
        }
    }
    
    @inlinable
    func withStream<T>(body: (OpaquePointer) throws -> T) rethrows -> T {
        let s = stream
        return try withExtendedLifetime(s) {
            try body(s.ctx)
        }
    }
}

public final class Stream {
    public let ctx: OpaquePointer!
    init(_ ctx_: mlx_stream) {
        ctx = ctx_
    }
    public init() {
        // TODO: does this work?  i am not sure on that mlx_free.  yes, a bunch of typedefs mean it is mlx_device_
        let dDev = mlx_default_device()
        ctx = mlx_default_stream(dDev)
        mlx_free(UnsafeMutableRawPointer(dDev))
    }
    public init(index: Int32, _ device: Device) {
        ctx = mlx_stream_new(index, device.ctx)
    }
    // TODO: property?  get/set?
    static public func defaultStream(_ device: Device) -> Stream {
        return Stream(mlx_default_stream(device.ctx))
    }
    deinit {
        Cmlx.mlx_free(UnsafeMutableRawPointer(ctx))
    }
}
