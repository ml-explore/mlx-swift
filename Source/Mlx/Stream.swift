import Cmlx

public struct StreamOrDevice {
    
    private let stream: Stream
    
    private init(_ stream: Stream) {
        self.stream = stream
    }
    
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream())
    }
    
    public static func device(_ device: Device) -> StreamOrDevice {
        StreamOrDevice(Stream.defaultStream(device))
    }
    
    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(stream)
    }
    
    var ctx: OpaquePointer {
        stream.ctx
    }
}

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

    // TODO: property?  get/set?
    static public func defaultStream(_ device: Device) -> Stream {
        return Stream(mlx_default_stream(device.ctx))
    }
}
