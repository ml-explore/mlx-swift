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
