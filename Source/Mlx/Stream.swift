import Cmlx

public final class Stream {
    let ctx: OpaquePointer!
    init(_ ctx_: mlx_stream) {
        ctx = ctx_
    }
    public init() {
        let dDev = mlx_default_device()
        ctx = mlx_default_stream(dDev)
        mlx_free(UnsafeMutableRawPointer(dDev))
    }
    public init(index: Int32, _ device: Device) {
        ctx = mlx_stream_new(index, device.ctx);
    }
    static public func defaultStream(_ device: Device) -> Stream {
        return Stream(mlx_default_stream(device.ctx))
    }
    deinit {
        Cmlx.mlx_free(UnsafeMutableRawPointer(ctx))
    }
}
