import Cmlx

public enum DeviceType {
    case cpu
    case gpu
}

public final class Device {
    let ctx: OpaquePointer!
    init(_ ctx_: mlx_device) {
        ctx = ctx_
    }
    public init(_ deviceType : DeviceType, index: Int32 = 0) {
        var cDeviceType : mlx_device_type
        switch(deviceType) {
        case DeviceType.cpu:
            cDeviceType = MLX_CPU
        case DeviceType.gpu:
            cDeviceType = MLX_GPU
        }
        ctx = mlx_device_new(cDeviceType, index)
    }
    public init() {
        ctx = mlx_default_device()
    }
    static public func defaultDevice() -> Device {
        return Device()
    }
    static public func setDefaultDevice(_ dev : Device) {
        mlx_set_default_device(dev.ctx)
    }
    deinit {
        Cmlx.mlx_free(UnsafeMutableRawPointer(ctx))
    }
}
