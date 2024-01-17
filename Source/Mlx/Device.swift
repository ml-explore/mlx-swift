import Foundation
import Cmlx

///Type of device.
///
///See ``Device`` and ``StreamOrDevice``.
public enum DeviceType {
    case cpu
    case gpu
}

/// Representation of a Device in MLX.
///
/// Typically this is used via the `stream: ` parameter on a method with a ``StreamOrDevice``.
///
/// ### See Also
/// - <doc:Using-Streams>
/// - ``StreamOrDevice``
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
    
    deinit {
        mlx_free(ctx)
    }
    
    static public let cpu: Device = Device(.cpu)
    static public let gpu: Device = Device(.gpu)

    static public func defaultDevice() -> Device {
        return Device()
    }
    
    /// Set the default device.
    ///
    /// For example:
    ///
    /// ```swift
    /// Device.setDefault(device: Device(.cpu, index: 1))
    /// ```
    ///
    /// By default this is ``gpu``.
    ///
    /// ### See Also
    /// - ``StreamOrDevice/default``
    static public func setDefault(device : Device) {
        mlx_set_default_device(device.ctx)
    }
    
}

extension Device: CustomStringConvertible {
    public var description: String {
        describeMLX(ctx) ?? String(describing: type(of: self))
    }
}
