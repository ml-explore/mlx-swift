// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

///Type of device.
///
///See ``Device`` and ``StreamOrDevice``.
public enum DeviceType: String {
    case cpu
    case gpu
}

/// Representation of a Device in MLX.
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
public final class Device {

    let ctx: mlx_device

    init(_ ctx: mlx_device) {
        self.ctx = ctx
    }

    public init(_ deviceType: DeviceType, index: Int32 = 0) {
        var cDeviceType: mlx_device_type
        switch deviceType {
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
    static public func setDefault(device: Device) {
        mlx_set_default_device(device.ctx)
    }

}

extension Device: CustomStringConvertible {
    public var description: String {
        mlx_describe(ctx) ?? String(describing: type(of: self))
    }
}
