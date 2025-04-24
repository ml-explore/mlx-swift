// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

///Type of device.
///
///See ``Device`` and ``StreamOrDevice``.
public enum DeviceType: String, Hashable, Sendable {
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
public final class Device: @unchecked Sendable, Equatable {

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
        self.ctx = mlx_device_new_type(cDeviceType, index)
    }

    public init() {
        var ctx = mlx_device_new()
        mlx_get_default_device(&ctx)
        self.ctx = ctx
    }

    deinit {
        mlx_device_free(ctx)
    }

    static public let cpu: Device = Device(.cpu)
    static public let gpu: Device = Device(.gpu)

    public var deviceType: DeviceType? {
        var cDeviceType = MLX_CPU
        mlx_device_get_type(&cDeviceType, ctx)
        return switch cDeviceType {
        case MLX_CPU: DeviceType.cpu
        case MLX_GPU: DeviceType.gpu
        default: nil
        }
    }

    static let _lock = NSLock()
    #if swift(>=5.10)
        nonisolated(unsafe) static var _defaultDevice = gpu
        nonisolated(unsafe) static var _defaultStream = Stream(gpu)
    #else
        static var _defaultDevice = gpu
        static var _defaultStream = Stream(gpu)
    #endif

    static public func defaultDevice() -> Device {
        _lock.withLock {
            _defaultDevice
        }
    }

    static func defaultStream() -> Stream {
        _lock.withLock {
            _defaultStream
        }
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
        _lock.withLock {
            mlx_set_default_device(device.ctx)
            _defaultDevice = device
            _defaultStream = Stream(device)
        }
    }

    /// Compare two ``Device`` for equality -- this does not compare the index, just the device type.
    public static func == (lhs: Device, rhs: Device) -> Bool {
        var lhs_type = MLX_CPU
        var rhs_type = MLX_CPU
        mlx_device_get_type(&lhs_type, lhs.ctx)
        mlx_device_get_type(&rhs_type, rhs.ctx)
        return lhs_type == rhs_type
    }
}

extension Device: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        mlx_device_tostring(&s, ctx)
        defer { mlx_string_free(s) }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}

/// Execute a block of code using a specific device.
///
/// Example:
/// ```swift
/// using(device: .gpu) {
///    // code here will run on the GPU
/// }
/// ```
///
/// - Parameters:
///     - device: device to be used
///     - fn: function to be executed
public func using<R>(device: Device, fn: () throws -> R) rethrows -> R {
    try Stream.withNewDefaultStream(device: device, fn)
}
