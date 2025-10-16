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
    let defaultStream: Stream

    init(_ ctx: mlx_device) {
        self.ctx = ctx

        var deviceType = MLX_GPU
        mlx_device_get_type(&deviceType, ctx)
        self.defaultStream =
            switch deviceType {
            case MLX_CPU: .cpu
            case MLX_GPU: .gpu
            default: .gpu
            }
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
        self.defaultStream =
            switch deviceType {
            case .cpu: .cpu
            case .gpu: .gpu
            }
    }

    @available(*, deprecated, message: "please use defaultDevice()")
    public convenience init() {
        var ctx = mlx_device_new()
        mlx_get_default_device(&ctx)
        self.init(ctx)
    }

    deinit {
        mlx_device_free(ctx)
    }

    /// static CPU device
    ///
    /// See ``withDefaultDevice(_:_:)-17vjl``
    static public let cpu: Device = Device(.cpu)

    /// static GPU device
    ///
    /// See ``withDefaultDevice(_:_:)-17vjl``
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

    // support for global default device
    static let _lock = NSLock()
    #if swift(>=5.10)
        nonisolated(unsafe) static var _defaultDevice: Device?
    #else
        static var _defaultDevice: Device?
    #endif

    @TaskLocal static var _tlDefaultDevice = _resolveGlobalDefaultDevice()

    private static func _resolveGlobalDefaultDevice() -> Device {
        _lock.withLock {
            _defaultDevice ?? .gpu
        }
    }

    /// Return the current default device.
    ///
    /// This is used by ``StreamOrDevice/default`` -- the default stream parameter
    /// to most functions.
    static public func defaultDevice() -> Device {
        _tlDefaultDevice
    }

    /// Use a device scoped to a task.
    static public func withDefaultDevice<R>(
        _ device: Device, _ body: () throws -> R
    ) rethrows -> R {
        try $_tlDefaultDevice.withValue(device, operation: body)
    }

    /// Use a device scoped to a task.
    static public func withDefaultDevice<R>(
        _ device: Device, _ body: () async throws -> R
    ) async rethrows -> R {
        try await $_tlDefaultDevice.withValue(device, operation: body)
    }

    /// Return the current default stream.
    static func defaultStream() -> Stream {
        _tlDefaultDevice.defaultStream
    }

    /// Set the default device globally.  Prefer the scoped version, ``withDefaultDevice(_:_:)-17vjl``.
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
    /// - ``withDefaultDevice(_:_:)-17vjl``
    /// - ``StreamOrDevice/default``
    @available(*, deprecated, message: "please use withDefaultDevice()")
    static public func setDefault(device: Device?) {
        _lock.withLock {
            if let device {
                // sets the mlx core default device -- only used
                // by the deprecated init().  this isn't thread
                // safe or really usable across tasks/threads
                // but is kept for backward compatibility
                mlx_set_default_device(device.ctx)
            }
            _defaultDevice = device
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
@available(*, deprecated, message: "please use Device.withDefaultDevice()")
public func using<R>(device: Device, fn: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(device, fn)
}
