// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

///Type of device.
///
///See ``Device`` and ``StreamOrDevice``.
public enum DeviceType: String, Hashable, Sendable {
    case cpu
    case gpu

    var cDeviceType: mlx_device_type {
        switch self {
        case .cpu: return MLX_CPU
        case .gpu: return MLX_GPU
        }
    }

    init(_ cDeviceType: mlx_device_type) {
        switch cDeviceType {
        case MLX_CPU: self = .cpu
        case MLX_GPU: self = .gpu
        default: fatalError("Unknown deviceType: \(cDeviceType)")
        }
    }
}

/// Representation of a Device in MLX.
///
/// This is typically used with ``withDefaultDevice(_:_:)-17vjl`` or ``Stream/withNewDefaultStream(device:_:)-5bwc3``.
///
/// ```swift
/// Device.withDefaultDevice(.cpu) {
///     // default device is cpu inside this scope/task
/// }
///
/// Stream.withNewDefaultStream(device: .cpu) {
///     // default device is cpu inside this scope/task AND there
///     // is a new Stream scoped to the work
/// }
/// ```
///
/// Implementation: ``Stream`` hold the current device as a Task local variable.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``StreamOrDevice``
/// - ``Stream``
public final class Device: @unchecked Sendable, Hashable {

    let ctx: mlx_device

    init(_ ctx: mlx_device) {
        self.ctx = ctx
    }

    /// Initialize a new `Device` with a type and optional index.
    ///
    /// If `deviceType` is ``DeviceType/cpu`` then index _must_ be 0.
    public init(_ deviceType: DeviceType, index: Int32 = 0) {
        if deviceType == .cpu {
            precondition(index == 0)
        }
        let cDeviceType = deviceType.cDeviceType
        self.ctx = mlx_device_new_type(cDeviceType, index)
    }

    @available(*, deprecated, message: "please use defaultDevice()")
    public convenience init() {
        self.init(copying: Stream.defaultDevice)
    }

    convenience init(stream: mlx_stream) {
        var ctx = mlx_device_new()
        mlx_stream_get_device(&ctx, stream)
        self.init(ctx)
    }

    convenience init(copying device: Device) {
        var ctx = mlx_device_new()
        mlx_device_set(&ctx, device.ctx)
        self.init(ctx)
    }

    deinit {
        mlx_device_free(ctx)
    }

    /// Current CPU device.
    ///
    /// Note: previously this returned a `static` CPU device.
    ///
    /// ### See Also
    /// - ``gpu``
    /// - ``Stream/cpu``
    /// - ``withDefaultDevice(_:_:)-17vjl``
    static public var cpu: Device {
        Stream.cpu.device
    }

    /// Current GPU device.
    ///
    /// Note: previously this returned a `static` GPU device -- it would
    /// always be index 0, even if there were multiple GPUs in the system.
    ///
    /// ### See Also
    /// - ``cpu``
    /// - ``Stream/gpu``
    /// - ``withDefaultDevice(_:_:)-17vjl``
    static public var gpu: Device {
        Stream.gpu.device
    }

    /// The ``DeviceType`` for the device.
    ///
    /// Note: previously this returned an Optional ``DeviceType``.  Now it is
    /// not optional and it will be a `fatalError` if the DeviceType is unknown.
    public var deviceType: DeviceType {
        var cDeviceType = MLX_CPU
        mlx_device_get_type(&cDeviceType, ctx)
        return DeviceType(cDeviceType)
    }

    /// Return the current default device.
    static public func defaultDevice() -> Device {
        Stream.defaultDevice
    }

    /// Use a device scoped to a Task.
    ///
    /// This can be used to set the default device to e.g. the CPU:
    ///
    /// ```swift
    /// Device.withDefaultDevice(.cpu) {
    ///     // default device is cpu inside this scope/task
    /// }
    /// ```
    ///
    /// If a GPU device is given and it does not match the current GPU it will
    /// override the default device and provide a new stream:
    ///
    /// ```swift
    /// Device.withDefaultDevice(.init(.gpu, index: 3)) {
    ///     // if the enclosing GPU was index 0, this will have
    ///     // both a new default device and a new stream
    /// }
    /// ```
    ///
    /// See also ``Stream/withNewDefaultStream(device:_:)-5bwc3``.
    static public func withDefaultDevice<R>(
        _ device: Device, _ body: () throws -> R
    ) rethrows -> R {
        try Stream.withNewDefaultDevice(device: device, body)
    }

    /// Use a device scoped to a Task.
    ///
    /// This can be used to set the default device to e.g. the CPU:
    ///
    /// ```swift
    /// Device.withDefaultDevice(.cpu) {
    ///     // default device is cpu inside this scope/task
    /// }
    /// ```
    ///
    /// If a GPU device is given and it does not match the current GPU it will
    /// override the default device and provide a new stream:
    ///
    /// ```swift
    /// Device.withDefaultDevice(.init(.gpu, index: 3)) {
    ///     // if the enclosing GPU was index 0, this will have
    ///     // both a new default device and a new stream
    /// }
    /// ```
    ///
    /// See also ``Stream/withNewDefaultStream(device:_:)-5bwc3``.
    static public func withDefaultDevice<R>(
        _ device: Device, _ body: () async throws -> R
    ) async rethrows -> R {
        try await Stream.withNewDefaultDevice(device: device, body)
    }

    /// Set the default device globally.  Use the scoped version, ``withDefaultDevice(_:_:)-17vjl``
    /// -- this is only usable before any mlx resources are created.
    ///
    /// Beware: using the static cpu and gpu values will render this unusable.  If required, call like this:
    ///
    /// ```swift
    /// Device.setDefault(device: .init(.cpu))
    /// ```
    @available(
        *, deprecated,
        message: "use withDefaultDevice() -- this only works before any mlx resources are created"
    )
    static public func setDefault(device: Device?) {
        if let ctx = device?.ctx {
            mlx_set_default_device(ctx)
        }
    }

    /// Compare two ``Device`` for equality
    public static func == (lhs: Device, rhs: Device) -> Bool {
        mlx_device_equal(lhs.ctx, rhs.ctx)
    }

    public func hash(into hasher: inout Hasher) {
        var index: Int32 = 0
        mlx_device_get_index(&index, ctx)
        hasher.combine(index)

        var type: mlx_device_type = MLX_CPU
        mlx_device_get_type(&type, ctx)
        hasher.combine(type.rawValue)
    }
}

extension Device: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        defer { mlx_string_free(s) }
        _ = withEvalLock {
            mlx_device_tostring(&s, ctx)
        }
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
