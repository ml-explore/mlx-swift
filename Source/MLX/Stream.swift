// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// ```swift
/// // produced on cpu
/// let a = MLXRandom.uniform([100, 100], stream: .cpu)
///
/// // produced on gpu
/// let b = MLXRandom.uniform([100, 100], stream: .gpu)
/// ```
///
/// If omitted it will use the ``default``, which will be ``Device/gpu`` unless
/// set otherwise.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``Stream``
/// - ``Device``
public struct StreamOrDevice: Sendable, CustomStringConvertible, Equatable {

    public let stream: Stream

    private init(_ stream: Stream) {
        self.stream = stream
    }

    /// The default stream on the default device.
    ///
    /// See ``Stream/defaultStream``.
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream.defaultStream)
    }

    /// Evaluate on the given ``Device``.
    ///
    /// Prefer ``cpu`` and ``gpu`` instead as they will use the current Device for the
    /// Task.
    @available(*, deprecated, message: "prefer stream(Stream)")
    public static func device(_ device: Device) -> StreamOrDevice {
        StreamOrDevice(Stream.defaultStream(device))
    }

    /// Evaluate on the current CPU stream, ``Stream/cpu``.
    public static var cpu: StreamOrDevice { .stream(.cpu) }

    /// Evaluate on the current GPU stream, ``Stream/gpu``.
    public static var gpu: StreamOrDevice { .stream(.gpu) }

    /// Evaluate on the given stream.
    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(stream)
    }

    /// Internal context -- used with Cmlx calls.
    public var ctx: mlx_stream {
        stream.ctx
    }

    public var description: String {
        stream.description
    }
}

/// Pool for recycling ``Stream`` (`mlx_stream`).
///
/// See also https://github.com/ml-explore/mlx/issues/2118 -- there are
/// metal/OS resources associated with a GPU stream that may never be collected.
/// Reuse the streams.
///
/// `Stream.deinit` will return the underlying `mlx_stream` to the pool.
private final class StreamPool: @unchecked Sendable {
    private var idle: [Device: [mlx_stream]] = [:]
    private let lock = NSLock()

    init() {
    }

    func newStream(_ device: Device) -> mlx_stream {
        // do not hold the evalLock at the same time as the pool lock
        let (valid, stream) = lock.withLock {
            if let value = idle[device, default: []].popLast() {
                return (true, value)
            }
            return (false, mlx_stream())
        }
        if valid {
            return stream
        }

        return withEvalLock {
            let result = mlx_stream_new_thread_unsafe(device.ctx)
            if result.ctx == nil {
                fatalError("Unable to create stream: \(device)")
            }
            return result
        }
    }

    func releaseStream(_ stream: mlx_stream) {
        let device = Device(stream: stream)

        lock.withLock {
            idle[device, default: []].append(stream)
        }
    }
}

/// global stream pool
private let streamPool = StreamPool()

private protocol StreamPromise: Sendable {
    var current: Stream { get }
    var device: Device { get }
}

private final class LazyStreamPromise: StreamPromise, @unchecked Sendable {
    private var stream: Stream?
    private let lock = NSLock()

    let device: Device

    init(_ device: Device) {
        self.stream = nil
        self.device = device
    }

    var current: Stream {
        if let stream = lock.withLock({ self.stream }) {
            return stream
        }

        // double checked lock to avoid holding a lock outside evalLock
        let stream = Stream(streamPool.newStream(device))

        return lock.withLock {
            if let current = self.stream {
                return current
            } else {
                self.stream = stream
                return stream
            }
        }
    }
}

private final class RealizedStreamPromise: StreamPromise, Sendable {
    let current: Stream
    let device: Device

    init(_ device: Device) {
        self.device = device
        self.current = Stream(streamPool.newStream(device))
    }

    init(_ promise: StreamPromise) {
        self.device = promise.device
        self.current = promise.current
    }
}

/// TaskLocal promise for a pair of Streams -- one for CPU and one for GPU.
///
/// Implementation note: the default device will be a realized Stream and the
/// none default device will be a lazy promise.
private final class StreamPairPromise: Sendable {
    let cpu: StreamPromise
    let gpu: StreamPromise
    let defaultDeviceType: DeviceType

    var defaultStream: StreamPromise {
        switch defaultDeviceType {
        case .cpu: cpu
        case .gpu: gpu
        }
    }

    /// Initialize the global promise.
    init() {
        var default_ctx = mlx_device_new()
        mlx_get_default_device(&default_ctx)

        let defaultDevice = Device(default_ctx)
        self.defaultDeviceType = defaultDevice.deviceType
        if defaultDevice.deviceType == .cpu {
            self.cpu = RealizedStreamPromise(defaultDevice)
            self.gpu = LazyStreamPromise(Device(.gpu))
        } else {
            self.cpu = LazyStreamPromise(Device(.cpu))
            self.gpu = RealizedStreamPromise(defaultDevice)
        }
    }

    /// Initialize with new streams.
    ///
    /// See ``Stream/withNewDefaultStream(device:_:)``.
    init(cpu: Device, gpu: Device, defaultDeviceType: DeviceType) {
        self.defaultDeviceType = defaultDeviceType
        if defaultDeviceType == .cpu {
            self.cpu = RealizedStreamPromise(cpu)
            self.gpu = LazyStreamPromise(gpu)
        } else {
            self.cpu = LazyStreamPromise(cpu)
            self.gpu = RealizedStreamPromise(gpu)
        }
    }

    /// Initialize with a new default device and a parent.
    ///
    /// See ``Device/withDefaultDevice(_:_:)``.
    init(parent: StreamPairPromise, device: Device) {
        self.defaultDeviceType = device.deviceType

        switch self.defaultDeviceType {
        case .cpu:
            self.cpu = RealizedStreamPromise(parent.cpu)
            self.gpu = parent.gpu

        case .gpu:
            self.cpu = parent.cpu
            if parent.gpu.device == device {
                self.gpu = RealizedStreamPromise(parent.gpu)
            } else {
                self.gpu = RealizedStreamPromise(device)
            }
        }
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically this is used via the `stream:` parameter on a method with a ``StreamOrDevice``:
///
/// ```swift
/// let a: MLXArray ...
/// let result = sqrt(a, stream: .gpu)
/// ```
///
/// Read more at <doc:using-streams>.
///
/// ## Implementation Notes
///
/// There is a global pair of CPU/GPU streams.  On devices with a GPU (and associated GPU software)
/// the default stream will be a GPU stream.
///
/// The default streams can be overridden and scoped to a `Task` using:
///
/// ```swift
/// Stream.withNewDefaultStream {
///     // the default stream is private to this Task
/// }
/// ```
///
/// Callers can override the default device by passing a device to that method or using
/// ``Device/withDefaultDevice(_:_:)-17vjl`` to override the default device without
/// getting a new (private) default stream.
///
/// Also note that the current ``Device/defaultDevice()`` is owned by ``Stream/defaultStream``.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``StreamOrDevice``
/// - ``Device``
public final class Stream: @unchecked Sendable, Equatable {

    /// reference to the backing C++ stream object
    let ctx: mlx_stream

    /// the global cpu and gpu streams
    private static let globalStreams = StreamPairPromise()

    /// the task local override streams, see ``withNewDefaultStream(device:_:)``
    @TaskLocal private static var localStreams: StreamPairPromise?

    /// the current GPU stream
    public static var gpu: Stream {
        localStreams?.gpu.current ?? globalStreams.gpu.current
    }

    /// the current CPU stream
    public static var cpu: Stream {
        localStreams?.cpu.current ?? globalStreams.cpu.current
    }

    /// the current default stream, see ``StreamOrDevice`` -- if no stream is
    /// specified, this will be the stream that all operations run on.
    public static var defaultStream: Stream {
        localStreams?.defaultStream.current ?? globalStreams.defaultStream.current
    }

    static var defaultDevice: Device {
        let streams = localStreams ?? globalStreams
        return streams.defaultStream.device
    }

    var device: Device { Device(stream: ctx) }
    var index: Int {
        var index: Int32 = 0
        mlx_stream_get_index(&index, ctx)
        return Int(index)
    }

    /// Obtain a new CPU and GPU stream scoped to the block and inherited through `Tasks`.
    ///
    /// For example you can get a new stream on the current device:
    ///
    /// ```swift
    /// Stream.withNewDefaultStream {
    ///     // A new Stream on the default Device scoped to the block and Task
    /// }
    /// ```
    ///
    /// Or a new stream on a different device (it will become the default device, scoped to the block
    /// and Task):
    ///
    /// ```swift
    /// Stream.withNewDefaultStream(device: .cpu) {
    ///     // A new Stream on the cpu Device scoped to the block and Task.
    ///     // The cpu is now the default device.
    /// }
    /// ```
    ///
    /// These calls can be nested arbitrarily.
    ///
    /// See also ``Device/withDefaultDevice(_:_:)-17vjl`` which can change the default device
    /// without creating new scoped Streams.
    public static func withNewDefaultStream<R>(
        device: Device? = nil, _ body: () throws -> R
    ) rethrows -> R {
        let streams = localStreams ?? globalStreams
        let device = device ?? streams.defaultStream.device
        let cpu = device.deviceType == .cpu ? device : streams.cpu.device
        let gpu = device.deviceType == .gpu ? device : streams.gpu.device
        return try $localStreams.withValue(
            StreamPairPromise(cpu: cpu, gpu: gpu, defaultDeviceType: device.deviceType),
            operation: body)
    }

    /// Obtain a new CPU and GPU stream scoped to the block and inherited through `Tasks`.
    ///
    /// For example you can get a new stream on the current device:
    ///
    /// ```swift
    /// Stream.withNewDefaultStream {
    ///     // A new Stream on the default Device scoped to the block and Task
    /// }
    /// ```
    ///
    /// Or a new stream on a different device (it will become the default device, scoped to the block
    /// and Task):
    ///
    /// ```swift
    /// Stream.withNewDefaultStream(device: .cpu) {
    ///     // A new Stream on the cpu Device scoped to the block and Task.
    ///     // The cpu is now the default device.
    /// }
    /// ```
    ///
    /// These calls can be nested arbitrarily.
    ///
    /// See also ``Device/withDefaultDevice(_:_:)-17vjl`` which can change the default device
    /// without creating new scoped Streams.
    public static func withNewDefaultStream<R>(
        device: Device? = nil, _ body: () async throws -> R
    ) async rethrows -> R {
        let streams = localStreams ?? globalStreams
        let device = device ?? streams.defaultStream.device
        let cpu = device.deviceType == .cpu ? device : streams.cpu.device
        let gpu = device.deviceType == .gpu ? device : streams.gpu.device
        return try await $localStreams.withValue(
            StreamPairPromise(cpu: cpu, gpu: gpu, defaultDeviceType: device.deviceType),
            operation: body)
    }

    /// Implementation for ``Device/withDefaultDevice(_:_:)``
    static func withNewDefaultDevice<R>(
        device: Device, _ body: () throws -> R
    ) rethrows -> R {
        let pair = StreamPairPromise(
            parent: localStreams ?? globalStreams, device: device)
        return try $localStreams.withValue(pair, operation: body)
    }

    /// Implementation for ``Device/withDefaultDevice(_:_:)``
    static func withNewDefaultDevice<R>(
        device: Device, _ body: () async throws -> R
    ) async rethrows -> R {
        let pair = StreamPairPromise(
            parent: localStreams ?? globalStreams, device: device)
        return try await $localStreams.withValue(pair, operation: body)
    }

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    /// New stream on the default device.
    @available(*, deprecated, message: "use withNewDefaultStream instead")
    public init() {
        self.ctx = streamPool.newStream(Device.defaultDevice())
    }

    /// New stream on the given device.
    ///
    /// Note: `index` is unused.
    @available(*, deprecated, message: "use withNewDefaultStream instead")
    public init(index: Int32, _ device: Device) {
        self.ctx = streamPool.newStream(device)
    }

    /// New stream on the given device.
    ///
    /// Prefer ``withNewDefaultStream(device:_:)-5bwc3``
    public init(_ device: Device) {
        self.ctx = streamPool.newStream(device)
    }

    deinit {
        streamPool.releaseStream(ctx)
    }

    /// Synchronize with the given stream
    public func synchronize() {
        _ = withEvalLock {
            mlx_synchronize(ctx)
        }
    }

    @_disfavoredOverload
    @available(
        *, deprecated, message: "use defaultStream(DeviceType) or withNewDefaultStream(Device)"
    )
    static public func defaultStream(_ device: Device) -> Stream {
        switch device.deviceType {
        case .cpu: return .cpu
        case .gpu:
            let streams = localStreams ?? globalStreams
            if streams.gpu.device == device {
                return streams.gpu.current
            } else {
                return Stream(device)
            }
        }
    }

    static public func defaultStream(_ deviceType: DeviceType) -> Stream {
        switch deviceType {
        case .cpu: .cpu
        case .gpu: .gpu
        }
    }

    public static func == (lhs: Stream, rhs: Stream) -> Bool {
        mlx_stream_equal(lhs.ctx, rhs.ctx)
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        defer { mlx_string_free(s) }
        _ = withEvalLock {
            mlx_stream_tostring(&s, ctx)
        }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
