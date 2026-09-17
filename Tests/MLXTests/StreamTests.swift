// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class StreamTests: XCTestCase {

    // MARK: - Device: Equatable / Hashable

    func testEquatableDevice() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu
        let s4 = Device(.gpu)

        // equality does not ignore the index -- this changed: it used to
        // compare the device type only
        XCTAssertNotEqual(s1, s2)

        XCTAssertEqual(s1, s4)

        XCTAssertNotEqual(s1, s3)
        XCTAssertNotEqual(s2, s3)
    }

    func testHashableDevice() {
        XCTAssertEqual(Device.gpu.hashValue, Device(.gpu).hashValue)
        XCTAssertEqual(Device.cpu.hashValue, Device(.cpu).hashValue)

        // usable as a dictionary key
        var counts: [Device: Int] = [:]
        counts[Device.gpu, default: 0] += 1
        counts[Device(.gpu), default: 0] += 1
        counts[Device(.gpu, index: 3), default: 0] += 1
        counts[Device.cpu, default: 0] += 1

        XCTAssertEqual(counts.count, 3)
        XCTAssertEqual(counts[Device.gpu], 2)
        XCTAssertEqual(counts[Device(.gpu, index: 3)], 1)
        XCTAssertEqual(counts[Device.cpu], 1)
    }

    @available(*, deprecated)
    func testDeprecatedDeviceInitCopiesDefault() {
        // Device() produces the same value as defaultDevice
        XCTAssertEqual(Device(), Device.defaultDevice())

        Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(Device(), Device.cpu)
        }

        let gpu = Device(.gpu, index: 7)
        Device.withDefaultDevice(gpu) {
            XCTAssertEqual(Device(), gpu)
        }
    }

    func testDeviceType() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        XCTAssertEqual(s1.deviceType, .gpu)
        XCTAssertEqual(s2.deviceType, .gpu)
        XCTAssertEqual(s3.deviceType, .cpu)
    }

    // MARK: - withDefaultDevice

    func testUsingDevice() {
        let defaultDevice = Device.defaultDevice()

        Device.withDefaultDevice(.cpu) {
            // the default device and the default stream must agree
            XCTAssertEqual(Device.defaultDevice(), Device.cpu)
            XCTAssertEqual(StreamOrDevice.default.stream, MLX.Stream.cpu)
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())

        Device.withDefaultDevice(.gpu) {
            XCTAssertEqual(Device.defaultDevice(), Device.gpu)
            XCTAssertEqual(StreamOrDevice.default.stream, MLX.Stream.gpu)
        }

        let gpu = Device(.gpu, index: 7)
        Device.withDefaultDevice(gpu) {
            XCTAssertEqual(Device.defaultDevice(), gpu)
            XCTAssertEqual(StreamOrDevice.default.stream, MLX.Stream.gpu)
        }

        // restored on scope exit
        XCTAssertEqual(defaultDevice, Device.defaultDevice())
    }

    func testWithDefaultDeviceNests() {
        Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(Device.defaultDevice(), Device.cpu)

            Device.withDefaultDevice(.gpu) {
                XCTAssertEqual(Device.defaultDevice(), Device.gpu)

                Device.withDefaultDevice(.cpu) {
                    XCTAssertEqual(Device.defaultDevice(), Device.cpu)
                }

                XCTAssertEqual(Device.defaultDevice(), Device.gpu)
            }

            XCTAssertEqual(Device.defaultDevice(), Device.cpu)
        }
    }

    func testNestedStreamsAndDevices() {
        let outerCpu = MLX.Stream.cpu
        let outerGpu = MLX.Stream.gpu

        Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(MLX.Stream.defaultStream, outerCpu)
            XCTAssertEqual(MLX.Stream.cpu, outerCpu)
            XCTAssertEqual(MLX.Stream.gpu, outerGpu)
        }

        let gpu7 = Device(.gpu, index: 7)
        Stream.withNewDefaultStream(device: gpu7) {
            // default is now gpu/7
            XCTAssertNotEqual(MLX.Stream.defaultStream, outerGpu)
            XCTAssertEqual(MLX.Stream.defaultStream, MLX.Stream.gpu)
            XCTAssertEqual(MLX.Stream.defaultStream.device, gpu7)

            // not the same stream
            XCTAssertNotEqual(MLX.Stream.cpu, outerCpu)

            let gpu7Stream = MLX.Stream.defaultStream

            // this will override the cpu stream but not the gpu stream
            Device.withDefaultDevice(.cpu) {
                XCTAssertEqual(MLX.Stream.defaultStream.device, outerCpu.device)

                // this is the same stream, we are just changing the default device
                XCTAssertEqual(MLX.Stream.gpu, gpu7Stream)
            }

            // new scoped stream on GPU index 0
            Stream.withNewDefaultStream(device: .init(.gpu, index: 0)) {
                XCTAssertEqual(MLX.Stream.defaultStream.device, outerGpu.device)
                XCTAssertNotEqual(MLX.Stream.gpu, gpu7Stream)
                XCTAssertNotEqual(MLX.Stream.gpu, outerGpu)
                XCTAssertEqual(MLX.Stream.gpu.device, outerGpu.device)
                XCTAssertNotEqual(MLX.Stream.gpu.device, gpu7)
            }

            // new scoped stream with default cpu but gpu inheriting from parent (gpu7)
            Stream.withNewDefaultStream(device: .cpu) {
                XCTAssertEqual(MLX.Stream.defaultStream.device, outerCpu.device)

                // not the same stream
                XCTAssertNotEqual(MLX.Stream.gpu, gpu7Stream)

                // but the same device
                XCTAssertEqual(MLX.Stream.gpu.device, gpu7)
            }

        }
    }

    func testWithNewDefaultStreamDefaultsToCurrentDevice() {
        // `device:` omitted means "a new stream on the current default device".
        Device.withDefaultDevice(.cpu) {
            MLX.Stream.withNewDefaultStream {
                XCTAssertEqual(Device.defaultDevice(), Device.cpu)
                XCTAssertEqual(MLX.Stream.defaultStream, MLX.Stream.cpu)
            }
        }
    }

    func testWithDefaultDevice() {
        // Issue #237 -- scoped variant
        for _ in 1 ..< 10000 {
            Device.withDefaultDevice(.cpu) {
                Device.withDefaultDevice(.gpu) {
                    let x = MLXArray(1)
                    let _ = x * x
                }
            }
        }
    }

    // MARK: - Task-local scoping (async + propagation boundaries)

    func testScopeIsInheritedByChildTasks() async {
        // The scope is task-local, so structured child tasks inherit it.
        await Device.withDefaultDevice(.cpu) {
            async let child = Device.defaultDevice()
            let value = await child
            XCTAssertEqual(value, Device.cpu)

            await withTaskGroup(of: Device.self) { group in
                for _ in 0 ..< 4 {
                    group.addTask { Device.defaultDevice() }
                }
                for await device in group {
                    XCTAssertEqual(device, Device.cpu)
                }
            }
        }
    }

    func testScopeIsNotInheritedByDetachedWork() async {
        // detached Tasks and Threads do not inherit
        let global = Device.defaultDevice()

        // pick a scope device that differs from the process default so the
        // assertion below can actually distinguish them
        let scoped = Device(.gpu, index: 3)
        XCTAssertNotEqual(scoped, global)

        await Device.withDefaultDevice(scoped) {
            XCTAssertEqual(Device.defaultDevice(), scoped)

            let detached = await Task.detached { Device.defaultDevice() }.value
            XCTAssertEqual(detached, global, "a detached task must see the global default")

            let fromThread = await withCheckedContinuation { continuation in
                let thread = Thread { continuation.resume(returning: Device.defaultDevice()) }
                thread.start()
            }
            XCTAssertEqual(fromThread, global, "a raw Thread must see the global default")
        }
    }

    func testConcurrentFirstTouchYieldsOneStream() async {
        // test of the double checked lock in StreamPromise
        await MLX.Stream.withNewDefaultStream(device: .cpu) {
            let streams = await withTaskGroup(of: ObjectIdentifier.self) { group in
                for _ in 0 ..< 16 {
                    group.addTask { ObjectIdentifier(MLX.Stream.defaultStream) }
                }
                var seen = Set<ObjectIdentifier>()
                for await id in group { seen.insert(id) }
                return seen
            }

            XCTAssertEqual(
                streams.count, 1,
                "a cold promise handed out \(streams.count) distinct streams under contention")
        }
    }

    // MARK: - Stream / StreamOrDevice resolution

    func testDeviceTypeOverloads() {
        XCTAssertEqual(MLX.Stream.defaultStream(.cpu), MLX.Stream.cpu)
        XCTAssertEqual(MLX.Stream.defaultStream(.gpu), MLX.Stream.gpu)

        XCTAssertEqual(StreamOrDevice.device(.cpu).stream, MLX.Stream.cpu)
        XCTAssertEqual(StreamOrDevice.device(.gpu).stream, MLX.Stream.gpu)

        XCTAssertEqual(StreamOrDevice.cpu.stream, MLX.Stream.cpu)
        XCTAssertEqual(StreamOrDevice.gpu.stream, MLX.Stream.gpu)
    }

    func testStreamOrDeviceStaticsTrackTheCurrentScope() {
        // `StreamOrDevice.cpu`/`.gpu` became computed, so they must track the
        // current scope rather than caching whatever was current at first access.
        MLX.Stream.withNewDefaultStream(device: .cpu) {
            XCTAssertEqual(StreamOrDevice.cpu.stream, MLX.Stream.cpu)
            XCTAssertEqual(StreamOrDevice.default.stream, MLX.Stream.defaultStream)
        }
    }

    @available(*, deprecated)
    func testDeprecatedDeviceOverloadHonorsTheDevice() {
        XCTAssertEqual(MLX.Stream.defaultStream(Device.cpu), MLX.Stream.cpu)
        XCTAssertEqual(MLX.Stream.defaultStream(Device.gpu), MLX.Stream.gpu)
        XCTAssertEqual(StreamOrDevice.device(Device.cpu).stream, MLX.Stream.cpu)
    }

    // MARK: - Stream pool

    func testConcurrentStreamsAreDistinct() {
        // Streams handed out at the same time must never alias -- a double-release
        // into the pool would let two owners share one `mlx_stream`.
        final class Box: @unchecked Sendable {
            private let lock = NSLock()
            private var streams: [MLX.Stream] = []
            func append(_ stream: MLX.Stream) { lock.withLock { streams.append(stream) } }
            var all: [MLX.Stream] { lock.withLock { streams } }
        }

        let n = 32
        let box = Box()

        DispatchQueue.concurrentPerform(iterations: n) { _ in
            box.append(MLX.Stream(.cpu))
        }

        let live = box.all
        XCTAssertEqual(live.count, n)
        for i in 0 ..< live.count {
            for j in (i + 1) ..< live.count {
                XCTAssertNotEqual(
                    live[i], live[j], "the pool handed the same stream to two owners")
            }
        }
    }

    func testPoolDoesNotCrossDevices() throws {
        // recycling stream innards should never mix types
        for _ in 0 ..< 100 {
            autoreleasepool {
                let device = Device(.gpu, index: Int32.random(in: 0 ..< 10))
                let stream = MLX.Stream(device)
                XCTAssertEqual(stream.device, device)
            }
        }
    }

    func testScopeExitRecyclesTheStream() {
        // see https://github.com/ml-explore/mlx/issues/2118 -- this used to
        // exhaust metal/OS resources; the pool makes it bounded

        // stream pooling should reuse the same stream over and over
        var indices = Set<Int>()

        for _ in 0 ..< 100 {
            MLX.Stream.withNewDefaultStream(device: .gpu) {
                indices.insert(MLX.Stream.defaultStream.index)

                let x = MLXArray(1)
                let _ = x * x
            }
        }

        XCTAssertEqual(
            indices.count, 1,
            "100 sequential scopes used \(indices.count) distinct streams, expected 1")
    }

    func testPoolingConcurrent() {
        // but if we have two live streams they cannot be the same
        let s1 = Stream(.cpu)
        let s2 = Stream(.cpu)

        XCTAssertNotEqual(s1.index, s2.index)
    }

    // MARK: - Misc

    func testStreamSynchronize() {
        let stream = MLX.Stream(.cpu)
        stream.synchronize()
        MLX.Stream.defaultStream.synchronize()
    }

    func testStreamOrDeviceDescriptionMatchesItsStream() {
        let stream = MLX.Stream(.cpu)
        XCTAssertEqual(StreamOrDevice.stream(stream).description, stream.description)
        XCTAssertEqual(StreamOrDevice.default.description, MLX.Stream.defaultStream.description)
    }

    func testStreamOrDeviceStreamUsesGivenStream() {
        // StreamOrDevice.stream(_:) used to ignore its argument and return
        // the default stream instead
        let stream = Stream(.cpu)
        XCTAssertEqual(StreamOrDevice.stream(stream).stream, stream)

        XCTAssertEqual(StreamOrDevice.stream(.cpu).stream, Stream.cpu)
        XCTAssertEqual(StreamOrDevice.stream(.gpu).stream, Stream.gpu)
    }

    func testStreamPoolRecyclesStreams() {
        // Streams must be recycled through the pool rather than allocated without
        // bound -- see https://github.com/ml-explore/mlx/issues/2118 and #237.

        let n = 5

        // hold n at once, drop them, then take n again -- the second batch must
        // be the same underlying streams
        var first = Set<Int>()
        var live: [MLX.Stream] = []
        for _ in 0 ..< n {
            let stream = Stream(.cpu)
            first.insert(stream.index)
            live.append(stream)
        }
        XCTAssertEqual(first.count, n, "streams held at the same time must be distinct")
        live.removeAll()

        var second = Set<Int>()
        for _ in 0 ..< n {
            let stream = Stream(.cpu)
            second.insert(stream.index)
            live.append(stream)
        }
        XCTAssertEqual(second, first, "the pool did not recycle the released streams")
        live.removeAll()

        // and the set of distinct streams stays bounded over many rounds
        var observed = first
        for _ in 0 ..< 10000 {
            let stream = Stream(.cpu)
            observed.insert(stream.index)
        }
        XCTAssertEqual(
            observed, first,
            "10k sequential streams allocated \(observed.count) distinct streams, expected \(n)")
    }

    func testCreateStream() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            let _ = Stream(.cpu)
        }
        print("here")
    }

}
