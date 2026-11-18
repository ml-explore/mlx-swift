// Copyright © 2024 Apple Inc.

import Foundation
import Testing

@testable import MLX

extension EvalLockSuites {

    /// `.serialized` because the stream pool is process-global: the pool tests below
    /// (e.g. ``streamPoolRecyclesStreams()`` and ``scopeExitRecyclesTheStream()``)
    /// check exactly which streams come back out of it, and the other tests in this
    /// suite take and return streams concurrently.
    ///
    /// This is part of EvalLockSuites because some of those tests also use
    /// streams.
    @Suite(.serialized) struct StreamTests {}
}

extension EvalLockSuites.StreamTests {

    // MARK: - Device: Equatable / Hashable

    @Test func equatableDevice() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu
        let s4 = Device(.gpu)

        // equality does not ignore the index -- this changed: it used to
        // compare the device type only
        #expect(s1 != s2)

        #expect(s1 == s4)

        #expect(s1 != s3)
        #expect(s2 != s3)
    }

    @Test func hashableDevice() {
        #expect(Device.gpu.hashValue == Device(.gpu).hashValue)
        #expect(Device.cpu.hashValue == Device(.cpu).hashValue)

        // usable as a dictionary key
        var counts: [Device: Int] = [:]
        counts[Device.gpu, default: 0] += 1
        counts[Device(.gpu), default: 0] += 1
        counts[Device(.gpu, index: 3), default: 0] += 1
        counts[Device.cpu, default: 0] += 1

        #expect(counts.count == 3)
        #expect(counts[Device.gpu] == 2)
        #expect(counts[Device(.gpu, index: 3)] == 1)
        #expect(counts[Device.cpu] == 1)
    }

    @available(*, deprecated)
    @Test func deprecatedDeviceInitCopiesDefault() {
        // Device() produces the same value as defaultDevice
        #expect(Device() == Device.defaultDevice())

        Device.withDefaultDevice(.cpu) {
            #expect(Device() == Device.cpu)
        }

        let gpu = Device(.gpu, index: 7)
        Device.withDefaultDevice(gpu) {
            #expect(Device() == gpu)
        }
    }

    @Test func deviceType() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        #expect(s1.deviceType == .gpu)
        #expect(s2.deviceType == .gpu)
        #expect(s3.deviceType == .cpu)
    }

    // MARK: - withDefaultDevice

    @Test func usingDevice() {
        let defaultDevice = Device.defaultDevice()

        Device.withDefaultDevice(.cpu) {
            // the default device and the default stream must agree
            #expect(Device.defaultDevice() == Device.cpu)
            #expect(StreamOrDevice.default.stream == MLX.Stream.cpu)
        }
        #expect(defaultDevice == Device.defaultDevice())

        Device.withDefaultDevice(.gpu) {
            #expect(Device.defaultDevice() == Device.gpu)
            #expect(StreamOrDevice.default.stream == MLX.Stream.gpu)
        }

        let gpu = Device(.gpu, index: 7)
        Device.withDefaultDevice(gpu) {
            #expect(Device.defaultDevice() == gpu)
            #expect(StreamOrDevice.default.stream == MLX.Stream.gpu)
        }

        // restored on scope exit
        #expect(defaultDevice == Device.defaultDevice())
    }

    @Test func withDefaultDeviceNests() {
        Device.withDefaultDevice(.cpu) {
            #expect(Device.defaultDevice() == Device.cpu)

            Device.withDefaultDevice(.gpu) {
                #expect(Device.defaultDevice() == Device.gpu)

                Device.withDefaultDevice(.cpu) {
                    #expect(Device.defaultDevice() == Device.cpu)
                }

                #expect(Device.defaultDevice() == Device.gpu)
            }

            #expect(Device.defaultDevice() == Device.cpu)
        }
    }

    @Test func nestedStreamsAndDevices() {
        let outerCpu = MLX.Stream.cpu
        let outerGpu = MLX.Stream.gpu

        Device.withDefaultDevice(.cpu) {
            #expect(MLX.Stream.defaultStream == outerCpu)
            #expect(MLX.Stream.cpu == outerCpu)
            #expect(MLX.Stream.gpu == outerGpu)
        }

        let gpu7 = Device(.gpu, index: 7)
        Stream.withNewDefaultStream(device: gpu7) {
            // default is now gpu/7
            #expect(MLX.Stream.defaultStream != outerGpu)
            #expect(MLX.Stream.defaultStream == MLX.Stream.gpu)
            #expect(MLX.Stream.defaultStream.device == gpu7)

            // not the same stream
            #expect(MLX.Stream.cpu != outerCpu)

            let gpu7Stream = MLX.Stream.defaultStream

            // this will override the cpu stream but not the gpu stream
            Device.withDefaultDevice(.cpu) {
                #expect(MLX.Stream.defaultStream.device == outerCpu.device)

                // this is the same stream, we are just changing the default device
                #expect(MLX.Stream.gpu == gpu7Stream)
            }

            // new scoped stream on GPU index 0
            Stream.withNewDefaultStream(device: .init(.gpu, index: 0)) {
                #expect(MLX.Stream.defaultStream.device == outerGpu.device)
                #expect(MLX.Stream.gpu != gpu7Stream)
                #expect(MLX.Stream.gpu != outerGpu)
                #expect(MLX.Stream.gpu.device == outerGpu.device)
                #expect(MLX.Stream.gpu.device != gpu7)
            }

            // new scoped stream with default cpu but gpu inheriting from parent (gpu7)
            Stream.withNewDefaultStream(device: .cpu) {
                #expect(MLX.Stream.defaultStream.device == outerCpu.device)

                // not the same stream
                #expect(MLX.Stream.gpu != gpu7Stream)

                // but the same device
                #expect(MLX.Stream.gpu.device == gpu7)
            }

        }
    }

    @Test func withNewDefaultStreamDefaultsToCurrentDevice() {
        // `device:` omitted means "a new stream on the current default device".
        Device.withDefaultDevice(.cpu) {
            MLX.Stream.withNewDefaultStream {
                #expect(Device.defaultDevice() == Device.cpu)
                #expect(MLX.Stream.defaultStream == MLX.Stream.cpu)
            }
        }
    }

    @Test func withDefaultDevice() {
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

    @Test func scopeIsInheritedByChildTasks() async {
        // The scope is task-local, so structured child tasks inherit it.
        await Device.withDefaultDevice(.cpu) {
            async let child = Device.defaultDevice()
            let value = await child
            #expect(value == Device.cpu)

            await withTaskGroup(of: Device.self) { group in
                for _ in 0 ..< 4 {
                    group.addTask { Device.defaultDevice() }
                }
                for await device in group {
                    #expect(device == Device.cpu)
                }
            }
        }
    }

    @Test func scopeIsNotInheritedByDetachedWork() async {
        // detached Tasks and Threads do not inherit
        let global = Device.defaultDevice()

        // pick a scope device that differs from the process default so the
        // assertion below can actually distinguish them
        let scoped = Device(.gpu, index: 3)
        #expect(scoped != global)

        await Device.withDefaultDevice(scoped) {
            #expect(Device.defaultDevice() == scoped)

            let detached = await Task.detached { Device.defaultDevice() }.value
            #expect(detached == global, "a detached task must see the global default")

            let fromThread = await withCheckedContinuation { continuation in
                let thread = Thread { continuation.resume(returning: Device.defaultDevice()) }
                thread.start()
            }
            #expect(fromThread == global, "a raw Thread must see the global default")
        }
    }

    @Test func concurrentFirstTouchYieldsOneStream() async {
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

            #expect(
                streams.count == 1,
                "a cold promise handed out \(streams.count) distinct streams under contention")
        }
    }

    // MARK: - Stream / StreamOrDevice resolution

    @Test func deviceTypeOverloads() {
        #expect(MLX.Stream.defaultStream(.cpu) == MLX.Stream.cpu)
        #expect(MLX.Stream.defaultStream(.gpu) == MLX.Stream.gpu)

        #expect(StreamOrDevice.device(.cpu).stream == MLX.Stream.cpu)
        #expect(StreamOrDevice.device(.gpu).stream == MLX.Stream.gpu)

        #expect(StreamOrDevice.cpu.stream == MLX.Stream.cpu)
        #expect(StreamOrDevice.gpu.stream == MLX.Stream.gpu)
    }

    @Test func streamOrDeviceStaticsTrackTheCurrentScope() {
        // `StreamOrDevice.cpu`/`.gpu` became computed, so they must track the
        // current scope rather than caching whatever was current at first access.
        MLX.Stream.withNewDefaultStream(device: .cpu) {
            #expect(StreamOrDevice.cpu.stream == MLX.Stream.cpu)
            #expect(StreamOrDevice.default.stream == MLX.Stream.defaultStream)
        }
    }

    @available(*, deprecated)
    @Test func deprecatedDeviceOverloadHonorsTheDevice() {
        #expect(MLX.Stream.defaultStream(Device.cpu) == MLX.Stream.cpu)
        #expect(MLX.Stream.defaultStream(Device.gpu) == MLX.Stream.gpu)
        #expect(StreamOrDevice.device(Device.cpu).stream == MLX.Stream.cpu)
    }

    // MARK: - Stream pool

    @Test func concurrentStreamsAreDistinct() {
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
        #expect(live.count == n)
        for i in 0 ..< live.count {
            for j in (i + 1) ..< live.count {
                #expect(live[i] != live[j], "the pool handed the same stream to two owners")
            }
        }
    }

    @Test func poolDoesNotCrossDevices() throws {
        // recycling stream innards should never mix types
        for _ in 0 ..< 100 {
            autoreleasepool {
                let device = Device(.gpu, index: Int32.random(in: 0 ..< 10))
                let stream = MLX.Stream(device)
                #expect(stream.device == device)
            }
        }
    }

    /// The global `Stream.cpu` / `Stream.gpu` are created lazily, from the stream pool,
    /// the first time anything in the process uses them.  If that first use (from a
    /// test in another suite, e.g. `stream: .cpu`) lands while a pool test has streams
    /// back in the pool, it takes one of them and the pool test sees a different
    /// stream.  Create them up front so the pool tests do not depend on test order.
    private func realizeGlobalStreams() {
        _ = MLX.Stream.cpu
        _ = MLX.Stream.gpu
    }

    @Test func scopeExitRecyclesTheStream() {
        // see https://github.com/ml-explore/mlx/issues/2118 -- this used to
        // exhaust metal/OS resources; the pool makes it bounded
        realizeGlobalStreams()

        // stream pooling should reuse the same stream over and over
        var indices = Set<Int>()

        for _ in 0 ..< 100 {
            MLX.Stream.withNewDefaultStream(device: .gpu) {
                indices.insert(MLX.Stream.defaultStream.index)

                let x = MLXArray(1)
                let _ = x * x
            }
        }

        #expect(
            indices.count == 1,
            "100 sequential scopes used \(indices.count) distinct streams, expected 1")
    }

    @Test func poolingConcurrent() {
        // but if we have two live streams they cannot be the same
        let s1 = Stream(.cpu)
        let s2 = Stream(.cpu)

        #expect(s1.index != s2.index)
    }

    // MARK: - Misc

    @Test func streamSynchronize() {
        let stream = MLX.Stream(.cpu)
        stream.synchronize()
        MLX.Stream.defaultStream.synchronize()
    }

    @Test func streamOrDeviceDescriptionMatchesItsStream() {
        let stream = MLX.Stream(.cpu)
        #expect(StreamOrDevice.stream(stream).description == stream.description)
        #expect(StreamOrDevice.default.description == MLX.Stream.defaultStream.description)
    }

    @Test func streamOrDeviceStreamUsesGivenStream() {
        // StreamOrDevice.stream(_:) used to ignore its argument and return
        // the default stream instead
        let stream = Stream(.cpu)
        #expect(StreamOrDevice.stream(stream).stream == stream)

        #expect(StreamOrDevice.stream(.cpu).stream == Stream.cpu)
        #expect(StreamOrDevice.stream(.gpu).stream == Stream.gpu)
    }

    @Test func streamPoolRecyclesStreams() {
        // Streams must be recycled through the pool rather than allocated without
        // bound -- see https://github.com/ml-explore/mlx/issues/2118 and #237.
        realizeGlobalStreams()

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
        #expect(first.count == n, "streams held at the same time must be distinct")
        live.removeAll()

        var second = Set<Int>()
        for _ in 0 ..< n {
            let stream = Stream(.cpu)
            second.insert(stream.index)
            live.append(stream)
        }
        #expect(second == first, "the pool did not recycle the released streams")
        live.removeAll()

        // and the set of distinct streams stays bounded over many rounds
        var observed = first
        for _ in 0 ..< 10000 {
            let stream = Stream(.cpu)
            observed.insert(stream.index)
        }
        #expect(
            observed == first,
            "10k sequential streams allocated \(observed.count) distinct streams, expected \(n)")
    }

    @Test func createStream() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            let _ = Stream(.cpu)
        }
    }

}
