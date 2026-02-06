// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

final class WiredMemoryTests: XCTestCase {
    enum TestError: Error {
        case missingInfo
        case timeout
        case missingBaseline
    }

    struct DoubleSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +) * 2
        }
    }

    struct SumPolicy: WiredMemoryPolicy, Hashable, Sendable {
        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +)
        }
    }

    struct CappedSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
        let capDelta: Int

        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +)
        }

        func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
            let projected = activeSizes.reduce(0, +) + max(0, newSize)
            return projected <= capDelta
        }
    }

    private let mib = 1024 * 1024

    /// Ensures that the manager can coordinate multiple policy groups and that
    /// the applied limit reflects the max across those groups.
    func testWiredMemoryPolicyStackingAndCustomPolicies() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
        guard maxBytes > 0 else {
            throw XCTSkip("No recommended working set size available.")
        }

        let manager = WiredMemoryManager.makeForTesting()
        let sumPolicy = SumPolicy()
        let customPolicy = DoubleSumPolicy()

        let sumTicket1 = WiredMemoryTicket(size: 64 * mib, policy: sumPolicy, manager: manager)
        let sumTicket2 = WiredMemoryTicket(size: 64 * mib, policy: sumPolicy, manager: manager)
        let customTicket = WiredMemoryTicket(size: 96 * mib, policy: customPolicy, manager: manager)

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await sumTicket1.start()
        _ = await sumTicket2.start()
        _ = await customTicket.start()
        _ = await customTicket.end()
        _ = await sumTicket2.end()
        _ = await sumTicket1.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        enum PolicyKind {
            case sum
            case custom
        }

        let policyByTicket: [UUID: PolicyKind] = [
            sumTicket1.id: .sum,
            sumTicket2.id: .sum,
            customTicket.id: .custom,
        ]

        var active: [UUID: Int] = [:]
        var sawCustomDominant = false
        var sawTwoPoliciesActive = false

        for event in events {
            switch event.kind {
            case .ticketStarted:
                guard let id = event.ticketID, let size = event.size else { continue }
                // Reconstruct active sizes from the event stream to validate policy math.
                active[id] = size
            case .ticketEnded:
                if let id = event.ticketID {
                    active.removeValue(forKey: id)
                }
            case .limitApplied:
                let sumSizes = active.reduce(0) { partial, entry in
                    guard let kind = policyByTicket[entry.key], kind == .sum else { return partial }
                    return partial + entry.value
                }
                let customSizes = active.reduce(0) { partial, entry in
                    guard let kind = policyByTicket[entry.key], kind == .custom else {
                        return partial
                    }
                    return partial + entry.value
                }

                if sumSizes > 0 && customSizes > 0 {
                    sawTwoPoliciesActive = true
                }

                let sumLimit = baseline + sumSizes
                let customLimit = baseline + (customSizes * 2)
                let expected = max(sumLimit, customLimit)

                if customSizes > 0 && expected == customLimit && customLimit > sumLimit {
                    sawCustomDominant = true
                }

                XCTAssertEqual(event.appliedLimit, expected)
            default:
                break
            }
        }

        XCTAssertTrue(sawCustomDominant, "Expected custom policy to influence the applied limit.")
        XCTAssertTrue(sawTwoPoliciesActive, "Expected tickets from two policies to overlap.")
    }

    /// Reservation-only tickets should not keep the wired limit elevated.
    func testReservationOnlyKeepsLimitAtBaseline() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let manager = WiredMemoryManager.makeForTesting()
        let policy = SumPolicy()

        let reservation = WiredMemoryTicket(
            size: 128 * mib,
            policy: policy,
            manager: manager,
            kind: .reservation
        )

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await reservation.start()
        _ = await reservation.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        // Reservation tickets can trigger limit application, but it should match baseline.
        for event in events where event.kind == .limitApplied {
            XCTAssertEqual(event.appliedLimit, baseline)
        }
    }

    /// Reservation + active tickets should raise the limit, then restore baseline when active ends.
    func testReservationPlusActiveRaisesAndRestoresBaseline() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let manager = WiredMemoryManager.makeForTesting()
        let policy = SumPolicy()

        let reservation = WiredMemoryTicket(
            size: 128 * mib,
            policy: policy,
            manager: manager,
            kind: .reservation
        )
        let active = WiredMemoryTicket(
            size: 64 * mib,
            policy: policy,
            manager: manager,
            kind: .active
        )

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await reservation.start()
        _ = await active.start()
        _ = await active.end()
        _ = await reservation.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        let expectedRaised = baseline + (128 * mib) + (64 * mib)
        let sawRaised = events.contains {
            $0.kind == .limitApplied && $0.appliedLimit == expectedRaised
        }
        XCTAssertTrue(sawRaised, "Expected a limit increase while active work runs.")

        // After active work ends, the manager should restore baseline even if reservations remain.
        if let restore = events.last(where: { $0.kind == .baselineRestored }) {
            XCTAssertEqual(restore.appliedLimit, baseline)
        } else {
            XCTFail("Expected baseline restore event after active work ended.")
        }
    }

    /// Hysteresis should suppress small decreases while active work is running.
    func testHysteresisPreventsSmallShrinkWhileActive() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let manager = WiredMemoryManager.makeForTesting(
            configuration: .init(shrinkThresholdRatio: 0.5, shrinkCooldown: 0)
        )
        let policy = SumPolicy()

        let large = WiredMemoryTicket(size: 300 * mib, policy: policy, manager: manager)
        let small = WiredMemoryTicket(size: 20 * mib, policy: policy, manager: manager)

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await large.start()
        _ = await small.start()
        _ = await small.end()
        _ = await large.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        let highLimit = baseline + (300 * mib) + (20 * mib)
        let lowLimit = baseline + (300 * mib)

        // Find the point where the small ticket ended; any shrink to lowLimit before
        // the large ticket ends would indicate hysteresis failed.
        var smallEndedIndex: Int?
        var largeEndedIndex: Int?
        for (index, event) in events.enumerated() {
            if event.kind == .ticketEnded && event.ticketID == small.id {
                smallEndedIndex = index
            }
            if event.kind == .ticketEnded && event.ticketID == large.id {
                largeEndedIndex = index
            }
        }

        guard let smallEnd = smallEndedIndex, let largeEnd = largeEndedIndex else {
            throw TestError.missingInfo
        }

        let shrinkApplied = events[smallEnd ..< largeEnd].contains {
            $0.kind == .limitApplied && $0.appliedLimit == lowLimit
        }
        XCTAssertFalse(shrinkApplied, "Expected hysteresis to suppress the small shrink.")

        // Ensure we actually reached the higher limit while both tickets were active.
        let sawHigh = events.contains { $0.kind == .limitApplied && $0.appliedLimit == highLimit }
        XCTAssertTrue(
            sawHigh, "Expected the higher limit to be applied while both tickets were active.")
    }

    /// Cooldown should delay shrink while active work is running, then allow a later shrink.
    func testCooldownPreventsRapidOscillationWhileActive() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let manager = WiredMemoryManager.makeForTesting(
            configuration: .init(shrinkThresholdRatio: 0, shrinkCooldown: 0.5)
        )
        let policy = SumPolicy()

        let a = WiredMemoryTicket(size: 100 * mib, policy: policy, manager: manager)
        let b = WiredMemoryTicket(size: 300 * mib, policy: policy, manager: manager)
        let c = WiredMemoryTicket(size: 50 * mib, policy: policy, manager: manager)

        let stream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored && event.activeCount == 0
        }

        _ = await a.start()
        _ = await b.start()
        _ = await b.end()

        // Allow time for any immediate shrink attempts to appear in the event stream.
        try await Task.sleep(nanoseconds: 50_000_000)

        // Wait out the cooldown before triggering another recompute.
        try await Task.sleep(nanoseconds: 600_000_000)

        _ = await c.start()
        _ = await c.end()
        _ = await a.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        let highLimit = baseline + (100 * mib) + (300 * mib)
        let midLimit = baseline + (100 * mib) + (50 * mib)
        let lowLimit = baseline + (100 * mib)

        let bEndIndex = events.firstIndex { $0.kind == .ticketEnded && $0.ticketID == b.id }
        let cStartIndex = events.firstIndex { $0.kind == .ticketStarted && $0.ticketID == c.id }
        let cEndIndex = events.firstIndex { $0.kind == .ticketEnded && $0.ticketID == c.id }

        if let bEndIndex, let cStartIndex {
            let immediateShrink = events[bEndIndex ..< cStartIndex].contains {
                $0.kind == .limitApplied && $0.appliedLimit == lowLimit
            }
            XCTAssertFalse(immediateShrink, "Expected cooldown to suppress immediate shrink.")
        }

        if let cStartIndex, let cEndIndex {
            let delayedShrink = events[cStartIndex ..< cEndIndex].contains {
                $0.kind == .limitApplied && $0.appliedLimit == midLimit
            }
            XCTAssertTrue(delayedShrink, "Expected a later shrink once cooldown elapsed.")
        }

        let sawHigh = events.contains { $0.kind == .limitApplied && $0.appliedLimit == highLimit }
        XCTAssertTrue(
            sawHigh, "Expected the higher limit to be applied while both tickets were active.")
    }

    /// Admission should account for reservation sizes so that long-lived weights
    /// participate in capacity checks.
    func testAdmissionUsesReservationSizesAndActiveSizes() async throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Wired memory control only available on GPU devices.")
        }

        let manager = WiredMemoryManager.makeForTesting()
        let policy = CappedSumPolicy(capDelta: 150 * mib)

        let reservation = WiredMemoryTicket(
            size: 100 * mib,
            policy: policy,
            manager: manager,
            kind: .reservation
        )
        let activeA = WiredMemoryTicket(
            size: 40 * mib,
            policy: policy,
            manager: manager,
            kind: .active
        )
        let blockedID = UUID()
        let activeB = WiredMemoryTicket(
            id: blockedID,
            size: 20 * mib,
            policy: policy,
            manager: manager,
            kind: .active
        )

        let stream = await manager.events()
        let admissionStream = await manager.events()
        async let collectedEvents = Self.collectEvents(stream: stream) { event in
            event.kind == .baselineRestored
        }

        _ = await reservation.start()
        _ = await activeA.start()

        let startBlocked = Task { await activeB.start() }
        let admissionEvents = try await Self.collectEvents(stream: admissionStream) { event in
            event.kind == .admissionWait && event.ticketID == blockedID
        }

        guard admissionEvents.contains(where: { $0.kind == .admissionWait }) else {
            XCTFail("Expected admission wait for the blocked ticket.")
            return
        }

        _ = await activeA.end()
        _ = await startBlocked.value
        _ = await activeB.end()
        _ = await reservation.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            throw XCTSkip("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            throw XCTSkip("Wired limit updates failed on this device.")
        }
    }

    /// Policy-only mode should still enforce admission when wired limits are unsupported.
    func testPolicyOnlyModeEnforcesAdmissionOnCPU() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let manager = WiredMemoryManager.makeForTesting(
                configuration: .init(
                    policyOnlyWhenUnsupported: true,
                    baselineOverride: 1024,
                    useRecommendedWorkingSetWhenUnsupported: false
                )
            )
            let policy = CappedSumPolicy(capDelta: 150 * mib)

            let reservation = WiredMemoryTicket(
                size: 100 * mib,
                policy: policy,
                manager: manager,
                kind: .reservation
            )
            let activeA = WiredMemoryTicket(
                size: 40 * mib,
                policy: policy,
                manager: manager,
                kind: .active
            )
            let blockedID = UUID()
            let activeB = WiredMemoryTicket(
                id: blockedID,
                size: 20 * mib,
                policy: policy,
                manager: manager,
                kind: .active
            )

            let stream = await manager.events()
            let admissionStream = await manager.events()
            async let collectedEvents = Self.collectEvents(stream: stream) { event in
                event.kind == .baselineRestored && event.activeCount == 0
            }

            _ = await reservation.start()
            _ = await activeA.start()

            let startBlocked = Task { await activeB.start() }
            let admissionEvents = try await Self.collectEvents(stream: admissionStream) { event in
                event.kind == .admissionWait && event.ticketID == blockedID
            }

            guard admissionEvents.contains(where: { $0.kind == .admissionWait }) else {
                XCTFail("Expected admission wait for the blocked ticket.")
                return
            }

            _ = await activeA.end()
            _ = await startBlocked.value
            _ = await activeB.end()
            _ = await reservation.end()

            let events = try await collectedEvents
            guard !events.isEmpty else {
                throw XCTSkip("Wired memory events not available in this build.")
            }

            if events.contains(where: { $0.kind == .limitApplyFailed }) {
                throw XCTSkip("Wired limit updates failed on this device.")
            }

            guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline
            else {
                throw TestError.missingBaseline
            }
            XCTAssertEqual(baseline, 1024)
        }
    }

    /// Collects events from a stream until a predicate matches or a timeout fires.
    private static func collectEvents(
        stream: AsyncStream<WiredMemoryEvent>,
        until predicate: @Sendable @escaping (WiredMemoryEvent) -> Bool,
        timeout: TimeInterval = 10
    ) async throws -> [WiredMemoryEvent] {
        return try await withThrowingTaskGroup(of: [WiredMemoryEvent].self) { group in
            group.addTask {
                var events: [WiredMemoryEvent] = []
                for await event in stream {
                    events.append(event)
                    if predicate(event) {
                        break
                    }
                }
                return events
            }

            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw TestError.timeout
            }

            let result = try await group.next()
            group.cancelAll()
            return result ?? []
        }
    }
}
