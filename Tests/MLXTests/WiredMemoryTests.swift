// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import Testing

/// Serialized: the GPU tests each apply (and restore) the process-wide wired
/// limit from their own isolated manager, which knows nothing of the others, so
/// running them in parallel would have them overwrite each other's limits.
@Suite(.serialized)
struct WiredMemoryTests {
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
    @Test(.wiredMemoryControl) func wiredMemoryPolicyStackingAndCustomPolicies() async throws {
        let maxBytes = GPU.deviceInfo().maxRecommendedWorkingSetSize
        guard maxBytes > 0 else {
            try Test.cancel("No recommended working set size available.")
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
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
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

                #expect(event.appliedLimit == expected)
            default:
                break
            }
        }

        #expect(sawCustomDominant, "Expected custom policy to influence the applied limit.")
        #expect(sawTwoPoliciesActive, "Expected tickets from two policies to overlap.")
    }

    /// Reservation-only tickets should not keep the wired limit elevated.
    @Test(.wiredMemoryControl) func reservationOnlyKeepsLimitAtBaseline() async throws {
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
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        // Reservation tickets can trigger limit application, but it should match baseline.
        for event in events where event.kind == .limitApplied {
            #expect(event.appliedLimit == baseline)
        }
    }

    /// Reservation + active tickets should raise the limit, then restore baseline when active ends.
    @Test(.wiredMemoryControl) func reservationPlusActiveRaisesAndRestoresBaseline() async throws {
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
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
        }

        guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline else {
            throw TestError.missingBaseline
        }

        let expectedRaised = baseline + (128 * mib) + (64 * mib)
        let sawRaised = events.contains {
            $0.kind == .limitApplied && $0.appliedLimit == expectedRaised
        }
        #expect(sawRaised, "Expected a limit increase while active work runs.")

        // After active work ends, the manager should restore baseline even if reservations remain.
        if let restore = events.last(where: { $0.kind == .baselineRestored }) {
            #expect(restore.appliedLimit == baseline)
        } else {
            Issue.record("Expected baseline restore event after active work ended.")
        }
    }

    /// Hysteresis should suppress small decreases while active work is running.
    @Test(.wiredMemoryControl) func hysteresisPreventsSmallShrinkWhileActive() async throws {
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
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
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
        #expect(!shrinkApplied, "Expected hysteresis to suppress the small shrink.")

        // Ensure we actually reached the higher limit while both tickets were active.
        let sawHigh = events.contains { $0.kind == .limitApplied && $0.appliedLimit == highLimit }
        #expect(
            sawHigh, "Expected the higher limit to be applied while both tickets were active.")
    }

    /// Cooldown should delay shrink while active work is running, then allow a later shrink.
    @Test(.wiredMemoryControl) func cooldownPreventsRapidOscillationWhileActive() async throws {
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
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
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
            #expect(!immediateShrink, "Expected cooldown to suppress immediate shrink.")
        }

        if let cStartIndex, let cEndIndex {
            let delayedShrink = events[cStartIndex ..< cEndIndex].contains {
                $0.kind == .limitApplied && $0.appliedLimit == midLimit
            }
            #expect(delayedShrink, "Expected a later shrink once cooldown elapsed.")
        }

        let sawHigh = events.contains { $0.kind == .limitApplied && $0.appliedLimit == highLimit }
        #expect(
            sawHigh, "Expected the higher limit to be applied while both tickets were active.")
    }

    /// Admission should account for reservation sizes so that long-lived weights
    /// participate in capacity checks.
    @Test(.wiredMemoryControl) func admissionUsesReservationSizesAndActiveSizes() async throws {
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
            Issue.record("Expected admission wait for the blocked ticket.")
            return
        }

        _ = await activeA.end()
        _ = await startBlocked.value
        _ = await activeB.end()
        _ = await reservation.end()

        let events = try await collectedEvents
        guard !events.isEmpty else {
            try Test.cancel("Wired memory events not available in this build.")
        }

        if events.contains(where: { $0.kind == .limitApplyFailed }) {
            try Test.cancel("Wired limit updates failed on this device.")
        }
    }

    /// Policy-only mode should still enforce admission when wired limits are unsupported.
    @Test func policyOnlyModeEnforcesAdmissionOnCPU() async throws {
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
                Issue.record("Expected admission wait for the blocked ticket.")
                return
            }

            _ = await activeA.end()
            _ = await startBlocked.value
            _ = await activeB.end()
            _ = await reservation.end()

            let events = try await collectedEvents
            guard !events.isEmpty else {
                try Test.cancel("Wired memory events not available in this build.")
            }

            if events.contains(where: { $0.kind == .limitApplyFailed }) {
                try Test.cancel("Wired limit updates failed on this device.")
            }

            guard let baseline = events.first(where: { $0.kind == .baselineCaptured })?.baseline
            else {
                throw TestError.missingBaseline
            }
            #expect(baseline == 1024)
        }
    }

    /// Cancelling a task while `withWiredLimit` is running must not double-end the ticket.
    @Test func withWiredLimitCancellationDoesNotDoubleEnd() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let manager = WiredMemoryManager.makeForTesting(
                configuration: .init(
                    policyOnlyWhenUnsupported: true,
                    baselineOverride: 1024,
                    useRecommendedWorkingSetWhenUnsupported: false
                )
            )
            let policy = SumPolicy()
            let ticket = WiredMemoryTicket(
                size: 64 * mib,
                policy: policy,
                manager: manager,
                kind: .active
            )

            let stream = await manager.events()

            // Signal so the body can indicate it has started.
            let bodyStarted = AsyncStream<Void>.makeStream()

            let task = Task {
                try await WiredMemoryTicket.withWiredLimit(ticket) {
                    bodyStarted.continuation.yield()
                    // Keep the body alive long enough for cancellation to race.
                    try await Task.sleep(nanoseconds: 5_000_000_000)
                }
            }

            // Wait for the body to begin, then cancel.
            for await _ in bodyStarted.stream { break }
            task.cancel()

            // Let the cancellation and body settle.
            _ = try? await task.value

            // Collect events — look for at most one ticketEnded and no assertionFailure (crash).
            let events = try await Self.collectEvents(stream: stream) { event in
                event.kind == .ticketEnded || event.kind == .ticketEndIgnored
            }

            let endedCount = events.filter { $0.kind == .ticketEnded }.count
            let ignoredCount = events.filter { $0.kind == .ticketEndIgnored }.count

            // Exactly one end should have occurred; no double-end.
            #expect(endedCount == 1, "Expected exactly one ticketEnded event.")
            #expect(ignoredCount == 0, "Expected no ticketEndIgnored (double-end) events.")

            // Verify the ticket is no longer active by starting a fresh ticket on the same manager.
            let ticket2 = WiredMemoryTicket(
                size: 32 * mib,
                policy: policy,
                manager: manager,
                kind: .active
            )
            _ = await ticket2.start()
            _ = await ticket2.end()
        }
    }

    /// Collects events from a stream until a predicate matches or a timeout fires.
    private static func collectEvents(
        stream: AsyncStream<WiredMemoryEvent>,
        until predicate: @escaping @Sendable (WiredMemoryEvent) -> Bool,
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

extension Trait where Self == ConditionTrait {
    /// Wired memory control is only available on GPU devices.
    fileprivate static var wiredMemoryControl: Self {
        .enabled(
            if: Device.defaultDevice().deviceType == .gpu,
            "Wired memory control only available on GPU devices.")
    }
}
