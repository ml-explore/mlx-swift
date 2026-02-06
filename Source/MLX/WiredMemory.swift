// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Low-level access to the process-wide wired memory limit.
///
/// This talks directly to the Cmlx API and is intentionally scoped to this file
/// so that higher-level policy logic remains testable and composable.
private enum WiredMemoryBackend {
    /// Whether this process can adjust wired memory on the current device.
    ///
    /// We only expose wired limit control on GPU devices. Unsupported devices
    /// behave as a no-op and simply keep baseline values.
    static var isSupported: Bool {
        Device.defaultDevice().deviceType == .gpu
    }

    /// Attempts to apply a new wired memory limit.
    ///
    /// Returns `true` on success. This does not enforce any hysteresis or policy
    /// logic; those are handled by `WiredMemoryManager`.
    static func applyLimit(_ limit: Int) -> Bool {
        guard isSupported else { return false }
        guard limit >= 0 else { return false }
        var previous: size_t = 0
        let result = evalLock.withLock {
            mlx_set_wired_limit(&previous, size_t(limit))
        }
        return result == 0
    }
}

/// Debug event emitted by ``WiredMemoryManager`` when coordinating wired memory changes.
///
/// Events are only emitted in DEBUG builds. In release builds the event stream
/// is empty and finishes immediately.
public struct WiredMemoryEvent: Sendable {
    public enum Kind: String, Sendable {
        /// The baseline wired limit was captured from the system.
        case baselineCaptured
        /// A ticket could not be admitted and will wait for capacity.
        case admissionWait
        /// A waiting ticket was admitted.
        case admissionGranted
        /// A waiting ticket was cancelled.
        case admissionCancelled
        /// A ticket became active.
        case ticketStarted
        /// A duplicate start was ignored (ticket already active).
        case ticketStartIgnored
        /// A ticket ended.
        case ticketEnded
        /// A duplicate end was ignored (ticket not active).
        case ticketEndIgnored
        /// The manager computed a desired limit from policy inputs.
        case limitComputed
        /// The manager successfully applied a new limit.
        case limitApplied
        /// The manager attempted to apply a new limit but failed.
        case limitApplyFailed
        /// The manager restored the baseline when work completed.
        case baselineRestored
    }

    /// Monotonic sequence number assigned by the manager.
    public let sequence: UInt64
    /// Timestamp when the event was emitted.
    public let timestamp: Date
    /// Event type.
    public let kind: Kind
    /// Ticket identifier, if the event is associated with a specific ticket.
    public let ticketID: UUID?
    /// Ticket size in bytes, if applicable.
    public let size: Int?
    /// Debug label for the policy group, if applicable.
    public let policy: String?
    /// Baseline wired limit captured before the manager applies any changes.
    public let baseline: Int?
    /// Desired limit computed by policy aggregation.
    public let desiredLimit: Int?
    /// Limit that was actually applied (or attempted).
    public let appliedLimit: Int?
    /// Number of active tickets at time of emission.
    public let activeCount: Int
    /// Number of admission waiters at time of emission.
    public let waiterCount: Int

    public init(
        sequence: UInt64,
        timestamp: Date,
        kind: Kind,
        ticketID: UUID? = nil,
        size: Int? = nil,
        policy: String? = nil,
        baseline: Int? = nil,
        desiredLimit: Int? = nil,
        appliedLimit: Int? = nil,
        activeCount: Int,
        waiterCount: Int
    ) {
        self.sequence = sequence
        self.timestamp = timestamp
        self.kind = kind
        self.ticketID = ticketID
        self.size = size
        self.policy = policy
        self.baseline = baseline
        self.desiredLimit = desiredLimit
        self.appliedLimit = appliedLimit
        self.activeCount = activeCount
        self.waiterCount = waiterCount
    }
}

/// Policy for computing a process-global wired memory limit.
///
/// Policies are grouped by their `id` so that multiple tickets can coordinate
/// through a shared policy instance. The manager computes one limit per policy
/// group, then uses the maximum across groups to avoid double-counting and to
/// allow heterogeneous strategies to coexist. For reference-type policies,
/// provide a stable `id` to define grouping semantics.
public protocol WiredMemoryPolicy: Sendable, Identifiable where ID == AnyHashable {
    /// Compute the desired wired limit in bytes for the current active set.
    ///
    /// This is called after admission succeeds, whenever the manager needs to
    /// update the process-wide wired limit. The manager groups tickets by policy,
    /// calls this method with the sizes for that policy, then takes the maximum
    /// limit across all policies.
    func limit(baseline: Int, activeSizes: [Int]) -> Int

    /// Decide whether a new ticket can be admitted. Defaults to allowing all tickets.
    ///
    /// This is evaluated before a ticket is started. If it returns false, `start()`
    /// will suspend until another ticket ends and capacity becomes available.
    /// Admission checks only consider tickets that share the same policy grouping.
    func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool
}

extension WiredMemoryPolicy {
    public func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        true
    }
}

/// Hashable policies get an `id` for free.
extension WiredMemoryPolicy where Self: Hashable {
    public var id: AnyHashable { AnyHashable(self) }
}

/// Policy that sums active ticket sizes and adds them to the baseline.
public struct WiredSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Stable grouping identifier for this policy instance.
    public let identifier: UUID

    public init(id: UUID = UUID()) {
        self.identifier = id
    }

    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        baseline + activeSizes.reduce(0, +)
    }
}

/// Policy that uses the maximum active ticket size and adds it to the baseline.
public struct WiredMaxPolicy: WiredMemoryPolicy, Hashable, Sendable {
    /// Stable grouping identifier for this policy instance.
    public let identifier: UUID

    public init(id: UUID = UUID()) {
        self.identifier = id
    }

    public func limit(baseline: Int, activeSizes: [Int]) -> Int {
        baseline + (activeSizes.max() ?? 0)
    }
}

/// Configuration knobs for `WiredMemoryManager` behavior.
///
/// These settings implement hysteresis to prevent small or frequent shrinks
/// while active work is running. Growing the limit is always allowed; shrinking
/// is gated by a minimum drop and a minimum time between changes.
public struct WiredMemoryManagerConfiguration: Sendable, Hashable {
    /// Minimum fractional drop required before shrinking while tickets are active.
    /// Example: 0.25 means the desired limit must be at least 25% lower than current.
    public var shrinkThresholdRatio: Double

    /// Minimum time in seconds between shrink attempts while tickets are active.
    public var shrinkCooldown: TimeInterval

    /// If true, policy admission and limit calculations still run even when
    /// wired memory control is unsupported (e.g. CPU-only execution). The
    /// manager will not attempt to change wired memory, but tickets can still
    /// gate admission and emit events. Defaults to `true` to keep admission
    /// behavior consistent across CPU/GPU backends.
    public var policyOnlyWhenUnsupported: Bool

    /// Optional baseline to use instead of the cached limit.
    /// This is useful in policy-only mode to provide a meaningful budget.
    public var baselineOverride: Int?

    /// If true and wired memory is unsupported, attempt to use Metal's
    /// recommended working set size as the baseline when no override is set.
    public var useRecommendedWorkingSetWhenUnsupported: Bool

    /// Creates a new configuration for hysteresis behavior.
    ///
    /// - Parameters:
    ///   - shrinkThresholdRatio: Minimum fractional drop to allow shrinking.
    ///   - shrinkCooldown: Minimum time between shrink attempts while active.
    public init(
        shrinkThresholdRatio: Double = 0.25,
        shrinkCooldown: TimeInterval = 1.0,
        policyOnlyWhenUnsupported: Bool = true,
        baselineOverride: Int? = nil,
        useRecommendedWorkingSetWhenUnsupported: Bool = true
    ) {
        self.shrinkThresholdRatio = max(0, min(1, shrinkThresholdRatio))
        self.shrinkCooldown = max(0, shrinkCooldown)
        self.policyOnlyWhenUnsupported = policyOnlyWhenUnsupported
        self.baselineOverride = baselineOverride
        self.useRecommendedWorkingSetWhenUnsupported = useRecommendedWorkingSetWhenUnsupported
    }
}

/// Describes whether a ticket represents active work or a long-lived reservation.
///
/// Reservation tickets participate in admission and limit calculation, but do not
/// keep the wired limit elevated on their own. This allows modeling long-lived
/// weights without wiring memory while the system is idle.
public enum WiredMemoryTicketKind: Sendable {
    /// Active work that should drive limit updates (e.g. inference).
    case active
    /// Passive reservation that should be considered for admission but should not
    /// keep the wired limit elevated on its own.
    case reservation
}

/// Handle for coordinating wired memory changes across concurrent work.
///
/// A ticket represents a single unit of memory demand. Tickets are started and
/// ended explicitly and are safe to start/end multiple times (extra calls are
/// ignored). Use `withWiredLimit` to ensure cancellation-safe pairing.
public struct WiredMemoryTicket: Sendable, Identifiable {
    /// Unique identifier for this ticket.
    public let id: UUID
    /// Requested size in bytes.
    public let size: Int
    /// Manager responsible for coordinating the ticket.
    public let manager: WiredMemoryManager
    /// Policy that controls admission and limit computation.
    public let policy: any WiredMemoryPolicy
    /// Whether this ticket represents active work or a reservation.
    public let kind: WiredMemoryTicketKind

    /// Creates a ticket bound to a policy and manager.
    public init(
        id: UUID = UUID(),
        size: Int,
        policy: any WiredMemoryPolicy,
        manager: WiredMemoryManager = .shared,
        kind: WiredMemoryTicketKind = .active
    ) {
        self.id = id
        self.size = size
        self.policy = policy
        self.manager = manager
        self.kind = kind
    }

    /// Starts the ticket and returns the applied limit.
    ///
    /// If admission is denied, this suspends until capacity is available or the
    /// task is cancelled.
    public func start() async -> Int {
        await manager.start(id: id, size: size, policy: policy, kind: kind)
    }

    /// Ends the ticket and returns the applied limit.
    ///
    /// Ending releases its size from the active set and resumes any waiters.
    public func end() async -> Int {
        await manager.end(id: id, policy: policy)
    }
}

extension WiredMemoryTicket {
    /// Convenience wrapper that guarantees start/end pairing and is safe under
    /// cancellation. This is the recommended pattern for inference.
    public static func withWiredLimit<R>(
        _ ticket: WiredMemoryTicket,
        _ body: () async throws -> R
    ) async rethrows -> R {
        _ = await ticket.start()
        return try await withTaskCancellationHandler {
            do {
                let result = try await body()
                _ = await ticket.end()
                return result
            } catch {
                _ = await ticket.end()
                throw error
            }
        } onCancel: {
            Task { _ = await ticket.end() }
        }
    }

    /// Convenience overload for using an instance directly.
    public func withWiredLimit<R>(_ body: () async throws -> R) async rethrows -> R {
        try await Self.withWiredLimit(self, body)
    }
}

/// Central coordinator for process-wide wired memory limits.
///
/// The wired limit is a global resource. This manager serializes updates,
/// performs admission control, and restores the baseline when work completes.
/// Use the shared singleton in production; multiple managers are undefined.
public actor WiredMemoryManager {
    /// Shared singleton used by default for tickets.
    public static let shared = WiredMemoryManager()

    #if DEBUG
        /// Test-only factory to create isolated managers.
        public static func makeForTesting(
            configuration: WiredMemoryManagerConfiguration = .init()
        ) -> WiredMemoryManager {
            WiredMemoryManager(configuration: configuration)
        }

        private var eventContinuations: [UUID: AsyncStream<WiredMemoryEvent>.Continuation] = [:]
        private var eventSequence: UInt64 = 0
    #endif

    /// Book-keeping for a live ticket.
    private struct TicketState {
        let size: Int
        let policyKey: PolicyKey
        let policyLabel: String
        let kind: WiredMemoryTicketKind
    }

    /// Stable grouping key for policies.
    private enum PolicyKey: Hashable {
        case identifier(AnyHashable)
    }

    /// Baseline limit captured before the manager applies any changes.
    private var baseline: Int?
    /// Active tickets keyed by ticket UUID.
    private var tickets: [UUID: TicketState] = [:]
    /// The policy instances currently represented by active tickets.
    private var policies: [PolicyKey: any WiredMemoryPolicy] = [:]
    /// Last limit applied by the manager.
    private var currentLimit: Int?
    /// Admission waiters parked while capacity is unavailable.
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]
    /// Timestamp used to enforce shrink cooldown.
    private var lastLimitChange: Date?
    /// Hysteresis configuration for shrink behavior.
    private var configuration: WiredMemoryManagerConfiguration

    /// True when policy-only mode is enabled on an unsupported backend.
    private var policyOnlyMode: Bool {
        configuration.policyOnlyWhenUnsupported && !WiredMemoryBackend.isSupported
    }

    /// Creates a manager with the given hysteresis configuration.
    ///
    /// Use ``shared`` in production; multiple managers are undefined behavior.
    init(configuration: WiredMemoryManagerConfiguration = .init()) {
        self.configuration = configuration
    }

    /// Update configuration when no tickets or waiters are active.
    public func updateConfiguration(
        _ update: (inout WiredMemoryManagerConfiguration) -> Void
    ) {
        precondition(
            !hasActiveWork() && waiters.isEmpty,
            "Configuration can only be updated when no active tickets are running."
        )
        update(&configuration)
    }

    /// Replace the shared manager configuration when no tickets or waiters exist.
    public static func configureShared(
        _ configuration: WiredMemoryManagerConfiguration
    ) async {
        await shared.updateConfiguration { $0 = configuration }
    }

    /// Debug-only event stream describing admission and limit changes.
    ///
    /// In release builds this stream is empty and finishes immediately.
    public func events() -> AsyncStream<WiredMemoryEvent> {
        #if DEBUG
            return AsyncStream { continuation in
                let id = UUID()
                eventContinuations[id] = continuation
                continuation.onTermination = { _ in
                    Task { await self.removeEventContinuation(id: id) }
                }
            }
        #else
            return AsyncStream { continuation in
                continuation.finish()
            }
        #endif
    }

    /// Main entry point for ticket admission.
    ///
    /// Flow:
    /// 1) Capture the baseline (current wired limit).
    /// 2) If `canAdmit` fails, suspend until capacity is available.
    /// 3) Register the ticket, recompute desired limit, and apply it.
    public func start(
        id: UUID,
        size: Int,
        policy: any WiredMemoryPolicy,
        kind: WiredMemoryTicketKind
    ) async -> Int {
        let normalizedSize = max(0, size)
        let policyOnly = policyOnlyMode
        if !WiredMemoryBackend.isSupported && !policyOnly {
            if baseline == nil {
                baseline = 0
            }
            return baseline ?? 0
        }

        var baselineValue = resolveBaselineAndEmit(refresh: baseline == nil || !hasActiveWork())
        if tickets[id] != nil {
            #if DEBUG
                assertionFailure("Ticket already started: \(id)")
            #endif
            emit(
                kind: .ticketStartIgnored,
                ticketID: id,
                size: normalizedSize,
                baseline: baselineValue
            )
            return currentLimit ?? baselineValue
        }

        let key = policyKey(for: policy)
        let label = policyLabel(for: policy, key: key)

        while !policy.canAdmit(
            baseline: baselineValue,
            activeSizes: activeSizes(for: key),
            newSize: normalizedSize
        ) {
            emit(
                kind: .admissionWait,
                ticketID: id,
                size: normalizedSize,
                policy: label,
                baseline: baselineValue
            )
            if Task.isCancelled {
                return currentLimit ?? baselineValue
            }

            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    waiters[id] = continuation
                }
            } onCancel: { [id] in
                Task { await self.cancelWaiter(id: id) }
            }

            baselineValue = resolveBaselineAndEmit(refresh: baseline == nil || !hasActiveWork())
            if Task.isCancelled {
                return currentLimit ?? baselineValue
            }
        }

        emit(
            kind: .admissionGranted,
            ticketID: id,
            size: normalizedSize,
            policy: label,
            baseline: baselineValue
        )
        tickets[id] = TicketState(
            size: normalizedSize,
            policyKey: key,
            policyLabel: label,
            kind: kind
        )
        policies[key] = policy
        emit(
            kind: .ticketStarted,
            ticketID: id,
            size: normalizedSize,
            policy: label,
            baseline: baselineValue
        )
        applyCurrentLimit()
        return currentLimit ?? baselineValue
    }

    /// Releases a ticket and recomputes the limit.
    ///
    /// If this was the last ticket, the manager restores the baseline and
    /// clears internal state.
    public func end(id: UUID, policy: any WiredMemoryPolicy) -> Int {
        if let waiter = waiters.removeValue(forKey: id) {
            waiter.resume()
        }

        guard WiredMemoryBackend.isSupported || policyOnlyMode else {
            if baseline == nil {
                baseline = 0
            }
            return baseline ?? 0
        }

        guard let state = tickets.removeValue(forKey: id) else {
            #if DEBUG
                assertionFailure("Ticket not active: \(id)")
            #endif
            emit(kind: .ticketEndIgnored, ticketID: id)
            return currentLimit ?? baseline ?? 0
        }

        emit(
            kind: .ticketEnded,
            ticketID: id,
            size: state.size,
            policy: state.policyLabel,
            baseline: baseline
        )
        if !tickets.values.contains(where: { $0.policyKey == state.policyKey }) {
            policies.removeValue(forKey: state.policyKey)
        }

        if tickets.isEmpty {
            let baselineValue = baseline ?? 0
            applyLimitIfNeeded(baselineValue)
            emit(
                kind: .baselineRestored,
                baseline: baselineValue,
                appliedLimit: currentLimit
            )
            baseline = nil
            resumeWaiters()
            return baselineValue
        }

        applyCurrentLimit()
        resumeWaiters()
        return currentLimit ?? baseline ?? 0
    }

    /// Derive a stable grouping key for the policy.
    private func policyKey(for policy: any WiredMemoryPolicy) -> PolicyKey {
        return .identifier(policy.id)
    }

    /// Debug label for event streams.
    private func policyLabel(for policy: any WiredMemoryPolicy, key: PolicyKey) -> String {
        let typeName = String(describing: type(of: policy))
        let suffix: String
        switch key {
        case .identifier(let id):
            suffix = "id=\(String(describing: id))"
        }
        return "\(typeName)#\(suffix)"
    }

    /// Active sizes for tickets belonging to a single policy group.
    private func activeSizes(for key: PolicyKey) -> [Int] {
        tickets.values.compactMap { state in
            state.policyKey == key ? state.size : nil
        }
    }

    /// Emit a debug event to any active continuations.
    private func emit(
        kind: WiredMemoryEvent.Kind,
        ticketID: UUID? = nil,
        size: Int? = nil,
        policy: String? = nil,
        baseline: Int? = nil,
        desiredLimit: Int? = nil,
        appliedLimit: Int? = nil,
        activeCount: Int? = nil,
        waiterCount: Int? = nil
    ) {
        #if DEBUG
            guard !eventContinuations.isEmpty else { return }
            eventSequence &+= 1
            let event = WiredMemoryEvent(
                sequence: eventSequence,
                timestamp: Date(),
                kind: kind,
                ticketID: ticketID,
                size: size,
                policy: policy,
                baseline: baseline,
                desiredLimit: desiredLimit,
                appliedLimit: appliedLimit,
                activeCount: activeCount ?? tickets.count,
                waiterCount: waiterCount ?? waiters.count
            )
            for (_, continuation) in eventContinuations {
                _ = continuation.yield(event)
            }
        #endif
    }

    private func resolveBaseline() -> Int {
        if let override = configuration.baselineOverride {
            return max(0, override)
        }
        // Use the manager's cached limit rather than probing the backend.
        if let currentLimit {
            return currentLimit
        }
        if !WiredMemoryBackend.isSupported && configuration.useRecommendedWorkingSetWhenUnsupported
        {
            #if canImport(Metal)
                if let recommended = GPU.maxRecommendedWorkingSetBytes() {
                    return recommended
                }
            #endif
        }
        return 0
    }

    private func resolveBaselineAndEmit(refresh: Bool) -> Int {
        // The baseline is the limit observed when the manager begins coordinating.
        // We re-resolve it when requested (e.g. after a full drain).
        if let baseline, !refresh {
            return baseline
        }
        let current = resolveBaseline()
        let previous = baseline
        baseline = current
        if previous == nil || previous != current {
            emit(kind: .baselineCaptured, baseline: current, appliedLimit: current)
        }
        return current
    }

    private func desiredLimit() -> Int? {
        // Compute per-policy limits and take the max across policies to
        // coordinate heterogeneous strategies without double-counting.
        guard let baseline else { return nil }
        if tickets.isEmpty {
            return baseline
        }

        var sizesByPolicy: [PolicyKey: [Int]] = [:]
        for state in tickets.values {
            sizesByPolicy[state.policyKey, default: []].append(state.size)
        }

        var desired: Int?
        for (key, sizes) in sizesByPolicy {
            guard let policy = policies[key] else { continue }
            let limit = policy.limit(baseline: baseline, activeSizes: sizes)
            desired = max(desired ?? limit, limit)
        }

        return desired ?? baseline
    }

    private func applyCurrentLimit() {
        // Only raise the wired limit while there is active work. Reservation
        // tickets influence admission but do not keep the limit elevated alone.
        guard let desired = desiredLimit() else { return }
        if !hasActiveWork() {
            if let baseline, currentLimit != baseline {
                applyLimitIfNeeded(baseline)
                emit(
                    kind: .baselineRestored,
                    baseline: baseline,
                    appliedLimit: currentLimit
                )
            }
            return
        }
        emit(
            kind: .limitComputed,
            baseline: baseline,
            desiredLimit: desired
        )
        applyLimitIfNeeded(desired)
    }

    private func applyLimitIfNeeded(_ limit: Int) {
        // Apply hysteresis (threshold + cooldown) to avoid shrinking thrash
        // while active work is running.
        if currentLimit == limit {
            return
        }

        if let currentLimit, limit < currentLimit, !tickets.isEmpty, hasActiveWork() {
            let ratioDrop = Double(currentLimit - limit) / Double(currentLimit)
            if ratioDrop < configuration.shrinkThresholdRatio {
                return
            }

            if let lastLimitChange {
                let elapsed = Date().timeIntervalSince(lastLimitChange)
                if elapsed < configuration.shrinkCooldown {
                    return
                }
            }
        }

        if WiredMemoryBackend.isSupported {
            if WiredMemoryBackend.applyLimit(limit) {
                currentLimit = limit
                lastLimitChange = Date()
                emit(
                    kind: .limitApplied,
                    baseline: baseline,
                    desiredLimit: limit,
                    appliedLimit: currentLimit
                )
            } else {
                emit(
                    kind: .limitApplyFailed,
                    baseline: baseline,
                    desiredLimit: limit,
                    appliedLimit: currentLimit
                )
            }
        } else {
            currentLimit = limit
            lastLimitChange = Date()
            emit(
                kind: .limitApplied,
                baseline: baseline,
                desiredLimit: limit,
                appliedLimit: currentLimit
            )
        }
    }

    /// Resume any tasks that were waiting for admission capacity.
    private func resumeWaiters() {
        guard !waiters.isEmpty else { return }
        let pending = waiters
        waiters.removeAll()
        for (_, continuation) in pending {
            continuation.resume()
        }
    }

    /// Returns true if any active (non-reservation) ticket is present.
    private func hasActiveWork() -> Bool {
        tickets.values.contains { $0.kind == .active }
    }

    /// Cancel an admission waiter and emit a debug event.
    private func cancelWaiter(id: UUID) {
        if let waiter = waiters.removeValue(forKey: id) {
            waiter.resume()
        }
        emit(kind: .admissionCancelled, ticketID: id)
    }

    #if DEBUG
        private func removeEventContinuation(id: UUID) {
            eventContinuations.removeValue(forKey: id)
        }
    #endif
}

extension WiredMemoryPolicy {
    /// Convenience to create a ticket bound to a policy and manager.
    public func ticket(
        size: Int,
        manager: WiredMemoryManager = .shared,
        kind: WiredMemoryTicketKind = .active
    ) -> WiredMemoryTicket {
        WiredMemoryTicket(size: size, policy: self, manager: manager, kind: kind)
    }
}
