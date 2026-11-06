// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct WiredMemoryLifecycleTests {
    private enum Failure: Error { case expected }
    private struct Policy: WiredMemoryPolicy, Hashable {
        let capacity: Int

        func limit(baseline: Int, activeSizes: [Int]) -> Int {
            baseline + activeSizes.reduce(0, +)
        }

        func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
            activeSizes.reduce(0, +) + newSize <= capacity
        }
    }

    private actor Gate {
        private var isOpen = false
        private var continuation: CheckedContinuation<Void, Never>?

        func wait() async {
            if isOpen { return }
            await withCheckedContinuation { continuation = $0 }
        }

        func open() {
            isOpen = true
            continuation?.resume()
            continuation = nil
        }
    }

    /// Wait for `signal` to yield, giving up after `timeout`.
    ///
    /// - Returns: `true` if the signal arrived in time
    private static func wait(
        for signal: AsyncStream<Void>, timeout: Duration = .seconds(5)
    ) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                // iteration ends (returning nil) if this task is cancelled
                for await _ in signal { return true }
                return false
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return false
            }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }
    }

    private static func manager() -> WiredMemoryManager {
        .makeForTesting(
            configuration: .init(
                policyOnlyWhenUnsupported: true, baselineOverride: 1024,
                useRecommendedWorkingSetWhenUnsupported: false))
    }

    @Test func duplicateStartAndUnmatchedEndAreIdempotent() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 64, policy: Policy(capacity: 64), manager: manager)
            let events = await manager.events()
            _ = await ticket.end()
            _ = await ticket.start()
            _ = await ticket.start()
            _ = await ticket.end()
            _ = await ticket.end()

            var started = 0
            var ignoredStarts = 0
            var ended = 0
            var ignoredEnds = 0
            for await event in events {
                switch event.kind {
                case .ticketStarted: started += 1
                case .ticketStartIgnored: ignoredStarts += 1
                case .ticketEnded: ended += 1
                case .ticketEndIgnored: ignoredEnds += 1
                default: break
                }
                if ignoredEnds == 2 { break }
            }
            #expect(started == 1)
            #expect(ignoredStarts == 1)
            #expect(ended == 1)
        }
    }

    @Test func cancellationWhileAdmissionIsDeniedStillRunsCleanup() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 1, policy: Policy(capacity: 0), manager: manager)
            let events = await manager.events()
            let task = Task {
                await ticket.withWiredLimit {
                    #expect(Task.isCancelled)
                    return 123
                }
            }
            for await event in events {
                if event.kind == .admissionWait { break }
            }
            task.cancel()
            let result = await task.value
            #expect(result == 123)
            // The cancelled waiter must be removed before the scope returns.
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    @Test func cancellationKeepsTicketUntilBodyFinishes() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 64, policy: Policy(capacity: 64), manager: manager)
            let events = await manager.events()
            let gate = Gate()
            let started = AsyncStream<Void>.makeStream()
            // true if the ticket ended before the observer was cancelled
            let observer = Task {
                for await event in events where event.kind == .ticketEnded {
                    return true
                }
                return false
            }
            let task = Task {
                await ticket.withWiredLimit {
                    started.continuation.yield()
                    await gate.wait()
                    #expect(Task.isCancelled)
                }
            }
            #expect(await Self.wait(for: started.stream), "body started")
            task.cancel()
            // give a premature end a chance to show up
            try? await Task.sleep(for: .milliseconds(200))
            observer.cancel()
            let endedEarly = await observer.value
            #expect(!endedEarly, "ticket ended before body completed")
            await gate.open()
            await task.value
            // End is awaited; no active ticket may remain after task.value.
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    @Test func throwingBodyReleasesTicketBeforeReturning() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 64, policy: Policy(capacity: 64), manager: manager)
            await #expect(throws: Failure.self, "Expected the body's error") {
                try await ticket.withWiredLimit { throw Failure.expected }
            }
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    @Test func alreadyCancelledScopeCanCleanUpWithoutAdmission() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 1, policy: Policy(capacity: 0), manager: manager)
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return await ticket.withWiredLimit {
                    #expect(Task.isCancelled)
                    return 123
                }
            }
            let result = await task.value
            #expect(result == 123)
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    @Test func cancellationRacingCompletionAlwaysFinishesCleanup() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            for index in 0 ..< 100 {
                let ticket = WiredMemoryTicket(
                    size: 64, policy: Policy(capacity: 64), manager: manager)
                let task = Task {
                    await ticket.withWiredLimit {
                        await Task.yield()
                        return index
                    }
                }
                if index.isMultiple(of: 2) { await Task.yield() }
                task.cancel()
                let result = await task.value
                #expect(result == index)
                await manager.updateConfiguration { $0.baselineOverride = 1024 }
            }
        }
    }

    @Test func concurrentStartsOfWaitingTicketDoNotLoseWaiters() async {
        await checkConcurrentStarts(cancelFirst: false)
    }

    @Test func cancellingOneStartDoesNotCancelAnotherWaiterForSameTicket() async {
        await checkConcurrentStarts(cancelFirst: true)
    }

    private func checkConcurrentStarts(cancelFirst: Bool) async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let policy = Policy(capacity: 64)
            let blocker = WiredMemoryTicket(size: 64, policy: policy, manager: manager)
            let ticket = WiredMemoryTicket(size: 32, policy: policy, manager: manager)
            _ = await blocker.start()
            let events = await manager.events()
            let firstFinished = AsyncStream<Void>.makeStream()
            let secondFinished = AsyncStream<Void>.makeStream()
            let first = Task {
                _ = await ticket.start()
                firstFinished.continuation.yield()
            }
            // Ensure the first waiter is installed before adding the second.
            for await event in events {
                if event.kind == .admissionWait { break }
            }
            let secondEvents = await manager.events()
            let second = Task {
                _ = await ticket.start()
                secondFinished.continuation.yield()
            }
            for await event in secondEvents {
                if event.kind == .admissionWait { break }
            }
            if cancelFirst {
                first.cancel()
                guard await Self.wait(for: firstFinished.stream) else {
                    Issue.record("Cancelling the first start did not resume its continuation")
                    second.cancel()
                    _ = await blocker.end()
                    return
                }
            }
            _ = await blocker.end()
            // when cancelFirst the first start already finished above
            let firstCompleted = cancelFirst ? true : await Self.wait(for: firstFinished.stream)
            let secondCompleted = await Self.wait(for: secondFinished.stream)
            guard firstCompleted && secondCompleted else {
                Issue.record("A duplicate start lost an admission continuation")
                first.cancel()
                second.cancel()
                return
            }
            await first.value
            await second.value
            _ = await ticket.end()
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }
}
