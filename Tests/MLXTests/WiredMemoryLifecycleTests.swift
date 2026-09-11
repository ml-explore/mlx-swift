// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

final class WiredMemoryLifecycleTests: XCTestCase {
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

    private static func manager() -> WiredMemoryManager {
        .makeForTesting(
            configuration: .init(
                policyOnlyWhenUnsupported: true, baselineOverride: 1024,
                useRecommendedWorkingSetWhenUnsupported: false))
    }

    func testDuplicateStartAndUnmatchedEndAreIdempotent() async {
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
            XCTAssertEqual(started, 1)
            XCTAssertEqual(ignoredStarts, 1)
            XCTAssertEqual(ended, 1)
        }
    }

    func testCancellationWhileAdmissionIsDeniedStillRunsCleanup() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 1, policy: Policy(capacity: 0), manager: manager)
            let events = await manager.events()
            let task = Task {
                await ticket.withWiredLimit {
                    XCTAssertTrue(Task.isCancelled)
                    return 123
                }
            }
            for await event in events {
                if event.kind == .admissionWait { break }
            }
            task.cancel()
            let result = await task.value
            XCTAssertEqual(result, 123)
            // The cancelled waiter must be removed before the scope returns.
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    func testCancellationKeepsTicketUntilBodyFinishes() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 64, policy: Policy(capacity: 64), manager: manager)
            let events = await manager.events()
            let gate = Gate()
            let started = expectation(description: "body started")
            let endedEarly = expectation(description: "ticket ended before body completed")
            endedEarly.isInverted = true
            let observer = Task {
                for await event in events {
                    if event.kind == .ticketEnded {
                        endedEarly.fulfill()
                        break
                    }
                }
            }
            let task = Task {
                await ticket.withWiredLimit {
                    started.fulfill()
                    await gate.wait()
                    XCTAssertTrue(Task.isCancelled)
                }
            }
            await fulfillment(of: [started], timeout: 5)
            task.cancel()
            await fulfillment(of: [endedEarly], timeout: 0.2)
            observer.cancel()
            await observer.value
            await gate.open()
            await task.value
            // End is awaited; no active ticket may remain after task.value.
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    func testThrowingBodyReleasesTicketBeforeReturning() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 64, policy: Policy(capacity: 64), manager: manager)
            do {
                try await ticket.withWiredLimit { throw Failure.expected }
                XCTFail("Expected the body's error")
            } catch {
                XCTAssertTrue(error is Failure)
            }
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    func testAlreadyCancelledScopeCanCleanUpWithoutAdmission() async {
        await Device.withDefaultDevice(.cpu) {
            let manager = Self.manager()
            let ticket = WiredMemoryTicket(size: 1, policy: Policy(capacity: 0), manager: manager)
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return await ticket.withWiredLimit {
                    XCTAssertTrue(Task.isCancelled)
                    return 123
                }
            }
            let result = await task.value
            XCTAssertEqual(result, 123)
            await manager.updateConfiguration { $0.baselineOverride = 2048 }
        }
    }

    func testCancellationRacingCompletionAlwaysFinishesCleanup() async {
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
                XCTAssertEqual(result, index)
                await manager.updateConfiguration { $0.baselineOverride = 1024 }
            }
        }
    }

    func testConcurrentStartsOfWaitingTicketDoNotLoseWaiters() async {
        await checkConcurrentStarts(cancelFirst: false)
    }

    func testCancellingOneStartDoesNotCancelAnotherWaiterForSameTicket() async {
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
            let firstFinished = expectation(description: "first start returned")
            let secondFinished = expectation(description: "second start returned")
            let first = Task {
                _ = await ticket.start()
                firstFinished.fulfill()
            }
            // Ensure the first waiter is installed before adding the second.
            for await event in events {
                if event.kind == .admissionWait { break }
            }
            let secondEvents = await manager.events()
            let second = Task {
                _ = await ticket.start()
                secondFinished.fulfill()
            }
            for await event in secondEvents {
                if event.kind == .admissionWait { break }
            }
            if cancelFirst {
                first.cancel()
                let result = await XCTWaiter.fulfillment(of: [firstFinished], timeout: 5)
                guard result == .completed else {
                    XCTFail("Cancelling the first start did not resume its continuation")
                    second.cancel()
                    _ = await blocker.end()
                    return
                }
            }
            _ = await blocker.end()
            let expected = cancelFirst ? [secondFinished] : [firstFinished, secondFinished]
            let result = await XCTWaiter.fulfillment(of: expected, timeout: 5)
            guard result == .completed else {
                XCTFail("A duplicate start lost an admission continuation")
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
