// Copyright © 2024 Apple Inc.

import Foundation

public protocol RandomStateOrKey {
    func asRandomKey() -> MLXArray
}

extension MLXArray: RandomStateOrKey {
    public func asRandomKey() -> MLXArray {
        self
    }
}

extension MLXRandom {

    /// Global random state.
    ///
    /// Note: although this type is thread-safe, the MLXArrays that it returns are not -- do not
    /// evaluate these values or expressions that depend on them across multiple threads
    /// simultaneously.
    public class RandomState: RandomStateOrKey, Updatable, Evaluatable, @unchecked (Sendable) {
        private var state: MLXArray
        private let lock = NSLock()

        public init() {
            let now = mach_approximate_time()
            state = MLXRandom.key(now)
        }

        public init(seed: UInt64) {
            state = MLXRandom.key(seed)
        }

        public func innerState() -> [MLXArray] {
            lock.withLock {
                [state]
            }
        }

        public func next() -> MLXArray {
            lock.withLock {
                let (a, b) = MLXRandom.split(key: state)
                self.state = a
                return b
            }
        }

        public func seed(_ seed: UInt64) {
            lock.withLock {
                state = MLXRandom.key(seed)
            }
        }

        public func asRandomKey() -> MLXArray {
            next()
        }
    }

    /// Global random state.
    ///
    /// This can be used with `compile(state: [MLXRandom.globalState, ...], ...)`
    ///
    /// ### See Also
    /// - ``seed(_:)``
    public static let globalState = RandomState()

}  // MLXRandom

@TaskLocal
private var taskLocalRandomState: MLXRandom.RandomState?

public func resolve(key: RandomStateOrKey?) -> MLXArray {
    key?.asRandomKey() ?? taskLocalRandomState?.asRandomKey() ?? MLXRandom.globalState.next()
}

public func withRandomState<R>(_ state: MLXRandom.RandomState, body: () throws -> R) rethrows -> R {
    try $taskLocalRandomState.withValue(state, operation: body)
}

public func withRandomState<R>(_ state: MLXRandom.RandomState, body: () async throws -> R)
    async rethrows -> R
{
    try await $taskLocalRandomState.withValue(state, operation: body)
}
