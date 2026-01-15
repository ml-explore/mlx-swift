// Copyright © 2024 Apple Inc.

import Foundation

/// Protocol for types that can be used as a provider of random keys, e.g. for ``MLXRandom``.
public protocol RandomStateOrKey {
    func asRandomKey() -> MLXArray
}

extension MLXArray: RandomStateOrKey {
    public func asRandomKey() -> MLXArray {
        self
    }
}

extension MLXRandom {

    /// Random state factory.
    ///
    ///
    ///
    /// Note: although this type is thread-safe, the MLXArrays that it returns are not -- do not
    /// evaluate these values or expressions that depend on them across multiple threads
    /// simultaneously.
    ///
    /// ### See Also
    /// - ``globalState``
    /// - ``withRandomState(_:body:)-18ob4``
    public class RandomState: RandomStateOrKey, Updatable, Evaluatable, @unchecked (Sendable) {
        private var state: MLXArray
        private let lock = NSLock()

        /// Initialize the RandomState with a seed based on the current time.
        public init() {
            let now = DispatchTime.now().uptimeNanoseconds
            state = MLXRandom.key(now)
        }

        /// Initialize the RandomState with the given seed value.
        public init(seed: UInt64) {
            state = MLXRandom.key(seed)
        }

        public func innerState() -> [MLXArray] {
            lock.withLock {
                [state]
            }
        }

        /// Split the current state and return a new Key.
        public func next() -> MLXArray {
            lock.withLock {
                let (a, b) = MLXRandom.split(key: state)
                self.state = a
                return b
            }
        }

        /// Reset the random state.
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

    /// See ``withRandomState(_:body:)`` and ``resolve(key:)``
    @TaskLocal
    static fileprivate var taskLocalRandomState: MLXRandom.RandomState?

}  // MLXRandom

/// Resolve the given key to a concrete MLXArray (random key).
///
/// This will use the following values (in order until one is found) to resolve the
/// random key:
///
/// - the passed key, either an ``MLXArray`` or ``MLXRandom/RandomState``
/// - the task-local ``MLXRandom/RandomState``, see ``withRandomState(_:body:)-18ob4``
/// - the global RandomState, ``MLXRandom/globalState``
public func resolve(key: (some RandomStateOrKey)? = MLXArray?.none) -> MLXArray {
    key?.asRandomKey() ?? MLXRandom.taskLocalRandomState?.asRandomKey()
        ?? MLXRandom.globalState.next()
}

/// Use the given ``MLXRandom/RandomState`` scoped to the current task and body.
public func withRandomState<R>(_ state: MLXRandom.RandomState, body: () throws -> R) rethrows -> R {
    try MLXRandom.$taskLocalRandomState.withValue(state, operation: body)
}

/// Use the given ``MLXRandom/RandomState`` scoped to the current task and body.
public func withRandomState<R>(_ state: MLXRandom.RandomState, body: () async throws -> R)
    async rethrows -> R
{
    try await MLXRandom.$taskLocalRandomState.withValue(state, operation: body)
}
