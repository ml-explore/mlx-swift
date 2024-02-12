// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Global random state
public class RandomState: Updatable, Evaluatable {
    private var state: MLXArray

    init() {
        let now = mach_approximate_time()
        state = key(now)
    }

    public func innerState() -> [MLXArray] {
        [state]
    }

    public func next() -> MLXArray {
        let (a, b) = split(key: state)
        self.state = a
        return b
    }

    public func seed(_ seed: UInt64) {
        state = key(seed)
    }
}

/// Global random state.
///
/// This can be used with `compile(state: [MLXRandom.globalState, ...], ...)`
///
/// ### See Also
/// - ``seed(_:)``
public let globalState = RandomState()
