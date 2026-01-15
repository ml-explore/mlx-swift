// Copyright © 2024 Apple Inc.

import Foundation

/// An object that can provide a list of the ``MLXArray`` in its inner state.
///
/// Note that the array itself is not a reference to the inner state, but the ``MLXArray`` instances
/// can be ``MLXArray/_updateInternal(_:)`` to mutate the inner state.  The exact working is an
/// implemention detail for MLX and should not be depended on by outside callers.
///
/// ### See Also
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public protocol Updatable {
    func innerState() -> [MLXArray]
}

/// An object that can be passed to ``eval(_:)-(Collection<MLXArray>)`` and can produce a list
/// of interior ``MLXArray`` to be evaluated.
///
/// Types such as:
///
/// - ``MLXArray``
/// - `MLXRandom.state`
/// - `MLXNN.Module`
/// - `MLXOptimizers.Optimizer`
///
/// Are all evaluatable.
public protocol Evaluatable {
    func innerState() -> [MLXArray]
}

extension [MLXArray]: Updatable, Evaluatable {
    public func innerState() -> [MLXArray] {
        self
    }
}
