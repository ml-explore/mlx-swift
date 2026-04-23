// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// API for controlling GPU related features.
///
/// Note: previously this also had properties that managed the buffer use -- those are now
/// found in ``Memory`` but remain here as deprecated.
///
/// ### See Also
/// - <doc:running-on-ios>
/// - ``Memory``
public enum GPU {

    /// Returns GPU's recommended working set size in bytes as an `Int`.
    ///
    /// This value is derived from ``DeviceInfo/maxRecommendedWorkingSetSize`` and
    /// is clamped to `Int.max` when necessary. Returns `nil` when unavailable.
    public static func maxRecommendedWorkingSetBytes() -> Int? {
        return nil
    }
}
