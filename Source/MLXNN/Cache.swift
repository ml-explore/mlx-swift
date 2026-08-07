// Copyright © 2024 Apple Inc.

import Foundation
import Synchronization

/// Simple cache for holding prepared MLXArrays, etc.
///
/// See ``RoPE``
// `@unchecked Sendable`: state is fully Mutex-protected, but `Mutex.withLock`'s
// `inout sending`/`sending` closure boundary independently checks every type
// nested inside the Mutex's protected state -- `Entry` and `State` below need their
// own `@unchecked Sendable` too, not just `Cache` itself. This holds regardless of
// whether a given instantiation's `Element` is itself `Sendable` (see e.g. `ALiBi`'s
// `Cache<Key, MLXArray>`, where it is not).
final class Cache<Key: Hashable, Element>: @unchecked Sendable {

    struct Entry: @unchecked Sendable {
        let value: Element
        let serial: Int
    }

    private struct State: @unchecked Sendable {
        var contents = [Key: Entry]()
        var serial = 0
    }

    let maxSize: Int
    private let state = Mutex(State())

    init(maxSize: Int = 10) {
        self.maxSize = maxSize
    }

    subscript(key: Key) -> Element? {
        get {
            state.withLock { $0.contents[key]?.value }
        }
        set {
            // store the key, value pair keeping the count <= maxSize
            state.withLock { state in
                if let newValue {
                    // handle wrap on the serial number
                    if state.serial == Int.max {
                        state.contents.removeAll()
                        state.serial = 0
                    }
                    state.contents[key] = Entry(value: newValue, serial: state.serial)
                    state.serial += 1

                    // if too large, remove oldest
                    if state.contents.count > maxSize {
                        let minKey = state.contents.min { lhs, rhs in
                            lhs.value.serial < rhs.value.serial
                        }?.key
                        if let minKey {
                            state.contents[minKey] = nil
                        }
                    }
                } else {
                    state.contents[key] = nil
                }
            }
        }
    }
}
