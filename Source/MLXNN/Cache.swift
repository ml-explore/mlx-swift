// Copyright © 2024 Apple Inc.

import Foundation

/// Simple cache for holding prepared MLXArrays, etc.
///
/// See ``RoPE``
class Cache<Key: Hashable, Element>: @unchecked (Sendable) {

    let queue = DispatchQueue(label: "Cache")

    let maxSize: Int

    struct Entry {
        let value: Element
        let serial: Int
    }

    var contents = [Key: Entry]()
    var serial = 0

    init(maxSize: Int = 10) {
        self.maxSize = maxSize
    }

    subscript(key: Key) -> Element? {
        get {
            queue.sync {
                contents[key]?.value
            }
        }
        set {
            // store the key, value pair keeping the count <= maxSize
            queue.sync {
                if let newValue {
                    // handle wrap on the serial number
                    if serial == Int.max {
                        contents.removeAll()
                        serial = 0
                    }
                    contents[key] = Entry(value: newValue, serial: serial)
                    serial += 1

                    // if too large, remove oldest
                    if contents.count > maxSize {
                        let minKey = contents.min { lhs, rhs in
                            lhs.value.serial < rhs.value.serial
                        }?.key
                        if let minKey {
                            contents[minKey] = nil
                        }
                    }
                } else {
                    contents[key] = nil
                }
            }
        }
    }
}
