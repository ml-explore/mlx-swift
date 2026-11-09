// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation
import Metal

// MARK: - Metal

extension MLXArray {

    /// Return the contents as a Metal buffer in the native ``dtype``.
    ///
    /// > If you can guarantee the lifetime of the ``MLXArray`` will exceed the MTLBuffer and that
    /// the array will not be mutated (e.g. using indexing or other means) it is possible to pass `noCopy: true`
    /// to reference the backing bytes.
    ///
    /// Returns `nil` if the array has no elements (Metal cannot create a zero-length buffer).
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asData(access:)``
    public func asMTLBuffer(device: some MTLDevice, noCopy: Bool = false) -> (any MTLBuffer)? {
        self.eval()

        // an empty array has no backing pointer and Metal cannot make a
        // zero-length buffer
        if self.size == 0 {
            return nil
        }

        if noCopy && self.contiguousToDimension() == 0 {
            // the backing is contiguous, we can provide a wrapper
            // for the contents without a copy (if requested)
            let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
            return device.makeBuffer(bytesNoCopy: source, length: self.nbytes)
        } else {
            let data = asDataCopy()
            return data.data.withUnsafeBytes { ptr in
                guard let base = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: base, length: ptr.count)
            }
        }
    }

}
