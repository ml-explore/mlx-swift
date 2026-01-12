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
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``asData(access:)``
    public func asMTLBuffer(device: some MTLDevice, noCopy: Bool = false) -> (any MTLBuffer)? {
        self.eval()

        if noCopy && self.contiguousToDimension() == 0 {
            // the backing is contiguous, we can provide a wrapper
            // for the contents without a copy (if requested)
            let source = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(self.ctx))!
            return device.makeBuffer(bytesNoCopy: source, length: self.nbytes)
        } else {
            let data = asDataCopy()
            return data.data.withUnsafeBytes { ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count)
            }
        }
    }

}
