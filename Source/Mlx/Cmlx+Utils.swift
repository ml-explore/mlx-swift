import Foundation
import Cmlx

@inline(__always)
func mlx_free(_ ptr: OpaquePointer) {
    mlx_free(UnsafeMutableRawPointer(ptr))
}

@inline(__always)
func mlx_retain(_ ptr: OpaquePointer) {
    mlx_retain(UnsafeMutableRawPointer(ptr))
}

func describeMLX(_ ptr: OpaquePointer) -> String? {
    if let cDsc = Cmlx.mlx_tostring(UnsafeMutableRawPointer(ptr)) {
        defer { free(cDsc) }
        return String(cString: cDsc)
    }
    
    return nil
}
