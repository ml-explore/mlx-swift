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
