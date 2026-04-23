import Foundation

enum EncudaError: Error, CustomStringConvertible {
    case nvccFailed(Int32)
    case clangFailed(Int32)

    var description: String {
        switch self {
            case .nvccFailed(let code): return "nvcc failed with exit code \(code)"
            case .clangFailed(let code): return "clang++ failed with exit code \(code)"
        }
    }
}
