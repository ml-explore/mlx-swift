import Cmlx
import Foundation

// MARK: - Custom Function Component
public enum MLXCustomFunctionComponent {
    case forward(([MLXArray]) -> [MLXArray])
    case vjp(([MLXArray], [MLXArray]) -> [MLXArray])
}

// MARK: - DSL Builders
public func Forward(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> MLXCustomFunctionComponent {
    .forward(f)
}

public func VJP(_ f: @escaping ([MLXArray], [MLXArray]) -> [MLXArray]) -> MLXCustomFunctionComponent
{
    .vjp(f)
}

// MARK: - Result Builder
@resultBuilder
public enum MLXCustomFunctionBuilder {
    public static func buildBlock(_ components: MLXCustomFunctionComponent...) -> MLXClosure {
        var forwardFn: (([MLXArray]) -> [MLXArray])?
        var vjpFn: (([MLXArray], [MLXArray]) -> [MLXArray])?

        for component in components {
            switch component {
            case .forward(let f): forwardFn = f
            case .vjp(let f): vjpFn = f
            }
        }

        guard let fwd = forwardFn else {
            fatalError("MLXCustomFunction must have a Forward block")
        }

        // Wrap Swift forward closure -> mlx_closure
        let forwardClosure = mlx_closure_new_func_payload(
            { out, inputs, payload in
                let swiftFn =
                    Unmanaged<AnyObject>.fromOpaque(payload!).takeUnretainedValue()
                    as! ([MLXArray]) -> [MLXArray]
                let inArrays = withUnsafePointer(to: inputs) {
                    mlx_vector_array_to_swift($0)
                }
                let result = swiftFn(inArrays)
                out!.pointee = swift_to_mlx_vector_array(result)
                return 0
            }, Unmanaged.passRetained(fwd as AnyObject).toOpaque()
        ) { ptr in
            Unmanaged<AnyObject>.fromOpaque(ptr!).release()
        }

        var resultClosure = mlx_closure_new()

        if let vjp = vjpFn {
            // Wrap Swift vjp closure -> mlx_closure_custom
            let vjpClosure = mlx_closure_custom_new_func_payload(
                { out, primals, cotangents, _, payload in
                    let swiftFn =
                        Unmanaged<AnyObject>.fromOpaque(payload!).takeUnretainedValue()
                        as! ([MLXArray], [MLXArray]) -> [MLXArray]
                    let p = withUnsafePointer(to: primals) {
                        mlx_vector_array_to_swift($0)
                    }
                    let c = withUnsafePointer(to: cotangents) {
                        mlx_vector_array_to_swift($0)
                    }
                    let grads = swiftFn(p, c)
                    out!.pointee = swift_to_mlx_vector_array(grads)
                    return 0
                }, Unmanaged.passRetained(vjp as AnyObject).toOpaque()
            ) { ptr in
                Unmanaged<AnyObject>.fromOpaque(ptr!).release()
            }

            // empty closures
            let jvpClosure = mlx_closure_custom_jvp_new()
            let vmapClosure = mlx_closure_custom_vmap_new()

            _ = mlx_custom_function(
                &resultClosure, forwardClosure, vjpClosure, jvpClosure, vmapClosure)
        } else {
            // empty closures
            let vjpClosure = mlx_closure_custom_new()
            let jvpClosure = mlx_closure_custom_jvp_new()
            let vmapClosure = mlx_closure_custom_vmap_new()

            _ = mlx_custom_function(
                &resultClosure, forwardClosure, vjpClosure, jvpClosure, vmapClosure)
        }

        return MLXClosure(resultClosure)
    }
}

// MARK: - Example: bridging helpers
// Convert a C mlx_vector_array pointer to a Swift array of MLXArray
func mlx_vector_array_to_swift(_ v: UnsafePointer<mlx_vector_array>?) -> [MLXArray] {
    guard let v = v else { return [] }
    var result: [MLXArray] = []

    // Get size from C API
    let size = mlx_vector_array_size(v.pointee)
    for i in 0 ..< size {
        var cArray = mlx_array()  // placeholder for the individual mlx_array
        mlx_vector_array_get(&cArray, v.pointee, i)
        result.append(MLXArray(cArray))
    }

    return result
}

// Convert Swift array of MLXArray back to C mlx_vector_array
func swift_to_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    var cArrays: [mlx_array] = arrays.map { $0.ctx }
    return mlx_vector_array_new_data(&cArrays, cArrays.count)
}
