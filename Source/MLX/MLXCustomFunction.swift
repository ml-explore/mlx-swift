import Cmlx
import Foundation

public enum MLXCustomFunctionComponent {
    case forward(([MLXArray]) -> [MLXArray])
    case vjp(([MLXArray], [MLXArray]) -> [MLXArray])
}

public func Forward(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> MLXCustomFunctionComponent {
    .forward(f)
}

public func VJP(_ f: @escaping ([MLXArray], [MLXArray]) -> [MLXArray]) -> MLXCustomFunctionComponent
{
    .vjp(f)
}

final class _CustomFunctionState: @unchecked Sendable {

    private let lock = NSLock()

    private let forwardFn: ([MLXArray]) -> [MLXArray]
    private let vjpFn: (([MLXArray], [MLXArray]) -> [MLXArray])?

    private var forwardClosure: mlx_closure!
    private var vjpClosure: mlx_closure_custom!
    private var jvpClosure: mlx_closure_custom_jvp!
    private var vmapClosure: mlx_closure_custom_vmap!
    private var combined: mlx_closure!

    init(
        forward: @escaping ([MLXArray]) -> [MLXArray],
        vjp: (([MLXArray], [MLXArray]) -> [MLXArray])?
    ) {
        self.forwardFn = forward
        self.vjpFn = vjp
        buildClosures()
    }

    deinit {
        mlx_closure_free(forwardClosure)
        mlx_closure_custom_free(vjpClosure)
        mlx_closure_custom_jvp_free(jvpClosure)
        mlx_closure_custom_vmap_free(vmapClosure)
        mlx_closure_free(combined)
    }

    private func buildClosures() {

        forwardClosure = mlx_closure_new_func_payload(
            { out, inputs, payload in
                let swiftFn =
                    Unmanaged<AnyObject>
                    .fromOpaque(payload!)
                    .takeUnretainedValue()
                    as! ([MLXArray]) -> [MLXArray]

                let inp = mlx_vector_array_values(inputs)
                let result = swiftFn(inp)
                out!.pointee = new_mlx_vector_array(result)
                return 0
            },
            Unmanaged.passRetained(forwardFn as AnyObject).toOpaque()
        ) { ptr in
            Unmanaged<AnyObject>.fromOpaque(ptr!).release()
        }

        if let vjpFn = vjpFn {
            vjpClosure = mlx_closure_custom_new_func_payload(
                { out, primals, cotangents, _, payload in
                    let fn =
                        Unmanaged<AnyObject>
                        .fromOpaque(payload!)
                        .takeUnretainedValue()
                        as! ([MLXArray], [MLXArray]) -> [MLXArray]

                    let p = mlx_vector_array_values(primals)
                    let c = mlx_vector_array_values(cotangents)
                    out!.pointee = new_mlx_vector_array(fn(p, c))
                    return 0
                },
                Unmanaged.passRetained(vjpFn as AnyObject).toOpaque()
            ) { ptr in
                Unmanaged<AnyObject>.fromOpaque(ptr!).release()
            }
        } else {
            vjpClosure = mlx_closure_custom_new()
        }

        jvpClosure = mlx_closure_custom_jvp_new()
        vmapClosure = mlx_closure_custom_vmap_new()

        combined = mlx_closure_new()
        _ = mlx_custom_function(
            &combined,
            forwardClosure,
            vjpClosure,
            jvpClosure,
            vmapClosure
        )
    }

    func call(_ inputs: [MLXArray]) -> [MLXArray] {
        lock.withLock {
            let inVec = new_mlx_vector_array(inputs)
            defer { mlx_vector_array_free(inVec) }

            var outVec = mlx_vector_array_new()
            defer { mlx_vector_array_free(outVec) }

            let status = mlx_closure_apply(&outVec, combined, inVec)
            precondition(status == 0, "mlx_closure_apply failed (\(status))")

            return mlx_vector_array_values(outVec)
        }
    }
}

public enum MLXCustomFunctionBuilder {
    @resultBuilder
    public struct Builder {

        public static func buildBlock(
            _ components: MLXCustomFunctionComponent...
        ) -> ([MLXArray]) -> [MLXArray] {

            var forwardFn: (([MLXArray]) -> [MLXArray])?
            var vjpFn: (([MLXArray], [MLXArray]) -> [MLXArray])?

            for c in components {
                switch c {
                case .forward(let f): forwardFn = f
                case .vjp(let f): vjpFn = f
                }
            }

            guard let f = forwardFn else {
                fatalError("CustomFunction must contain a Forward block")
            }

            let state = _CustomFunctionState(forward: f, vjp: vjpFn)

            return { arrays in state.call(arrays) }
        }
    }
}

public func CustomFunction(
    @MLXCustomFunctionBuilder.Builder _ build: () -> ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    build()
}
