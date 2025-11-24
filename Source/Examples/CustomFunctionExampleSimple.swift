import Foundation
import MLX

@main
struct SimpleAddExample {
    static func main() throws {
        // -----------------------------
        // Forward kernel: Adding a constant
        // -----------------------------
        let forwardSource = """
            uint elem = thread_position_in_grid.x;
            int B = x_shape[0];
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];

            int c = elem % C;
            int w = (elem / C) % W;
            int h = (elem / (C * W)) % H;
            int b = elem / (C * W * H);

            if (b >= B) return;

            int base_idx = b * H * W * C + c;
            out[base_idx] = x[base_idx] + 1.0f;  // Add a constant (1.0f) to each element (use Float32)
            """

        let forwardKernel = MLXFast.metalKernel(
            name: "add_constant_forward",
            inputNames: ["x"],
            outputNames: ["out"],
            source: forwardSource
        )

        // -----------------------------
        // Backward kernel: Derivative of adding a constant
        // -----------------------------
        let vjpSource = """
            uint elem = thread_position_in_grid.x;
            int B = x_shape[0];
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];

            int c = elem % C;
            int w = (elem / C) % W;
            int h = (elem / (C * W)) % H;
            int b = elem / (C * W * H);

            if (b >= B) return;

            int base_idx = b * H * W * C + c;
            x_grad[base_idx] = cotangent[base_idx];  // Derivative of x + 1 is 1, so we just pass the cotangent
            """

        let vjpKernel = MLXFast.metalKernel(
            name: "add_constant_vjp",
            inputNames: ["x", "cotangent"],
            outputNames: ["x_grad"],
            source: vjpSource
        )

        // -----------------------------
        // Custom MLX function
        // -----------------------------
        let addConstantFunction: MLXClosure = {
            @MLXCustomFunctionBuilder
            var f: MLXClosure {
                Forward { inputs in
                    let x = inputs[0]
                    let totalElems = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]

                    let result = forwardKernel(
                        [x],
                        grid: (totalElems, 1, 1),
                        threadGroup: (32, 1, 1),
                        outputShapes: [x.shape],
                        outputDTypes: [x.dtype]
                    )

                    return result
                }

                VJP { primals, cotangents in
                    let x = primals[0]
                    let cot = cotangents[0]
                    let totalElems = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]

                    let result = vjpKernel(
                        [x, cot],
                        grid: (totalElems, 1, 1),
                        threadGroup: (32, 1, 1),
                        outputShapes: [x.shape],
                        outputDTypes: [x.dtype],
                        initValue: 0
                    )

                    return result
                }
            }
            return f
        }()

        // -----------------------------
        // Example input
        // -----------------------------
        let data: [Float32] = [1.0, 2.0, 3.0]  // ensure Float32, as Float64 is not supported
        let x = MLXArray(data, [1, 3, 1, 1])
        print("x:", x)

        // https://github.com/ml-explore/mlx/discussions/842#discussioncomment-8835095
        // The output of the model needs to be a scalar
        func fn(_ x: MLXArray) -> MLXArray {
            let y = try! MLXClosure.apply(addConstantFunction, [x])[0]
            let result = MLX.sum(y)
            print("y:", y)
            print("result:", result)
            return result
        }

        let gradFn = grad(fn)
        let dfdx = gradFn(x)
        print("df/dx:", dfdx)

        // -----------------------------
        // Numerical check
        // -----------------------------
        let eps: Float = 1e-3
        let xPerturbed = x
        xPerturbed[0, 0, 0, 0] += eps
        let f1 = fn(xPerturbed).item(Float.self)
        xPerturbed[0, 0, 0, 0] -= 2 * eps
        let f2 = fn(xPerturbed).item(Float.self)
        let numericGrad = (f1 - f2) / (2 * eps)
        print("numeric grad @ (0,0):", numericGrad)
        print("autodiff grad @ (0,0):", dfdx[0, 0, 0, 0])
    }
}

/*
## gives following output:

 x: array([[[[1]],[[2]],[[3]]]], dtype=float32)
 y: array([[[[2]],[[0]],[[0]]]], dtype=float32)
 result: array(2, dtype=float32)
 df/dx: array([[[[1]],[[0]],[[0]]]], dtype=float32)
 y: array([[[[2.001]],[[0]],[[0]]]], dtype=float32)
 result: array(2.001, dtype=float32)
 y: array([[[[1.999]],[[0]],[[0]]]], dtype=float32)
 result: array(1.999, dtype=float32)
 numeric grad @ (0,0): 0.99992746
 autodiff grad @ (0,0): array(1, dtype=float32)
*/
