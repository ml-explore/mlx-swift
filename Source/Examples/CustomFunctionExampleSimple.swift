import Foundation
import MLX

@main
struct SimpleAddExample {
    static func main() throws {

        // -----------------------------
        // Forward Metal kernel
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

               int base_idx = ((b * H + h) * W + w) * C + c;

               out[base_idx] = x[base_idx] + 1.0f;
            """

        let forwardKernel = MLXFast.metalKernel(
            name: "add_constant_forward",
            inputNames: ["x"],
            outputNames: ["out"],
            source: forwardSource
        )

        // -----------------------------
        // VJP Metal kernel
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
            	x_grad[base_idx] = cotangent[base_idx];
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
        let addConstant = CustomFunction {
            Forward { inputs in
                let x = inputs[0]
                let totalElems = x.shape.reduce(1, *)

                return forwardKernel(
                    [x],
                    grid: (totalElems, 1, 1),
                    threadGroup: (32, 1, 1),
                    outputShapes: [x.shape],
                    outputDTypes: [x.dtype]
                )
            }

            VJP { primals, cotangents in
                let x = primals[0]
                let cot = cotangents[0]
                let totalElems = x.shape.reduce(1, *)

                return vjpKernel(
                    [x, cot],
                    grid: (totalElems, 1, 1),
                    threadGroup: (32, 1, 1),
                    outputShapes: [x.shape],
                    outputDTypes: [x.dtype],
                    initValue: 0
                )
            }
        }

        // -----------------------------
        // Example input
        // -----------------------------
        let data: [Float32] = [1, 2, 3]
        let x = MLXArray(data, [1, 3, 1, 1])
        print("x:", x)

        // -----------------------------
        // Scalar-valued function
        // -----------------------------
        func fn(_ x: MLXArray) -> MLXArray {
            let y = addConstant([x])[0]
            print("y:", y)
            let result = MLX.sum(y)
            print("result:", result)
            return result
        }

        let gradFn = grad(fn)
        let dfdx = gradFn(x)
        print("df/dx:", dfdx)

        // -----------------------------
        // Numerical derivative check
        // -----------------------------
        let eps: Float = 1e-3

        let xp = x
        xp[0, 0, 0, 0] += eps
        let f1 = fn(xp).item(Float.self)

        xp[0, 0, 0, 0] -= 2 * eps
        let f2 = fn(xp).item(Float.self)

        let numeric = (f1 - f2) / (2 * eps)
        print("numeric grad @ (0,0):", numeric)
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

 x: array([[[[1]],[[2]],[[3]]]], dtype=float32)
 y: array([[[[2]],[[3]],[[4]]]], dtype=float32)
 result: array(9, dtype=float32)
 df/dx: array([[[[1]],[[0]],[[0]]]], dtype=float32)
 y: array([[[[2.001]],[[3]],[[4]]]], dtype=float32)
 result: array(9.001, dtype=float32)
 y: array([[[[1.999]],[[3]],[[4]]]], dtype=float32)
 result: array(8.999, dtype=float32)
 numeric grad @ (0,0): 0.9994506
 autodiff grad @ (0,0): array(1, dtype=float32)
 Program ended with exit code: 0
*/
