import Foundation
import MLX

@main
struct GridSampleExample {
    static func main() throws {
        // -----------------------------
        // Forward kernel: bilinear grid_sample
        // -----------------------------
        let forwardSource = """
            uint elem = thread_position_in_grid.x;

            int B = x_shape[0];
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            // Properly decode batch/channel/grid indices
            int c = elem % C;
            int w = (elem / C) % gW;
            int h = (elem / (C * gW)) % gH;
            int b = elem / (C * gW * gH);

            if (b >= B) return;

            uint grid_idx = ((b * gH + h) * gW + w) * 2;
            float ix = ((grid[grid_idx] + 1) * (W - 1)) / 2.0;
            float iy = ((grid[grid_idx + 1] + 1) * (H - 1)) / 2.0;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;
            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;
            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            float nw = (ix_se - ix) * (iy_se - iy);
            float ne = (ix - ix_sw) * (iy_sw - iy);
            float sw = (ix_ne - ix) * (iy - iy_ne);
            float se = (ix - ix_nw) * (iy - iy_nw);

            int base_idx = b * b_stride + c;
            float I_nw = 0.0;
            float I_ne = 0.0;
            float I_sw = 0.0;
            float I_se = 0.0;

            if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
            	I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
            if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W)
            	I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
            if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W)
            	I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
            if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W)
            	I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

            int out_idx = ((b * gH + h) * gW + w) * C + c;
            out[out_idx] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
            """

        let forwardKernel = MLXFast.metalKernel(
            name: "grid_sample_forward",
            inputNames: ["x", "grid"],
            outputNames: ["out"],
            source: forwardSource
        )

        // -----------------------------
        // Backward kernel: VJP
        // -----------------------------
        let vjpSource = """
            uint elem = thread_position_in_grid.x;

            int B = x_shape[0];
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            int c = elem % C;
            int w = (elem / C) % gW;
            int h = (elem / (C * gW)) % gH;
            int b = elem / (C * gW * gH);

            if (b >= B) return;

            uint grid_idx = ((b * gH + h) * gW + w) * 2;
            float ix = ((grid[grid_idx] + 1) * (W - 1)) / 2.0;
            float iy = ((grid[grid_idx + 1] + 1) * (H - 1)) / 2.0;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);
            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;
            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;
            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            float dx = ix - ix_nw;
            float dy = iy - iy_nw;

            float nw = (1 - dx) * (1 - dy);
            float ne = dx * (1 - dy);
            float sw = (1 - dx) * dy;
            float se = dx * dy;

            int base_idx = b * b_stride + c;

            // -------------------------
            // Accumulate x_grad
            // -------------------------
            if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W)
             atomic_fetch_add_explicit(&x_grad[base_idx + iy_nw * h_stride + ix_nw * w_stride], nw, memory_order_relaxed);
            if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W)
             atomic_fetch_add_explicit(&x_grad[base_idx + iy_ne * h_stride + ix_ne * w_stride], ne, memory_order_relaxed);
            if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W)
             atomic_fetch_add_explicit(&x_grad[base_idx + iy_sw * h_stride + ix_sw * w_stride], sw, memory_order_relaxed);
            if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W)
             atomic_fetch_add_explicit(&x_grad[base_idx + iy_se * h_stride + ix_se * w_stride], se, memory_order_relaxed);

            // -------------------------
            // Compute grid_grad (∂L/∂grid)
            // -------------------------
            float gix = 0.0;
            float giy = 0.0;

            if (iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W) {
             float val = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
             gix += -(1 - dy) * val;
             giy += -(1 - dx) * val;
            }
            if (iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W) {
             float val = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
             gix += (1 - dy) * val;
             giy += -dx * val;
            }
            if (iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W) {
             float val = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
             gix += -dy * val;
             giy += (1 - dx) * val;
            }
            if (iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W) {
             float val = x[base_idx + iy_se * h_stride + ix_se * w_stride];
             gix += dy * val;
             giy += dx * val;
            }

            // Normalize to [-1,1] coordinates
            gix *= 2.0 / float(W - 1);
            giy *= 2.0 / float(H - 1);

            atomic_fetch_add_explicit(&grid_grad[grid_idx], gix, memory_order_relaxed);
            atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy, memory_order_relaxed);
            """

        let vjpKernel = MLXFast.metalKernel(
            name: "grid_sample_vjp",
            inputNames: ["x", "grid", "cotangent"],
            outputNames: ["x_grad", "grid_grad"],
            source: vjpSource,
            atomicOutputs: true
        )

        // -----------------------------
        // Custom MLX function
        // -----------------------------
        let gridSample = CustomFunction {
            Forward { inputs in
                let x = inputs[0]
                let grid = inputs[1]
                let totalElems = x.shape[0] * grid.shape[1] * grid.shape[2] * x.shape[3]

                // Timing the forward pass
                let startTime = Date()
                let result = forwardKernel(
                    [x, grid],
                    grid: (totalElems, 1, 1),
                    threadGroup: (32, 1, 1),
                    outputShapes: [x.shape],
                    outputDTypes: [x.dtype]
                )

                let elapsedTime = Date().timeIntervalSince(startTime)
                print("Forward pass time: \(elapsedTime) seconds")

                return result
            }

            VJP { primals, cotangents in
                let x = primals[0]
                let grid = primals[1]
                let cot = cotangents[0]
                let totalElems = x.shape[0] * grid.shape[1] * grid.shape[2] * x.shape[3]

                // Timing the backward pass
                let startTime = Date()
                let result = vjpKernel(
                    [x, grid, cot],
                    grid: (totalElems, 1, 1),
                    threadGroup: (32, 1, 1),
                    outputShapes: [x.shape, grid.shape],
                    outputDTypes: [x.dtype, grid.dtype],
                    initValue: 0
                )

                let elapsedTime = Date().timeIntervalSince(startTime)
                print("Backward pass time: \(elapsedTime) seconds")

                return result
            }
        }

        // -----------------------------
        // Example input
        // -----------------------------
        let x = MLXArray(stride(from: Float(1), through: 16, by: 1), [1, 4, 4, 1])
        let grid = MLXArray(stride(from: Float(-1), through: 1, by: 0.5), [1, 5, 1, 1])

        print("x:", x)
        print("grid:", grid)

        // https://github.com/ml-explore/mlx/discussions/842#discussioncomment-8835095
        // The output of the model needs to be a scalar
        func fn(_ x: MLXArray) -> MLXArray {
            let y = gridSample([x, grid])[0]
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
        xPerturbed[0, 1, 1, 0] += eps
        let f1 = fn(xPerturbed).item(Float.self)
        xPerturbed[0, 1, 1, 0] -= 2 * eps
        let f2 = fn(xPerturbed).item(Float.self)
        let numericGrad = (f1 - f2) / (2 * eps)
        print("numeric grad @ (1,1):", numericGrad)
        print("autodiff grad @ (1,1):", dfdx[0, 1, 1, 0])
    }
}

/*
 ## gives following output:

 x: array([[[[1],[2],[3],[4]],
		 [[5],[6],[7],[8]],
		 [[9],[10],[11],[12]],
		 [[13],[14],[15],[16]]]], dtype=float32)
 grid: array([[[[-1]],[[-0.5]],[[0]],[[0.5]],[[1]]]], dtype=float32)
 Forward pass time: 7.796287536621094e-05 seconds
 y: array([[[[4],[11.5],[10],[8.5]],
		 [[8.5],[0],[0],[0]],
		 [[0],[0],[0],[0]],
		 [[0],[0],[0],[0]]]], dtype=float32)
 result: array(42.5, dtype=float32)
 Backward pass time: 4.303455352783203e-05 seconds
 df/dx: array([[[[0.25],[0],[0],[0]],
		 [[0.75],[0.5],[0.5],[0.5]],
		 [[0],[0.875],[0.875],[0.5]],
		 [[0],[0.125],[0.125],[0]]]], dtype=float32)
 Forward pass time: 1.4066696166992188e-05 seconds
 y: array([[[[4],[11.5],[10],[8.50025]],
		 [[8.50025],[0],[0],[0]],
		 [[0],[0],[0],[0]],
		 [[0],[0],[0],[0]]]], dtype=float32)
 result: array(42.5005, dtype=float32)
 Forward pass time: 1.0967254638671875e-05 seconds
 y: array([[[[4],[11.5],[10],[8.49975]],
		 [[8.49975],[0],[0],[0]],
		 [[0],[0],[0],[0]],
		 [[0],[0],[0],[0]]]], dtype=float32)
 result: array(42.4995, dtype=float32)
 numeric grad @ (1,1): 0.50354004
 autodiff grad @ (1,1): array(0.5, dtype=float32)
 Program ended with exit code: 0

*/
