// Copyright © 2024 Apple Inc.

// Note: this stubs out accelerate/softmax on x86 (e.g. via Release builds)

#if defined(__aarch64__)

#include "../mlx/mlx/backend/accelerate/softmax.cpp"

#else

#include "mlx/primitives.h"

namespace mlx::core {
    void Softmax::eval_cpu(const std::vector<array>& inputs, array& out) {
        Softmax::eval(inputs, out);
    }
}

#endif
