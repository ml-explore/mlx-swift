// Copyright © 2025 Apple Inc.

#include <TargetConditionals.h>

// `JitCompiler` shells out via `std::system()`, which is unavailable on iOS and
// visionOS. It is only referenced from `backend/cpu/compiled.cpp`, and
// `compiled_conditional.cpp` selects `no_cpu/compiled.cpp` on those platforms,
// so it isn't needed there. Keep this condition in sync with
// `compiled_conditional.cpp`.
#if !(TARGET_OS_IOS || TARGET_OS_VISION)
#include "../mlx/mlx/backend/cpu/jit_compiler.cpp"
#endif
