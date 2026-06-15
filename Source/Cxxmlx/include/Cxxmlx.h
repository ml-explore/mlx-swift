#pragma once

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/device.h>
#include <mlx/dtype.h>
#include <mlx/einsum.h>
#include <mlx/fast.h>
#include <mlx/fft.h>
#include <mlx/io.h>
#include <mlx/linalg.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>
#include <mlx/utils.h>
#include <mlx/version.h>

#if defined(__APPLE__)
#include <mlx/backend/metal/metal.h>
#endif
