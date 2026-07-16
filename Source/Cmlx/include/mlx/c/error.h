/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ERROR_H
#define MLX_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_error Error management
 */
/**@{*/

/* -------------------------------------------------------------------------
 * Legacy push-based handler API (unchanged, retained for source compat).
 * ---------------------------------------------------------------------- */

typedef void (*mlx_error_handler_func)(const char* msg, void* data);

/**
 * Set the error handler.
 */
void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));

/**
 * Throw an error.
 */
void _mlx_error(const char* file, const int line, const char* fmt, ...);

/**
 * Throw an error. Macro which passes file name and line number to _mlx_error().
 */
#define mlx_error(...) _mlx_error(__FILE__, __LINE__, __VA_ARGS__)

/* -------------------------------------------------------------------------
 * New pull-based, structured, thread-local error state.
 *
 * Every generated binding stores the error here (in addition to invoking the
 * legacy handler) before returning a non-zero status. A language binding that
 * checks the status can then pull a *typed* error out of this thread's slot and
 * surface it natively (e.g. a Swift `throw`) instead of relying on a global
 * callback that has no frame to throw from.
 * ---------------------------------------------------------------------- */

/**
 * Error classification, derived from the C++ exception type (and, until
 * mx::core grows typed exceptions, from message inspection for OOM/IO).
 */
typedef enum mlx_error_code_ {
  MLX_ERROR_NONE = 0,
  MLX_ERROR_INVALID_ARGUMENT, /* std::invalid_argument: shape/dtype/axis */
  MLX_ERROR_OUT_OF_RANGE,     /* std::out_of_range: indexing */
  MLX_ERROR_OUT_OF_MEMORY,    /* std::bad_alloc / Metal allocation failure */
  MLX_ERROR_IO,               /* load/save/format failures */
  MLX_ERROR_RUNTIME,          /* std::runtime_error and other std::exception */
  MLX_ERROR_UNKNOWN           /* catch (...) : non-std throw */
} mlx_error_code;

/**
 * Code of the most recent error on the *calling thread*, or MLX_ERROR_NONE.
 * Does not clear the state.
 */
mlx_error_code mlx_last_error_code(void);

/**
 * Message of the most recent error on the calling thread, or "".
 * The returned pointer is owned by MLX and remains valid until the next failing
 * MLX call on this thread or a call to mlx_clear_last_error().
 */
const char* mlx_last_error_message(void);

/**
 * Clear the calling thread's error state. Bindings call this after consuming
 * an error so a subsequent successful call is not misread as a failure.
 */
void mlx_clear_last_error(void);

/**
 * Store a classified error for the calling thread and invoke the legacy
 * handler. Used by generated bindings; also usable directly.
 * Macro variant passes __FILE__/__LINE__ like mlx_error().
 */
void _mlx_error_with_code(
    mlx_error_code code,
    const char* file,
    const int line,
    const char* fmt,
    ...);
#define mlx_error_with_code(code, ...) \
  _mlx_error_with_code(code, __FILE__, __LINE__, __VA_ARGS__)

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
