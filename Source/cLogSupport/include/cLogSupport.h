#if defined(__APPLE__)
#import <os/log.h>
#import <stdbool.h>

static inline bool is_os_log_enabled(os_log_t log, os_log_type_t type) {
    return os_log_type_enabled(log, type);
}
#endif

