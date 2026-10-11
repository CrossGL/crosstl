#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
  const char* name;
  const char* dtype;
  void* data;
  uint64_t count;
  // 0: input, 1: output, 2: initialized input/output.
  uint32_t output;
} CrosstlMlxBuffer;

enum { CROSTL_MLX_DISPATCH_VERSION = 3 };
enum { CROSTL_MLX_BUFFER_INOUT = 2 };

typedef struct {
  uint32_t workgroup_count[3];
  uint32_t workgroup_size[3];
  // All zero for full workgroups; otherwise the exact source thread extent.
  uint32_t thread_grid_size[3];
} CrosstlMlxLaunch;

typedef int (*CrosstlMlxDispatch)(
    const char* entry,
    const CrosstlMlxBuffer* buffers,
    uint32_t buffer_count,
    uint64_t threads,
    const CrosstlMlxLaunch* launch,
    char* error,
    size_t error_capacity);

// Return one only when the named entry is available in the registered runtime.
typedef int (*CrosstlMlxEntryAvailable)(const char* entry);
