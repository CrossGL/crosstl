#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
  const char* name;
  const char* dtype;
  void* data;
  uint64_t count;
  uint32_t output;
} CrosstlMlxBuffer;

enum { CROSTL_MLX_DISPATCH_VERSION = 2 };

typedef struct {
  uint32_t workgroup_count[3];
  uint32_t workgroup_size[3];
} CrosstlMlxLaunch;

typedef int (*CrosstlMlxDispatch)(
    const char* entry,
    const CrosstlMlxBuffer* buffers,
    uint32_t buffer_count,
    uint64_t threads,
    const CrosstlMlxLaunch* launch,
    char* error,
    size_t error_capacity);
