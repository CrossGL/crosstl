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

typedef int (*CrosstlMlxDispatch)(
    const char* entry,
    const CrosstlMlxBuffer* buffers,
    uint32_t buffer_count,
    uint64_t threads,
    char* error,
    size_t error_capacity);
