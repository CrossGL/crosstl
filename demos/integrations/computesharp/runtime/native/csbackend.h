/*
 * Native compute backend ABI used by the portable ComputeSharp runtime.
 *
 * The same functions are implemented over Metal (csbackend_metal.m) and over
 * desktop OpenGL 4.5 through EGL (csbackend_gl.c). All entry points return
 * zero on success. On failure they return a nonzero status and write a
 * NUL-terminated UTF-8 message into the caller-provided error buffer.
 *
 * Texture formats are DXGI_FORMAT numeric values, so the runtime and both
 * backends share one format vocabulary with the original Direct3D 12 baseline.
 */
#ifndef CSBACKEND_H
#define CSBACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CS_EXPORT __attribute__((visibility("default")))
#else
#define CS_EXPORT
#endif

#define CS_ABI_VERSION 1u

/* Texture dimensions. */
#define CS_TEXTURE_1D 1u
#define CS_TEXTURE_2D 2u
#define CS_TEXTURE_3D 3u

/* Texture usage flags. */
#define CS_USAGE_SHADER_READ 1u
#define CS_USAGE_SHADER_WRITE 2u

/* Binding kinds for cs_list_dispatch. */
#define CS_BIND_BUFFER 1u          /* read-only or read-write storage buffer */
#define CS_BIND_TEXTURE 2u         /* sampled or read-only texture */
#define CS_BIND_IMAGE 3u           /* read-write texture (storage image) */
#define CS_BIND_SAMPLER 4u         /* sampler state */
#define CS_BIND_CONSTANT_BYTES 5u  /* inline constant data, copied at record time */
#define CS_BIND_UNIFORM_BUFFER 6u  /* constant buffer object */

/* Sampler filter and address modes (mirroring D3D12 static sampler choices). */
#define CS_FILTER_POINT 0u
#define CS_FILTER_LINEAR 1u
#define CS_ADDRESS_CLAMP 0u
#define CS_ADDRESS_REPEAT 1u
#define CS_ADDRESS_MIRROR 2u

typedef struct cs_device_info {
    uint32_t abi_version;
    uint32_t is_hardware;
    uint32_t supports_fp64;
    uint32_t unified_memory;
    uint64_t dedicated_memory;
    uint64_t shared_memory;
    uint32_t compute_units;
    uint32_t wavefront_size;
    uint32_t max_threads_per_group;
    uint32_t max_group_size_x;
    uint32_t max_group_size_y;
    uint32_t max_group_size_z;
    uint32_t reserved;
    char name[256];
    char api[64];
    char driver[512];
} cs_device_info;

typedef struct cs_binding {
    uint32_t kind;
    uint32_t slot;
    uint32_t format;   /* DXGI format for images and textures, 0 otherwise */
    uint32_t access;   /* CS_USAGE_* for images */
    void *object;
    uint64_t offset;
    uint64_t size;
    const void *data;  /* CS_BIND_CONSTANT_BYTES only */
} cs_binding;

CS_EXPORT uint32_t cs_abi_version(void);

CS_EXPORT int cs_device_create(void **device, cs_device_info *info, char *error, int32_t error_capacity);
CS_EXPORT void cs_device_destroy(void *device);
CS_EXPORT int cs_format_supported(void *device, uint32_t dimension, uint32_t format, uint32_t usage);

CS_EXPORT int cs_buffer_create(void *device, uint64_t size, void **buffer, char *error, int32_t error_capacity);
CS_EXPORT void cs_buffer_destroy(void *device, void *buffer);
CS_EXPORT int cs_buffer_write(void *device, void *buffer, uint64_t offset, const void *data, uint64_t size, char *error, int32_t error_capacity);
CS_EXPORT int cs_buffer_read(void *device, void *buffer, uint64_t offset, void *data, uint64_t size, char *error, int32_t error_capacity);

CS_EXPORT int cs_texture_create(void *device, uint32_t dimension, uint32_t format, uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, void **texture, char *error, int32_t error_capacity);
CS_EXPORT void cs_texture_destroy(void *device, void *texture);
/* Region writes/reads use tightly described host memory with explicit row and slice pitches. */
CS_EXPORT int cs_texture_write(void *device, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, const void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity);
CS_EXPORT int cs_texture_read(void *device, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity);

CS_EXPORT int cs_sampler_create(void *device, uint32_t filter, uint32_t address_mode, void **sampler, char *error, int32_t error_capacity);
CS_EXPORT void cs_sampler_destroy(void *device, void *sampler);

/* Compiles target source text. The log receives compiler diagnostics (warnings on success). */
CS_EXPORT int cs_pipeline_create(void *device, const char *source, const char *entry_point, uint32_t threads_x, uint32_t threads_y, uint32_t threads_z, void **pipeline, char *log, int32_t log_capacity);
CS_EXPORT void cs_pipeline_destroy(void *device, void *pipeline);

/* Command lists record work that executes in order when submitted. */
CS_EXPORT int cs_list_create(void *device, void **list, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_dispatch(void *list, void *pipeline, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, const cs_binding *bindings, uint32_t binding_count, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_barrier(void *list, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_copy_buffer(void *list, void *source, uint64_t source_offset, void *destination, uint64_t destination_offset, uint64_t size, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_fill_buffer(void *list, void *buffer, uint64_t offset, uint64_t size, uint32_t value, char *error, int32_t error_capacity);
/* Copies host-packed texel data from a buffer into a texture region, in command order. */
CS_EXPORT int cs_list_copy_buffer_to_texture(void *list, void *buffer, uint64_t offset, uint64_t row_pitch, uint64_t slice_pitch, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_copy_texture(void *list, void *source, uint32_t source_x, uint32_t source_y, uint32_t source_z, void *destination, uint32_t destination_x, uint32_t destination_y, uint32_t destination_z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity);
CS_EXPORT int cs_list_submit(void *list, char *error, int32_t error_capacity);
CS_EXPORT void cs_list_destroy(void *list);

#ifdef __cplusplus
}
#endif

#endif
