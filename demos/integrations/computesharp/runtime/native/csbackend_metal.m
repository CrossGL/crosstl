/*
 * Metal implementation of csbackend.h for the portable ComputeSharp runtime.
 *
 * Build: clang -fobjc-arc -shared -framework Metal -framework Foundation \
 *        -framework IOKit -o libcsbackend_metal.dylib csbackend_metal.m
 */
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csbackend.h"

static void cs_set_error(char *error, int32_t capacity, NSString *message) {
    if (error == NULL || capacity <= 0) {
        return;
    }
    const char *text = message != nil ? [message UTF8String] : "unknown error";
    snprintf(error, (size_t)capacity, "%s", text != NULL ? text : "unknown error");
}

static void cs_copy_text(char *destination, size_t capacity, NSString *text) {
    const char *utf8 = text != nil ? [text UTF8String] : "";
    snprintf(destination, capacity, "%s", utf8 != NULL ? utf8 : "");
}

/*
 * Shader compile options and a description of the floating-point model they
 * select. Direct3D shaders keep IEEE NaN/Inf semantics and evaluation order,
 * which Metal's safe math mode keeps. The translation itself evaluates sin,
 * cos, tan and contraction-free arithmetic the way WARP does, so the
 * floating-point function setting only selects the native implementation of
 * the remaining transcendentals (sqrt, exp, log, pow, atan...). Measured
 * against WARP with the intrinsic probe, fast functions are closer for exp
 * and pow and precise functions for sqrt and log.
 * Before macOS 15 the only way to keep IEEE semantics is to disable fast
 * math entirely, which also selects precise functions.
 */
static MTLCompileOptions *cs_compile_options(NSString **description) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        if (description != NULL) {
            *description = @"math mode safe, fast floating-point functions";
        }
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        if (description != NULL) {
            *description = @"fast math disabled, precise floating-point functions";
        }
    }
    return options;
}

@interface CSMetalDevice : NSObject {
@public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    BOOL unified;
}
@end
@implementation CSMetalDevice
@end

@interface CSMetalBuffer : NSObject {
@public
    id<MTLBuffer> buffer;
    uint64_t size;
}
@end
@implementation CSMetalBuffer
@end

@interface CSMetalTexture : NSObject {
@public
    id<MTLTexture> texture;
    uint32_t dimension;
    uint32_t format;
    uint32_t texel_size;      /* bytes per texel in host memory */
    uint32_t storage_size;    /* bytes per texel in the Metal texture */
}
@end
@implementation CSMetalTexture
@end

@interface CSMetalSampler : NSObject {
@public
    id<MTLSamplerState> sampler;
}
@end
@implementation CSMetalSampler
@end

@interface CSMetalPipeline : NSObject {
@public
    id<MTLComputePipelineState> state;
    MTLSize threads;
}
@end
@implementation CSMetalPipeline
@end

@interface CSMetalList : NSObject {
@public
    CSMetalDevice *device;
    id<MTLCommandBuffer> commands;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    BOOL submitted;
}
@end
@implementation CSMetalList
@end

typedef struct cs_format_desc {
    uint32_t dxgi;
    MTLPixelFormat metal;
    uint32_t texel_size;
    uint32_t storage_size;
    uint32_t rgb32;  /* three-channel 32-bit formats are stored with a fourth channel */
} cs_format_desc;

static const cs_format_desc CS_FORMATS[] = {
    {2, MTLPixelFormatRGBA32Float, 16, 16, 0},
    {3, MTLPixelFormatRGBA32Uint, 16, 16, 0},
    {4, MTLPixelFormatRGBA32Sint, 16, 16, 0},
    {6, MTLPixelFormatRGBA32Float, 12, 16, 1},
    {7, MTLPixelFormatRGBA32Uint, 12, 16, 1},
    {8, MTLPixelFormatRGBA32Sint, 12, 16, 1},
    {11, MTLPixelFormatRGBA16Unorm, 8, 8, 0},
    {16, MTLPixelFormatRG32Float, 8, 8, 0},
    {17, MTLPixelFormatRG32Uint, 8, 8, 0},
    {18, MTLPixelFormatRG32Sint, 8, 8, 0},
    {28, MTLPixelFormatRGBA8Unorm, 4, 4, 0},
    {35, MTLPixelFormatRG16Unorm, 4, 4, 0},
    {41, MTLPixelFormatR32Float, 4, 4, 0},
    {42, MTLPixelFormatR32Uint, 4, 4, 0},
    {43, MTLPixelFormatR32Sint, 4, 4, 0},
    {49, MTLPixelFormatRG8Unorm, 2, 2, 0},
    {56, MTLPixelFormatR16Unorm, 2, 2, 0},
    {61, MTLPixelFormatR8Unorm, 1, 1, 0},
    {87, MTLPixelFormatBGRA8Unorm, 4, 4, 0},
};

static const cs_format_desc *cs_find_format(uint32_t dxgi) {
    for (size_t i = 0; i < sizeof(CS_FORMATS) / sizeof(CS_FORMATS[0]); i++) {
        if (CS_FORMATS[i].dxgi == dxgi) {
            return &CS_FORMATS[i];
        }
    }
    return NULL;
}

static uint32_t cs_gpu_core_count(void) {
    uint32_t cores = 0;
    io_iterator_t iterator = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) != KERN_SUCCESS) {
        return 0;
    }
    io_object_t service;
    while ((service = IOIteratorNext(iterator)) != 0) {
        CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value != NULL) {
            if (CFGetTypeID(value) == CFNumberGetTypeID()) {
                int32_t count = 0;
                CFNumberGetValue((CFNumberRef)value, kCFNumberSInt32Type, &count);
                if (count > 0) {
                    cores = (uint32_t)count;
                }
            }
            CFRelease(value);
        }
        IOObjectRelease(service);
        if (cores != 0) {
            break;
        }
    }
    IOObjectRelease(iterator);
    return cores;
}

uint32_t cs_abi_version(void) { return CS_ABI_VERSION; }

int cs_device_create(void **out_device, cs_device_info *info, char *error, int32_t error_capacity) {
    @autoreleasepool {
        if (out_device == NULL || info == NULL) {
            cs_set_error(error, error_capacity, @"invalid arguments");
            return 1;
        }
        *out_device = NULL;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            cs_set_error(error, error_capacity, @"MTLCreateSystemDefaultDevice returned no device");
            return 2;
        }
        CSMetalDevice *wrapper = [CSMetalDevice new];
        wrapper->device = device;
        wrapper->queue = [device newCommandQueue];
        wrapper->unified = device.hasUnifiedMemory;
        if (!wrapper->unified) {
            cs_set_error(error, error_capacity, @"the portable runtime requires a unified-memory Metal device");
            return 3;
        }
        if (wrapper->queue == nil) {
            cs_set_error(error, error_capacity, @"failed to create a Metal command queue");
            return 3;
        }

        memset(info, 0, sizeof(*info));
        info->abi_version = CS_ABI_VERSION;
        info->is_hardware = 1;
        info->supports_fp64 = 0;
        info->unified_memory = 1;
        uint64_t working_set = device.recommendedMaxWorkingSetSize;
        info->dedicated_memory = 0;
        info->shared_memory = working_set;
        MTLSize max_threads = device.maxThreadsPerThreadgroup;
        info->max_group_size_x = (uint32_t)max_threads.width;
        info->max_group_size_y = (uint32_t)max_threads.height;
        info->max_group_size_z = (uint32_t)max_threads.depth;

        NSError *compile_error = nil;
        NSString *probe = @"#include <metal_stdlib>\nusing namespace metal;\n"
                          @"kernel void cs_probe(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }\n";
        id<MTLLibrary> library = [device newLibraryWithSource:probe options:nil error:&compile_error];
        id<MTLFunction> function = [library newFunctionWithName:@"cs_probe"];
        id<MTLComputePipelineState> state = function != nil ? [device newComputePipelineStateWithFunction:function error:&compile_error] : nil;
        if (state == nil) {
            cs_set_error(error, error_capacity, [NSString stringWithFormat:@"Metal probe pipeline failed: %@", compile_error]);
            return 4;
        }
        info->wavefront_size = (uint32_t)state.threadExecutionWidth;
        info->max_threads_per_group = (uint32_t)state.maxTotalThreadsPerThreadgroup;
        uint32_t cores = cs_gpu_core_count();
        info->compute_units = cores != 0 ? cores * 128u : 0u;

        cs_copy_text(info->name, sizeof(info->name), device.name);
        cs_copy_text(info->api, sizeof(info->api), @"Metal");
        NSMutableArray<NSString *> *families = [NSMutableArray array];
        struct { MTLGPUFamily family; const char *name; } known[] = {
            {MTLGPUFamilyApple7, "Apple7"}, {MTLGPUFamilyApple8, "Apple8"}, {MTLGPUFamilyApple9, "Apple9"},
            {MTLGPUFamilyMetal3, "Metal3"},
        };
        for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
            if ([device supportsFamily:known[i].family]) {
                [families addObject:[NSString stringWithUTF8String:known[i].name]];
            }
        }
        NSString *math = nil;
        (void)cs_compile_options(&math);
        NSString *driver = [NSString stringWithFormat:@"macOS %@; families %@; read-write texture tier %lu; gpu cores %u; shaders %@",
                            [[NSProcessInfo processInfo] operatingSystemVersionString],
                            [families componentsJoinedByString:@","],
                            (unsigned long)device.readWriteTextureSupport, cores, math];
        cs_copy_text(info->driver, sizeof(info->driver), driver);
        *out_device = (__bridge_retained void *)wrapper;
        return 0;
    }
}

void cs_device_destroy(void *device) {
    if (device != NULL) {
        CSMetalDevice *wrapper = (__bridge_transfer CSMetalDevice *)device;
        (void)wrapper;
    }
}

int cs_format_supported(void *device, uint32_t dimension, uint32_t format, uint32_t usage) {
    (void)device;
    const cs_format_desc *desc = cs_find_format(format);
    if (desc == NULL || dimension < CS_TEXTURE_1D || dimension > CS_TEXTURE_3D) {
        return 0;
    }
    /* Direct3D 12 does not support typed unordered access for 96-bit formats. */
    if (desc->rgb32 && (usage & CS_USAGE_SHADER_WRITE) != 0) {
        return 0;
    }
    return 1;
}

int cs_buffer_create(void *device, uint64_t size, void **out_buffer, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalDevice *wrapper = (__bridge CSMetalDevice *)device;
        id<MTLBuffer> buffer = [wrapper->device newBufferWithLength:(NSUInteger)(size == 0 ? 16 : size) options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            cs_set_error(error, error_capacity, [NSString stringWithFormat:@"failed to allocate a %llu byte Metal buffer", (unsigned long long)size]);
            return 1;
        }
        memset(buffer.contents, 0, buffer.length);
        CSMetalBuffer *result = [CSMetalBuffer new];
        result->buffer = buffer;
        result->size = size;
        *out_buffer = (__bridge_retained void *)result;
        return 0;
    }
}

void cs_buffer_destroy(void *device, void *buffer) {
    (void)device;
    if (buffer != NULL) {
        CSMetalBuffer *wrapper = (__bridge_transfer CSMetalBuffer *)buffer;
        (void)wrapper;
    }
}

int cs_buffer_write(void *device, void *buffer, uint64_t offset, const void *data, uint64_t size, char *error, int32_t error_capacity) {
    @autoreleasepool {
        (void)device;
        CSMetalBuffer *wrapper = (__bridge CSMetalBuffer *)buffer;
        if (offset + size > wrapper->buffer.length) {
            cs_set_error(error, error_capacity, @"buffer write out of range");
            return 1;
        }
        memcpy((uint8_t *)wrapper->buffer.contents + offset, data, (size_t)size);
        return 0;
    }
}

int cs_buffer_read(void *device, void *buffer, uint64_t offset, void *data, uint64_t size, char *error, int32_t error_capacity) {
    @autoreleasepool {
        (void)device;
        CSMetalBuffer *wrapper = (__bridge CSMetalBuffer *)buffer;
        if (offset + size > wrapper->buffer.length) {
            cs_set_error(error, error_capacity, @"buffer read out of range");
            return 1;
        }
        memcpy(data, (const uint8_t *)wrapper->buffer.contents + offset, (size_t)size);
        return 0;
    }
}

int cs_texture_create(void *device, uint32_t dimension, uint32_t format, uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, void **out_texture, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalDevice *dev = (__bridge CSMetalDevice *)device;
        const cs_format_desc *desc = cs_find_format(format);
        if (desc == NULL) {
            cs_set_error(error, error_capacity, [NSString stringWithFormat:@"unsupported texture format %u", format]);
            return 1;
        }
        MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
        switch (dimension) {
        case CS_TEXTURE_1D:
            descriptor.textureType = MTLTextureType1D;
            height = 1;
            depth = 1;
            break;
        case CS_TEXTURE_2D:
            descriptor.textureType = MTLTextureType2D;
            depth = 1;
            break;
        case CS_TEXTURE_3D:
            descriptor.textureType = MTLTextureType3D;
            break;
        default:
            cs_set_error(error, error_capacity, @"invalid texture dimension");
            return 2;
        }
        descriptor.pixelFormat = desc->metal;
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = depth;
        descriptor.mipmapLevelCount = 1;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead | ((usage & CS_USAGE_SHADER_WRITE) != 0 ? MTLTextureUsageShaderWrite : 0);
        id<MTLTexture> texture = [dev->device newTextureWithDescriptor:descriptor];
        if (texture == nil) {
            cs_set_error(error, error_capacity, [NSString stringWithFormat:@"failed to create a %ux%ux%u Metal texture with format %u", width, height, depth, format]);
            return 3;
        }
        CSMetalTexture *wrapper = [CSMetalTexture new];
        wrapper->texture = texture;
        wrapper->dimension = dimension;
        wrapper->format = format;
        wrapper->texel_size = desc->texel_size;
        wrapper->storage_size = desc->storage_size;

        /* Direct3D 12 committed resources start zeroed; match that for reads before writes. */
        size_t row = (size_t)width * desc->storage_size;
        size_t slice = row * height;
        void *zeros = calloc(1, slice * depth);
        if (zeros == NULL) {
            cs_set_error(error, error_capacity, @"out of host memory");
            return 4;
        }
        MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, depth);
        [texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:zeros
                   bytesPerRow:(dimension == CS_TEXTURE_1D ? 0 : row)
                 bytesPerImage:(dimension == CS_TEXTURE_3D ? slice : 0)];
        free(zeros);
        *out_texture = (__bridge_retained void *)wrapper;
        return 0;
    }
}

void cs_texture_destroy(void *device, void *texture) {
    (void)device;
    if (texture != NULL) {
        CSMetalTexture *wrapper = (__bridge_transfer CSMetalTexture *)texture;
        (void)wrapper;
    }
}

int cs_texture_write(void *device, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, const void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity) {
    @autoreleasepool {
        (void)device;
        CSMetalTexture *wrapper = (__bridge CSMetalTexture *)texture;
        const uint8_t *source = (const uint8_t *)data;
        size_t storage_row = (size_t)width * wrapper->storage_size;
        size_t storage_slice = storage_row * height;
        uint8_t *staging = NULL;
        if (wrapper->storage_size != wrapper->texel_size || row_pitch != storage_row || slice_pitch != storage_slice) {
            staging = (uint8_t *)calloc(1, storage_slice * depth);
            if (staging == NULL) {
                cs_set_error(error, error_capacity, @"out of host memory");
                return 1;
            }
            for (uint32_t k = 0; k < depth; k++) {
                for (uint32_t j = 0; j < height; j++) {
                    const uint8_t *src_row = source + k * slice_pitch + j * row_pitch;
                    uint8_t *dst_row = staging + k * storage_slice + j * storage_row;
                    if (wrapper->storage_size == wrapper->texel_size) {
                        memcpy(dst_row, src_row, storage_row);
                    } else {
                        for (uint32_t i = 0; i < width; i++) {
                            memcpy(dst_row + (size_t)i * wrapper->storage_size, src_row + (size_t)i * wrapper->texel_size, wrapper->texel_size);
                        }
                    }
                }
            }
            source = staging;
        }
        MTLRegion region = MTLRegionMake3D(x, y, z, width, height, depth);
        [wrapper->texture replaceRegion:region mipmapLevel:0 slice:0 withBytes:source
                            bytesPerRow:(wrapper->dimension == CS_TEXTURE_1D ? 0 : storage_row)
                          bytesPerImage:(wrapper->dimension == CS_TEXTURE_3D ? storage_slice : 0)];
        free(staging);
        return 0;
    }
}

int cs_texture_read(void *device, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity) {
    @autoreleasepool {
        (void)device;
        CSMetalTexture *wrapper = (__bridge CSMetalTexture *)texture;
        size_t storage_row = (size_t)width * wrapper->storage_size;
        size_t storage_slice = storage_row * height;
        uint8_t *staging = (uint8_t *)malloc(storage_slice * depth);
        if (staging == NULL) {
            cs_set_error(error, error_capacity, @"out of host memory");
            return 2;
        }
        MTLRegion region = MTLRegionMake3D(x, y, z, width, height, depth);
        [wrapper->texture getBytes:staging
                       bytesPerRow:(wrapper->dimension == CS_TEXTURE_1D ? 0 : storage_row)
                     bytesPerImage:(wrapper->dimension == CS_TEXTURE_3D ? storage_slice : 0)
                        fromRegion:region mipmapLevel:0 slice:0];
        uint8_t *destination = (uint8_t *)data;
        for (uint32_t k = 0; k < depth; k++) {
            for (uint32_t j = 0; j < height; j++) {
                const uint8_t *src_row = staging + k * storage_slice + j * storage_row;
                uint8_t *dst_row = destination + k * slice_pitch + j * row_pitch;
                if (wrapper->storage_size == wrapper->texel_size) {
                    memcpy(dst_row, src_row, storage_row);
                } else {
                    for (uint32_t i = 0; i < width; i++) {
                        memcpy(dst_row + (size_t)i * wrapper->texel_size, src_row + (size_t)i * wrapper->storage_size, wrapper->texel_size);
                    }
                }
            }
        }
        free(staging);
        return 0;
    }
}

int cs_sampler_create(void *device, uint32_t filter, uint32_t address_mode, void **out_sampler, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalDevice *dev = (__bridge CSMetalDevice *)device;
        MTLSamplerDescriptor *descriptor = [MTLSamplerDescriptor new];
        MTLSamplerMinMagFilter min_mag = filter == CS_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        descriptor.minFilter = min_mag;
        descriptor.magFilter = min_mag;
        descriptor.mipFilter = filter == CS_FILTER_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        MTLSamplerAddressMode mode = MTLSamplerAddressModeClampToEdge;
        if (address_mode == CS_ADDRESS_REPEAT) {
            mode = MTLSamplerAddressModeRepeat;
        } else if (address_mode == CS_ADDRESS_MIRROR) {
            mode = MTLSamplerAddressModeMirrorRepeat;
        }
        descriptor.sAddressMode = mode;
        descriptor.tAddressMode = mode;
        descriptor.rAddressMode = mode;
        descriptor.normalizedCoordinates = YES;
        id<MTLSamplerState> sampler = [dev->device newSamplerStateWithDescriptor:descriptor];
        if (sampler == nil) {
            cs_set_error(error, error_capacity, @"failed to create a Metal sampler state");
            return 1;
        }
        CSMetalSampler *wrapper = [CSMetalSampler new];
        wrapper->sampler = sampler;
        *out_sampler = (__bridge_retained void *)wrapper;
        return 0;
    }
}

void cs_sampler_destroy(void *device, void *sampler) {
    (void)device;
    if (sampler != NULL) {
        CSMetalSampler *wrapper = (__bridge_transfer CSMetalSampler *)sampler;
        (void)wrapper;
    }
}

int cs_pipeline_create(void *device, const char *source, const char *entry_point, uint32_t threads_x, uint32_t threads_y, uint32_t threads_z, void **out_pipeline, char *log, int32_t log_capacity) {
    @autoreleasepool {
        CSMetalDevice *dev = (__bridge CSMetalDevice *)device;
        if (log != NULL && log_capacity > 0) {
            log[0] = '\0';
        }
        MTLCompileOptions *options = cs_compile_options(NULL);
        NSError *compile_error = nil;
        NSString *text = [NSString stringWithUTF8String:source];
        id<MTLLibrary> library = [dev->device newLibraryWithSource:text options:options error:&compile_error];
        if (library == nil) {
            cs_set_error(log, log_capacity, [NSString stringWithFormat:@"Metal compilation failed: %@", compile_error.localizedDescription]);
            return 1;
        }
        if (compile_error != nil) {
            cs_set_error(log, log_capacity, compile_error.localizedDescription);
        }
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:entry_point]];
        if (function == nil) {
            cs_set_error(log, log_capacity, [NSString stringWithFormat:@"Metal library has no function named %s", entry_point]);
            return 2;
        }
        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> state = [dev->device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (state == nil) {
            cs_set_error(log, log_capacity, [NSString stringWithFormat:@"Metal pipeline creation failed: %@", pipeline_error.localizedDescription]);
            return 3;
        }
        uint64_t total = (uint64_t)threads_x * threads_y * threads_z;
        if (total == 0 || total > state.maxTotalThreadsPerThreadgroup) {
            cs_set_error(log, log_capacity, [NSString stringWithFormat:@"thread group %ux%ux%u exceeds the pipeline limit of %lu threads",
                                             threads_x, threads_y, threads_z, (unsigned long)state.maxTotalThreadsPerThreadgroup]);
            return 4;
        }
        CSMetalPipeline *wrapper = [CSMetalPipeline new];
        wrapper->state = state;
        wrapper->threads = MTLSizeMake(threads_x, threads_y, threads_z);
        *out_pipeline = (__bridge_retained void *)wrapper;
        return 0;
    }
}

void cs_pipeline_destroy(void *device, void *pipeline) {
    (void)device;
    if (pipeline != NULL) {
        CSMetalPipeline *wrapper = (__bridge_transfer CSMetalPipeline *)pipeline;
        (void)wrapper;
    }
}

int cs_list_create(void *device, void **out_list, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalDevice *dev = (__bridge CSMetalDevice *)device;
        CSMetalList *list = [CSMetalList new];
        list->device = dev;
        list->commands = [dev->queue commandBuffer];
        if (list->commands == nil) {
            cs_set_error(error, error_capacity, @"failed to create a Metal command buffer");
            return 1;
        }
        *out_list = (__bridge_retained void *)list;
        return 0;
    }
}

static void cs_end_compute(CSMetalList *list) {
    if (list->compute != nil) {
        [list->compute endEncoding];
        list->compute = nil;
    }
}

static void cs_end_blit(CSMetalList *list) {
    if (list->blit != nil) {
        [list->blit endEncoding];
        list->blit = nil;
    }
}

static id<MTLComputeCommandEncoder> cs_compute_encoder(CSMetalList *list) {
    cs_end_blit(list);
    if (list->compute == nil) {
        list->compute = [list->commands computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
    return list->compute;
}

static id<MTLBlitCommandEncoder> cs_blit_encoder(CSMetalList *list) {
    cs_end_compute(list);
    if (list->blit == nil) {
        list->blit = [list->commands blitCommandEncoder];
    }
    return list->blit;
}

int cs_list_dispatch(void *list_handle, void *pipeline_handle, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, const cs_binding *bindings, uint32_t binding_count, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        CSMetalPipeline *pipeline = (__bridge CSMetalPipeline *)pipeline_handle;
        if (list->submitted) {
            cs_set_error(error, error_capacity, @"command list already submitted");
            return 1;
        }
        id<MTLComputeCommandEncoder> encoder = cs_compute_encoder(list);
        [encoder setComputePipelineState:pipeline->state];
        for (uint32_t i = 0; i < binding_count; i++) {
            const cs_binding *binding = &bindings[i];
            switch (binding->kind) {
            case CS_BIND_BUFFER:
            case CS_BIND_UNIFORM_BUFFER: {
                CSMetalBuffer *buffer = (__bridge CSMetalBuffer *)binding->object;
                [encoder setBuffer:buffer->buffer offset:(NSUInteger)binding->offset atIndex:binding->slot];
                break;
            }
            case CS_BIND_CONSTANT_BYTES:
                [encoder setBytes:binding->data length:(NSUInteger)binding->size atIndex:binding->slot];
                break;
            case CS_BIND_TEXTURE:
            case CS_BIND_IMAGE: {
                CSMetalTexture *texture = (__bridge CSMetalTexture *)binding->object;
                [encoder setTexture:texture->texture atIndex:binding->slot];
                break;
            }
            case CS_BIND_SAMPLER: {
                CSMetalSampler *sampler = (__bridge CSMetalSampler *)binding->object;
                [encoder setSamplerState:sampler->sampler atIndex:binding->slot];
                break;
            }
            default:
                cs_set_error(error, error_capacity, [NSString stringWithFormat:@"invalid binding kind %u", binding->kind]);
                return 2;
            }
        }
        [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z) threadsPerThreadgroup:pipeline->threads];
        return 0;
    }
}

int cs_list_barrier(void *list_handle, char *error, int32_t error_capacity) {
    (void)error;
    (void)error_capacity;
    CSMetalList *list = (__bridge CSMetalList *)list_handle;
    if (list->compute != nil) {
        [list->compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
    }
    return 0;
}

int cs_list_copy_buffer(void *list_handle, void *source, uint64_t source_offset, void *destination, uint64_t destination_offset, uint64_t size, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        CSMetalBuffer *src = (__bridge CSMetalBuffer *)source;
        CSMetalBuffer *dst = (__bridge CSMetalBuffer *)destination;
        if (source_offset + size > src->buffer.length || destination_offset + size > dst->buffer.length) {
            cs_set_error(error, error_capacity, @"buffer copy out of range");
            return 1;
        }
        if (size == 0) {
            return 0;
        }
        [cs_blit_encoder(list) copyFromBuffer:src->buffer sourceOffset:(NSUInteger)source_offset toBuffer:dst->buffer destinationOffset:(NSUInteger)destination_offset size:(NSUInteger)size];
        return 0;
    }
}

int cs_list_fill_buffer(void *list_handle, void *buffer, uint64_t offset, uint64_t size, uint32_t value, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        CSMetalBuffer *target = (__bridge CSMetalBuffer *)buffer;
        if (offset + size > target->buffer.length) {
            cs_set_error(error, error_capacity, @"buffer fill out of range");
            return 1;
        }
        if (value > 0xFFu) {
            cs_set_error(error, error_capacity, @"Metal buffer fills take a byte value");
            return 2;
        }
        if (size == 0) {
            return 0;
        }
        [cs_blit_encoder(list) fillBuffer:target->buffer range:NSMakeRange((NSUInteger)offset, (NSUInteger)size) value:(uint8_t)value];
        return 0;
    }
}

int cs_list_copy_buffer_to_texture(void *list_handle, void *buffer, uint64_t offset, uint64_t row_pitch, uint64_t slice_pitch, void *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        CSMetalBuffer *source = (__bridge CSMetalBuffer *)buffer;
        CSMetalTexture *target = (__bridge CSMetalTexture *)texture;
        if (target->storage_size != target->texel_size) {
            cs_set_error(error, error_capacity, @"recorded copies into padded three-channel textures are not supported");
            return 1;
        }
        [cs_blit_encoder(list) copyFromBuffer:source->buffer
                                 sourceOffset:(NSUInteger)offset
                            sourceBytesPerRow:(NSUInteger)(target->dimension == CS_TEXTURE_1D ? 0 : row_pitch)
                          sourceBytesPerImage:(NSUInteger)(target->dimension == CS_TEXTURE_3D ? slice_pitch : 0)
                                   sourceSize:MTLSizeMake(width, height, depth)
                                    toTexture:target->texture
                             destinationSlice:0
                             destinationLevel:0
                            destinationOrigin:MTLOriginMake(x, y, z)];
        return 0;
    }
}

int cs_list_copy_texture(void *list_handle, void *source, uint32_t source_x, uint32_t source_y, uint32_t source_z, void *destination, uint32_t destination_x, uint32_t destination_y, uint32_t destination_z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity) {
    @autoreleasepool {
        (void)error;
        (void)error_capacity;
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        CSMetalTexture *src = (__bridge CSMetalTexture *)source;
        CSMetalTexture *dst = (__bridge CSMetalTexture *)destination;
        [cs_blit_encoder(list) copyFromTexture:src->texture sourceSlice:0 sourceLevel:0
                                  sourceOrigin:MTLOriginMake(source_x, source_y, source_z)
                                    sourceSize:MTLSizeMake(width, height, depth)
                                     toTexture:dst->texture destinationSlice:0 destinationLevel:0
                             destinationOrigin:MTLOriginMake(destination_x, destination_y, destination_z)];
        return 0;
    }
}

int cs_list_submit(void *list_handle, char *error, int32_t error_capacity) {
    @autoreleasepool {
        CSMetalList *list = (__bridge CSMetalList *)list_handle;
        if (list->submitted) {
            cs_set_error(error, error_capacity, @"command list already submitted");
            return 1;
        }
        cs_end_compute(list);
        cs_end_blit(list);
        list->submitted = YES;
        [list->commands commit];
        [list->commands waitUntilCompleted];
        if (list->commands.status != MTLCommandBufferStatusCompleted) {
            cs_set_error(error, error_capacity, [NSString stringWithFormat:@"Metal command buffer failed (status %ld): %@",
                                                 (long)list->commands.status, list->commands.error.localizedDescription]);
            return 2;
        }
        return 0;
    }
}

void cs_list_destroy(void *list_handle) {
    if (list_handle != NULL) {
        CSMetalList *list = (__bridge_transfer CSMetalList *)list_handle;
        if (!list->submitted) {
            cs_end_compute(list);
            cs_end_blit(list);
        }
        (void)list;
    }
}
