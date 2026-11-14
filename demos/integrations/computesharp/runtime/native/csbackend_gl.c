/*
 * Desktop OpenGL 4.5 core implementation of csbackend.h for the portable
 * ComputeSharp runtime.
 *
 * The context is created through EGL without a window system, preferring
 * EGL_MESA_platform_surfaceless, then EGL_EXT_platform_device, then the
 * default display. It therefore runs headless on Mesa llvmpipe as well as on
 * hardware drivers. A GL context is current on one thread only, so the
 * runtime issues every backend call on a dedicated thread.
 *
 * Command lists record their operations on the host and replay them in order
 * on submit. As with Direct3D 12 command lists and Metal command buffers,
 * recorded work therefore executes after immediate copies made while the list
 * was open. Every dispatch is followed by a full memory barrier, which orders
 * dependent dispatches exactly like the serial Metal compute encoder.
 *
 * Build: cc -O2 -fPIC -shared -o libcsbackend_gl.so csbackend_gl.c -lEGL
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "csbackend.h"

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#ifndef EGL_PLATFORM_DEVICE_EXT
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif
#ifndef EGL_NO_CONFIG_KHR
#define EGL_NO_CONFIG_KHR ((EGLConfig)0)
#endif
#ifndef GL_SUBGROUP_SIZE_KHR
#define GL_SUBGROUP_SIZE_KHR 0x9532
#endif

static void cs_set_error(char *error, int32_t capacity, const char *format, ...) {
    if (error == NULL || capacity <= 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    vsnprintf(error, (size_t)capacity, format, args);
    va_end(args);
}

/* ------------------------------------------------------------------------- */
/* OpenGL entry points, loaded through eglGetProcAddress.                     */
/* ------------------------------------------------------------------------- */

#define CS_GL_FUNCTIONS(X)                                                   \
    X(PFNGLGETSTRINGPROC, GetString)                                         \
    X(PFNGLGETSTRINGIPROC, GetStringi)                                       \
    X(PFNGLGETINTEGERVPROC, GetIntegerv)                                     \
    X(PFNGLGETINTEGERI_VPROC, GetIntegeri_v)                                 \
    X(PFNGLGETERRORPROC, GetError)                                           \
    X(PFNGLFINISHPROC, Finish)                                               \
    X(PFNGLPIXELSTOREIPROC, PixelStorei)                                     \
    X(PFNGLCREATEBUFFERSPROC, CreateBuffers)                                 \
    X(PFNGLNAMEDBUFFERSTORAGEPROC, NamedBufferStorage)                       \
    X(PFNGLNAMEDBUFFERSUBDATAPROC, NamedBufferSubData)                       \
    X(PFNGLGETNAMEDBUFFERSUBDATAPROC, GetNamedBufferSubData)                 \
    X(PFNGLCLEARNAMEDBUFFERSUBDATAPROC, ClearNamedBufferSubData)             \
    X(PFNGLCOPYNAMEDBUFFERSUBDATAPROC, CopyNamedBufferSubData)               \
    X(PFNGLDELETEBUFFERSPROC, DeleteBuffers)                                 \
    X(PFNGLBINDBUFFERPROC, BindBuffer)                                       \
    X(PFNGLBINDBUFFERRANGEPROC, BindBufferRange)                             \
    X(PFNGLCREATETEXTURESPROC, CreateTextures)                               \
    X(PFNGLTEXTURESTORAGE1DPROC, TextureStorage1D)                           \
    X(PFNGLTEXTURESTORAGE2DPROC, TextureStorage2D)                           \
    X(PFNGLTEXTURESTORAGE3DPROC, TextureStorage3D)                           \
    X(PFNGLTEXTURESUBIMAGE1DPROC, TextureSubImage1D)                         \
    X(PFNGLTEXTURESUBIMAGE2DPROC, TextureSubImage2D)                         \
    X(PFNGLTEXTURESUBIMAGE3DPROC, TextureSubImage3D)                         \
    X(PFNGLGETTEXTURESUBIMAGEPROC, GetTextureSubImage)                       \
    X(PFNGLTEXTUREPARAMETERIPROC, TextureParameteri)                         \
    X(PFNGLCLEARTEXIMAGEPROC, ClearTexImage)                                 \
    X(PFNGLCOPYIMAGESUBDATAPROC, CopyImageSubData)                           \
    X(PFNGLDELETETEXTURESPROC, DeleteTextures)                               \
    X(PFNGLBINDTEXTUREUNITPROC, BindTextureUnit)                             \
    X(PFNGLBINDIMAGETEXTUREPROC, BindImageTexture)                           \
    X(PFNGLCREATESAMPLERSPROC, CreateSamplers)                               \
    X(PFNGLSAMPLERPARAMETERIPROC, SamplerParameteri)                         \
    X(PFNGLDELETESAMPLERSPROC, DeleteSamplers)                               \
    X(PFNGLBINDSAMPLERPROC, BindSampler)                                     \
    X(PFNGLCREATESHADERPROC, CreateShader)                                   \
    X(PFNGLSHADERSOURCEPROC, ShaderSource)                                   \
    X(PFNGLCOMPILESHADERPROC, CompileShader)                                 \
    X(PFNGLGETSHADERIVPROC, GetShaderiv)                                     \
    X(PFNGLGETSHADERINFOLOGPROC, GetShaderInfoLog)                           \
    X(PFNGLDELETESHADERPROC, DeleteShader)                                   \
    X(PFNGLCREATEPROGRAMPROC, CreateProgram)                                 \
    X(PFNGLATTACHSHADERPROC, AttachShader)                                   \
    X(PFNGLLINKPROGRAMPROC, LinkProgram)                                     \
    X(PFNGLGETPROGRAMIVPROC, GetProgramiv)                                   \
    X(PFNGLGETPROGRAMINFOLOGPROC, GetProgramInfoLog)                         \
    X(PFNGLDELETEPROGRAMPROC, DeleteProgram)                                 \
    X(PFNGLUSEPROGRAMPROC, UseProgram)                                       \
    X(PFNGLDISPATCHCOMPUTEPROC, DispatchCompute)                             \
    X(PFNGLMEMORYBARRIERPROC, MemoryBarrier)

typedef struct cs_gl_api {
#define CS_DECLARE(type, name) type name;
    CS_GL_FUNCTIONS(CS_DECLARE)
#undef CS_DECLARE
} cs_gl_api;

typedef struct cs_gl_device {
    EGLDisplay display;
    EGLContext context;
    cs_gl_api gl;
    GLint uniform_offset_alignment;
    GLint storage_offset_alignment;
    int formatted_image_loads;
} cs_gl_device;

typedef struct cs_gl_buffer {
    GLuint name;
    uint64_t size;     /* requested size */
    uint64_t capacity; /* allocated size (at least 16 bytes) */
} cs_gl_buffer;

typedef struct cs_gl_format {
    uint32_t dxgi;
    GLenum internal_format;
    GLenum transfer_format;
    GLenum transfer_type;
    uint32_t texel_size;
    int rgb32; /* three-channel 32-bit formats have no typed storage image access */
} cs_gl_format;

/*
 * DXGI formats used by the portable runtime. B8G8R8A8 textures are stored as
 * RGBA8 and transferred with GL_BGRA, so shaders observe the same component
 * order as a Direct3D B8G8R8A8_UNORM view (x is red).
 */
static const cs_gl_format CS_FORMATS[] = {
    {2, GL_RGBA32F, GL_RGBA, GL_FLOAT, 16, 0},
    {3, GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 16, 0},
    {4, GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 16, 0},
    {6, GL_RGB32F, GL_RGB, GL_FLOAT, 12, 1},
    {7, GL_RGB32UI, GL_RGB_INTEGER, GL_UNSIGNED_INT, 12, 1},
    {8, GL_RGB32I, GL_RGB_INTEGER, GL_INT, 12, 1},
    {11, GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, 8, 0},
    {16, GL_RG32F, GL_RG, GL_FLOAT, 8, 0},
    {17, GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 8, 0},
    {18, GL_RG32I, GL_RG_INTEGER, GL_INT, 8, 0},
    {28, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, 0},
    {35, GL_RG16, GL_RG, GL_UNSIGNED_SHORT, 4, 0},
    {41, GL_R32F, GL_RED, GL_FLOAT, 4, 0},
    {42, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 4, 0},
    {43, GL_R32I, GL_RED_INTEGER, GL_INT, 4, 0},
    {49, GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, 0},
    {56, GL_R16, GL_RED, GL_UNSIGNED_SHORT, 2, 0},
    {61, GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, 0},
    {87, GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4, 0},
};

static const cs_gl_format *cs_find_format(uint32_t dxgi) {
    for (size_t i = 0; i < sizeof(CS_FORMATS) / sizeof(CS_FORMATS[0]); i++) {
        if (CS_FORMATS[i].dxgi == dxgi) {
            return &CS_FORMATS[i];
        }
    }
    return NULL;
}

typedef struct cs_gl_texture {
    GLuint name;
    GLenum target;
    uint32_t dimension;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    const cs_gl_format *format;
} cs_gl_texture;

typedef struct cs_gl_sampler {
    GLuint name;
} cs_gl_sampler;

typedef struct cs_gl_pipeline {
    GLuint program;
    uint32_t threads[3];
} cs_gl_pipeline;

enum {
    CS_OP_DISPATCH = 1,
    CS_OP_BARRIER,
    CS_OP_COPY_BUFFER,
    CS_OP_FILL_BUFFER,
    CS_OP_COPY_BUFFER_TO_TEXTURE,
    CS_OP_COPY_TEXTURE,
};

typedef struct cs_gl_op {
    uint32_t kind;
    cs_gl_pipeline *pipeline;
    uint32_t groups[3];
    cs_binding *bindings;   /* owned copy; constant bytes point into owned memory */
    uint32_t binding_count;
    void *source;
    void *destination;
    uint64_t source_offset;
    uint64_t destination_offset;
    uint64_t size;
    uint64_t row_pitch;
    uint64_t slice_pitch;
    uint32_t value;
    uint32_t source_origin[3];
    uint32_t destination_origin[3];
    uint32_t extent[3];
} cs_gl_op;

typedef struct cs_gl_list {
    cs_gl_device *device;
    cs_gl_op *ops;
    size_t count;
    size_t capacity;
    int submitted;
} cs_gl_list;

/* Drains pending GL errors and reports the first one, if any. */
static int cs_gl_check(cs_gl_device *device, const char *operation, char *error, int32_t capacity) {
    GLenum first = GL_NO_ERROR;
    GLenum code;
    while ((code = device->gl.GetError()) != GL_NO_ERROR) {
        if (first == GL_NO_ERROR) {
            first = code;
        }
    }
    if (first != GL_NO_ERROR) {
        cs_set_error(error, capacity, "OpenGL error 0x%04X during %s", (unsigned)first, operation);
        return 1;
    }
    return 0;
}

static void cs_gl_clear_errors(cs_gl_device *device) {
    while (device->gl.GetError() != GL_NO_ERROR) {
    }
}

static int cs_has_extension(const char *list, const char *name) {
    if (list == NULL || name == NULL) {
        return 0;
    }
    size_t length = strlen(name);
    const char *cursor = list;
    while ((cursor = strstr(cursor, name)) != NULL) {
        int starts = cursor == list || cursor[-1] == ' ';
        int ends = cursor[length] == '\0' || cursor[length] == ' ';
        if (starts && ends) {
            return 1;
        }
        cursor += length;
    }
    return 0;
}

static int cs_gl_has_extension(cs_gl_device *device, const char *name) {
    GLint count = 0;
    device->gl.GetIntegerv(GL_NUM_EXTENSIONS, &count);
    for (GLint i = 0; i < count; i++) {
        const char *extension = (const char *)device->gl.GetStringi(GL_EXTENSIONS, (GLuint)i);
        if (extension != NULL && strcmp(extension, name) == 0) {
            return 1;
        }
    }
    return 0;
}

uint32_t cs_abi_version(void) { return CS_ABI_VERSION; }

static EGLDisplay cs_open_display(const char **platform) {
    const char *client = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    const char *requested = getenv("CROSSTL_COMPUTESHARP_EGL_PLATFORM");
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay display = EGL_NO_DISPLAY;

    int allow_surfaceless = requested == NULL || strcmp(requested, "surfaceless") == 0;
    int allow_device = requested == NULL || strcmp(requested, "device") == 0;
    int allow_default = requested == NULL || strcmp(requested, "default") == 0;

    if (allow_surfaceless && get_platform_display != NULL && cs_has_extension(client, "EGL_MESA_platform_surfaceless")) {
        display = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
        if (display != EGL_NO_DISPLAY) {
            *platform = "EGL_MESA_platform_surfaceless";
            return display;
        }
    }
    if (allow_device && get_platform_display != NULL && cs_has_extension(client, "EGL_EXT_platform_device")) {
        PFNEGLQUERYDEVICESEXTPROC query_devices = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        EGLDeviceEXT devices[8];
        EGLint count = 0;
        if (query_devices != NULL && query_devices(8, devices, &count) && count > 0) {
            display = get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
            if (display != EGL_NO_DISPLAY) {
                *platform = "EGL_EXT_platform_device";
                return display;
            }
        }
    }
    if (allow_default) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        *platform = "default display";
    }
    return display;
}

int cs_device_create(void **out_device, cs_device_info *info, char *error, int32_t error_capacity) {
    if (out_device == NULL || info == NULL) {
        cs_set_error(error, error_capacity, "invalid arguments");
        return 1;
    }
    *out_device = NULL;

    const char *platform = "none";
    EGLDisplay display = cs_open_display(&platform);
    if (display == EGL_NO_DISPLAY) {
        cs_set_error(error, error_capacity, "no EGL display is available (EGL error 0x%04X)", (unsigned)eglGetError());
        return 2;
    }
    EGLint egl_major = 0;
    EGLint egl_minor = 0;
    if (!eglInitialize(display, &egl_major, &egl_minor)) {
        cs_set_error(error, error_capacity, "eglInitialize failed on the %s (EGL error 0x%04X)", platform, (unsigned)eglGetError());
        return 3;
    }
    if (!eglBindAPI(EGL_OPENGL_API)) {
        cs_set_error(error, error_capacity, "eglBindAPI(EGL_OPENGL_API) failed (EGL error 0x%04X)", (unsigned)eglGetError());
        return 4;
    }

    const char *display_extensions = eglQueryString(display, EGL_EXTENSIONS);
    if (!cs_has_extension(display_extensions, "EGL_KHR_surfaceless_context")) {
        cs_set_error(error, error_capacity, "the %s lacks EGL_KHR_surfaceless_context", platform);
        return 5;
    }
    EGLConfig config = EGL_NO_CONFIG_KHR;
    if (!cs_has_extension(display_extensions, "EGL_KHR_no_config_context")) {
        const EGLint config_attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
        EGLint count = 0;
        if (!eglChooseConfig(display, config_attributes, &config, 1, &count) || count == 0) {
            cs_set_error(error, error_capacity, "no EGL configuration supports desktop OpenGL (EGL error 0x%04X)", (unsigned)eglGetError());
            return 6;
        }
    }
    const EGLint context_attributes[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT) {
        cs_set_error(error, error_capacity, "creating an OpenGL 4.5 core context failed (EGL error 0x%04X)", (unsigned)eglGetError());
        return 7;
    }
    if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
        cs_set_error(error, error_capacity, "eglMakeCurrent failed (EGL error 0x%04X)", (unsigned)eglGetError());
        eglDestroyContext(display, context);
        return 8;
    }

    cs_gl_device *device = (cs_gl_device *)calloc(1, sizeof(cs_gl_device));
    if (device == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
        return 9;
    }
    device->display = display;
    device->context = context;
#define CS_LOAD(type, name)                                                                 \
    device->gl.name = (type)eglGetProcAddress("gl" #name);                                  \
    if (device->gl.name == NULL) {                                                          \
        cs_set_error(error, error_capacity, "the OpenGL driver does not provide gl" #name); \
        cs_device_destroy(device);                                                          \
        return 10;                                                                          \
    }
    CS_GL_FUNCTIONS(CS_LOAD)
#undef CS_LOAD

    cs_gl_api *gl = &device->gl;
    GLint major = 0;
    GLint minor = 0;
    gl->GetIntegerv(GL_MAJOR_VERSION, &major);
    gl->GetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 5)) {
        cs_set_error(error, error_capacity, "OpenGL %d.%d is older than the required 4.5 core", major, minor);
        cs_device_destroy(device);
        return 11;
    }
    gl->GetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &device->uniform_offset_alignment);
    gl->GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &device->storage_offset_alignment);
    if (device->uniform_offset_alignment <= 0) {
        device->uniform_offset_alignment = 256;
    }
    device->formatted_image_loads = cs_gl_has_extension(device, "GL_EXT_shader_image_load_formatted");
    int subgroups = cs_gl_has_extension(device, "GL_KHR_shader_subgroup");

    const char *renderer = (const char *)gl->GetString(GL_RENDERER);
    const char *vendor = (const char *)gl->GetString(GL_VENDOR);
    const char *version = (const char *)gl->GetString(GL_VERSION);
    const char *glsl = (const char *)gl->GetString(GL_SHADING_LANGUAGE_VERSION);
    renderer = renderer != NULL ? renderer : "unknown renderer";
    int software = strstr(renderer, "llvmpipe") != NULL || strstr(renderer, "softpipe") != NULL ||
                   strstr(renderer, "swrast") != NULL;

    memset(info, 0, sizeof(*info));
    info->abi_version = CS_ABI_VERSION;
    info->is_hardware = software ? 0u : 1u;
    /* Double precision is core since OpenGL 4.0; the probe below compiles a double shader. */
    info->supports_fp64 = 1;
    info->unified_memory = software ? 1u : 0u;
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (software && pages > 0 && page_size > 0) {
        info->shared_memory = (uint64_t)pages * (uint64_t)page_size;
    }
    GLint value = 0;
    gl->GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &value);
    info->max_threads_per_group = (uint32_t)value;
    for (GLuint i = 0; i < 3; i++) {
        value = 0;
        gl->GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &value);
        if (i == 0) {
            info->max_group_size_x = (uint32_t)value;
        } else if (i == 1) {
            info->max_group_size_y = (uint32_t)value;
        } else {
            info->max_group_size_z = (uint32_t)value;
        }
    }
    uint32_t threads = 0;
    const char *lp_threads = getenv("LP_NUM_THREADS");
    if (software) {
        threads = lp_threads != NULL ? (uint32_t)strtoul(lp_threads, NULL, 10) : (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
        info->compute_units = threads != 0 ? threads : 1u;
    }
    GLint subgroup_size = 0;
    if (subgroups) {
        gl->GetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroup_size);
    }
    info->wavefront_size = subgroup_size > 0 ? (uint32_t)subgroup_size : 1u;
    cs_gl_clear_errors(device);

    snprintf(info->name, sizeof(info->name), "%s", renderer);
    snprintf(info->api, sizeof(info->api), "OpenGL %d.%d core", major, minor);
    snprintf(info->driver, sizeof(info->driver),
             "%s; GLSL %s; vendor %s; EGL %d.%d %s (%s); %s execution%s%s; "
             "formatted image loads %s; subgroup size %s",
             version != NULL ? version : "?", glsl != NULL ? glsl : "?", vendor != NULL ? vendor : "?",
             egl_major, egl_minor, eglQueryString(display, EGL_VENDOR), platform,
             software ? "software" : "hardware", software ? ", threads " : "",
             software ? (lp_threads != NULL ? lp_threads : "all processors") : "",
             device->formatted_image_loads ? "supported" : "unsupported",
             subgroups ? "reported" : "not exposed (reported as 1)");

    *out_device = device;
    return 0;
}

void cs_device_destroy(void *handle) {
    cs_gl_device *device = (cs_gl_device *)handle;
    if (device == NULL) {
        return;
    }
    if (device->context != EGL_NO_CONTEXT) {
        eglMakeCurrent(device->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(device->display, device->context);
    }
    /* The display is shared by every device on this platform; it is not terminated. */
    eglReleaseThread();
    free(device);
}

int cs_format_supported(void *device, uint32_t dimension, uint32_t format, uint32_t usage) {
    (void)device;
    const cs_gl_format *desc = cs_find_format(format);
    if (desc == NULL || dimension < CS_TEXTURE_1D || dimension > CS_TEXTURE_3D) {
        return 0;
    }
    /* OpenGL image load/store, like Direct3D 12 typed UAVs, has no 96-bit formats. */
    if (desc->rgb32 && (usage & CS_USAGE_SHADER_WRITE) != 0) {
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Buffers                                                                    */
/* ------------------------------------------------------------------------- */

int cs_buffer_create(void *handle, uint64_t size, void **out_buffer, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_api *gl = &device->gl;
    cs_gl_clear_errors(device);
    cs_gl_buffer *buffer = (cs_gl_buffer *)calloc(1, sizeof(cs_gl_buffer));
    if (buffer == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 1;
    }
    buffer->size = size;
    buffer->capacity = size < 16 ? 16 : size;
    gl->CreateBuffers(1, &buffer->name);
    gl->NamedBufferStorage(buffer->name, (GLsizeiptr)buffer->capacity, NULL, GL_DYNAMIC_STORAGE_BIT);
    /* Direct3D 12 committed resources start zeroed. */
    gl->ClearNamedBufferSubData(buffer->name, GL_R8UI, 0, (GLsizeiptr)buffer->capacity, GL_RED_INTEGER, GL_UNSIGNED_BYTE, NULL);
    if (cs_gl_check(device, "buffer creation", error, error_capacity) != 0) {
        gl->DeleteBuffers(1, &buffer->name);
        free(buffer);
        return 2;
    }
    *out_buffer = buffer;
    return 0;
}

void cs_buffer_destroy(void *handle, void *buffer_handle) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_buffer *buffer = (cs_gl_buffer *)buffer_handle;
    if (buffer == NULL) {
        return;
    }
    if (device != NULL) {
        device->gl.DeleteBuffers(1, &buffer->name);
    }
    free(buffer);
}

int cs_buffer_write(void *handle, void *buffer_handle, uint64_t offset, const void *data, uint64_t size, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_buffer *buffer = (cs_gl_buffer *)buffer_handle;
    if (offset + size > buffer->capacity) {
        cs_set_error(error, error_capacity, "buffer write out of range");
        return 1;
    }
    if (size == 0) {
        return 0;
    }
    cs_gl_clear_errors(device);
    device->gl.NamedBufferSubData(buffer->name, (GLintptr)offset, (GLsizeiptr)size, data);
    return cs_gl_check(device, "buffer write", error, error_capacity) != 0 ? 2 : 0;
}

int cs_buffer_read(void *handle, void *buffer_handle, uint64_t offset, void *data, uint64_t size, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_buffer *buffer = (cs_gl_buffer *)buffer_handle;
    if (offset + size > buffer->capacity) {
        cs_set_error(error, error_capacity, "buffer read out of range");
        return 1;
    }
    if (size == 0) {
        return 0;
    }
    cs_gl_clear_errors(device);
    device->gl.GetNamedBufferSubData(buffer->name, (GLintptr)offset, (GLsizeiptr)size, data);
    return cs_gl_check(device, "buffer read", error, error_capacity) != 0 ? 2 : 0;
}

/* ------------------------------------------------------------------------- */
/* Textures                                                                   */
/* ------------------------------------------------------------------------- */

static void cs_texture_region(cs_gl_texture *texture, uint32_t *y, uint32_t *z, uint32_t *height, uint32_t *depth) {
    if (texture->dimension == CS_TEXTURE_1D) {
        *y = 0;
        *height = 1;
    }
    if (texture->dimension != CS_TEXTURE_3D) {
        *z = 0;
        *depth = 1;
    }
}

static int cs_region_in_bounds(cs_gl_texture *texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth) {
    return (uint64_t)x + width <= texture->width && (uint64_t)y + height <= texture->height &&
           (uint64_t)z + depth <= texture->depth;
}

static void cs_upload_region(cs_gl_device *device, cs_gl_texture *texture, uint32_t x, uint32_t y, uint32_t z,
                             uint32_t width, uint32_t height, uint32_t depth, const void *pixels) {
    cs_gl_api *gl = &device->gl;
    const cs_gl_format *format = texture->format;
    switch (texture->dimension) {
    case CS_TEXTURE_1D:
        gl->TextureSubImage1D(texture->name, 0, (GLint)x, (GLsizei)width, format->transfer_format, format->transfer_type, pixels);
        break;
    case CS_TEXTURE_2D:
        gl->TextureSubImage2D(texture->name, 0, (GLint)x, (GLint)y, (GLsizei)width, (GLsizei)height,
                              format->transfer_format, format->transfer_type, pixels);
        break;
    default:
        gl->TextureSubImage3D(texture->name, 0, (GLint)x, (GLint)y, (GLint)z, (GLsizei)width, (GLsizei)height,
                              (GLsizei)depth, format->transfer_format, format->transfer_type, pixels);
        break;
    }
}

static void cs_set_pixel_store(cs_gl_device *device, GLenum row_length, GLenum image_height, GLint row, GLint image) {
    device->gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
    device->gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    device->gl.PixelStorei(row_length, row);
    device->gl.PixelStorei(image_height, image);
}

int cs_texture_create(void *handle, uint32_t dimension, uint32_t format, uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, void **out_texture, char *error, int32_t error_capacity) {
    (void)usage;
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_api *gl = &device->gl;
    const cs_gl_format *desc = cs_find_format(format);
    if (desc == NULL) {
        cs_set_error(error, error_capacity, "unsupported texture format %u", format);
        return 1;
    }
    if (width == 0) {
        cs_set_error(error, error_capacity, "texture width must be positive");
        return 2;
    }
    cs_gl_texture *texture = (cs_gl_texture *)calloc(1, sizeof(cs_gl_texture));
    if (texture == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 3;
    }
    texture->dimension = dimension;
    texture->format = desc;
    texture->width = width;
    texture->height = dimension == CS_TEXTURE_1D ? 1 : height;
    texture->depth = dimension == CS_TEXTURE_3D ? depth : 1;
    cs_gl_clear_errors(device);
    switch (dimension) {
    case CS_TEXTURE_1D:
        texture->target = GL_TEXTURE_1D;
        gl->CreateTextures(GL_TEXTURE_1D, 1, &texture->name);
        gl->TextureStorage1D(texture->name, 1, desc->internal_format, (GLsizei)width);
        break;
    case CS_TEXTURE_2D:
        texture->target = GL_TEXTURE_2D;
        gl->CreateTextures(GL_TEXTURE_2D, 1, &texture->name);
        gl->TextureStorage2D(texture->name, 1, desc->internal_format, (GLsizei)width, (GLsizei)texture->height);
        break;
    case CS_TEXTURE_3D:
        texture->target = GL_TEXTURE_3D;
        gl->CreateTextures(GL_TEXTURE_3D, 1, &texture->name);
        gl->TextureStorage3D(texture->name, 1, desc->internal_format, (GLsizei)width, (GLsizei)texture->height, (GLsizei)texture->depth);
        break;
    default:
        free(texture);
        cs_set_error(error, error_capacity, "invalid texture dimension %u", dimension);
        return 4;
    }
    gl->TextureParameteri(texture->name, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl->TextureParameteri(texture->name, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl->TextureParameteri(texture->name, GL_TEXTURE_MAX_LEVEL, 0);
    /* Direct3D 12 committed resources start zeroed; match that for reads before writes. */
    gl->ClearTexImage(texture->name, 0, desc->transfer_format, desc->transfer_type, NULL);
    if (cs_gl_check(device, "texture creation", error, error_capacity) != 0) {
        gl->DeleteTextures(1, &texture->name);
        free(texture);
        return 5;
    }
    *out_texture = texture;
    return 0;
}

void cs_texture_destroy(void *handle, void *texture_handle) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_texture *texture = (cs_gl_texture *)texture_handle;
    if (texture == NULL) {
        return;
    }
    if (device != NULL) {
        device->gl.DeleteTextures(1, &texture->name);
    }
    free(texture);
}

/* Copies a pitched host region into (pack=0) or out of (pack=1) tightly packed rows. */
static void cs_repack(uint8_t *tight, uint8_t *pitched, uint64_t row_bytes, uint32_t height, uint32_t depth,
                      uint64_t row_pitch, uint64_t slice_pitch, int to_pitched) {
    for (uint32_t k = 0; k < depth; k++) {
        for (uint32_t j = 0; j < height; j++) {
            uint8_t *packed_row = tight + ((uint64_t)k * height + j) * row_bytes;
            uint8_t *pitched_row = pitched + (uint64_t)k * slice_pitch + (uint64_t)j * row_pitch;
            if (to_pitched) {
                memcpy(pitched_row, packed_row, (size_t)row_bytes);
            } else {
                memcpy(packed_row, pitched_row, (size_t)row_bytes);
            }
        }
    }
}

int cs_texture_write(void *handle, void *texture_handle, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, const void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_texture *texture = (cs_gl_texture *)texture_handle;
    cs_texture_region(texture, &y, &z, &height, &depth);
    if (!cs_region_in_bounds(texture, x, y, z, width, height, depth)) {
        cs_set_error(error, error_capacity, "texture write out of range");
        return 1;
    }
    if (width == 0 || height == 0 || depth == 0) {
        return 0;
    }
    uint64_t row_bytes = (uint64_t)width * texture->format->texel_size;
    uint8_t *tight = (uint8_t *)malloc((size_t)(row_bytes * height * depth));
    if (tight == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 2;
    }
    cs_repack(tight, (uint8_t *)data, row_bytes, height, depth, row_pitch, slice_pitch, 0);
    cs_gl_clear_errors(device);
    device->gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cs_set_pixel_store(device, GL_UNPACK_ROW_LENGTH, GL_UNPACK_IMAGE_HEIGHT, 0, 0);
    cs_upload_region(device, texture, x, y, z, width, height, depth, tight);
    free(tight);
    return cs_gl_check(device, "texture write", error, error_capacity) != 0 ? 3 : 0;
}

int cs_texture_read(void *handle, void *texture_handle, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, void *data, uint64_t row_pitch, uint64_t slice_pitch, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_texture *texture = (cs_gl_texture *)texture_handle;
    cs_texture_region(texture, &y, &z, &height, &depth);
    if (!cs_region_in_bounds(texture, x, y, z, width, height, depth)) {
        cs_set_error(error, error_capacity, "texture read out of range");
        return 1;
    }
    if (width == 0 || height == 0 || depth == 0) {
        return 0;
    }
    uint64_t row_bytes = (uint64_t)width * texture->format->texel_size;
    uint64_t total = row_bytes * height * depth;
    uint8_t *tight = (uint8_t *)malloc((size_t)total);
    if (tight == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 2;
    }
    cs_gl_clear_errors(device);
    device->gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    cs_set_pixel_store(device, GL_PACK_ROW_LENGTH, GL_PACK_IMAGE_HEIGHT, 0, 0);
    device->gl.GetTextureSubImage(texture->name, 0, (GLint)x, (GLint)y, (GLint)z, (GLsizei)width, (GLsizei)height,
                                  (GLsizei)depth, texture->format->transfer_format, texture->format->transfer_type,
                                  (GLsizei)total, tight);
    if (cs_gl_check(device, "texture read", error, error_capacity) != 0) {
        free(tight);
        return 3;
    }
    cs_repack(tight, (uint8_t *)data, row_bytes, height, depth, row_pitch, slice_pitch, 1);
    free(tight);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Samplers and pipelines                                                     */
/* ------------------------------------------------------------------------- */

int cs_sampler_create(void *handle, uint32_t filter, uint32_t address_mode, void **out_sampler, char *error, int32_t error_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_api *gl = &device->gl;
    cs_gl_sampler *sampler = (cs_gl_sampler *)calloc(1, sizeof(cs_gl_sampler));
    if (sampler == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 1;
    }
    GLint mode = GL_CLAMP_TO_EDGE;
    if (address_mode == CS_ADDRESS_REPEAT) {
        mode = GL_REPEAT;
    } else if (address_mode == CS_ADDRESS_MIRROR) {
        mode = GL_MIRRORED_REPEAT;
    }
    cs_gl_clear_errors(device);
    gl->CreateSamplers(1, &sampler->name);
    gl->SamplerParameteri(sampler->name, GL_TEXTURE_MIN_FILTER, filter == CS_FILTER_LINEAR ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST);
    gl->SamplerParameteri(sampler->name, GL_TEXTURE_MAG_FILTER, filter == CS_FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
    gl->SamplerParameteri(sampler->name, GL_TEXTURE_WRAP_S, mode);
    gl->SamplerParameteri(sampler->name, GL_TEXTURE_WRAP_T, mode);
    gl->SamplerParameteri(sampler->name, GL_TEXTURE_WRAP_R, mode);
    if (cs_gl_check(device, "sampler creation", error, error_capacity) != 0) {
        gl->DeleteSamplers(1, &sampler->name);
        free(sampler);
        return 2;
    }
    *out_sampler = sampler;
    return 0;
}

void cs_sampler_destroy(void *handle, void *sampler_handle) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_sampler *sampler = (cs_gl_sampler *)sampler_handle;
    if (sampler == NULL) {
        return;
    }
    if (device != NULL) {
        device->gl.DeleteSamplers(1, &sampler->name);
    }
    free(sampler);
}

int cs_pipeline_create(void *handle, const char *source, const char *entry_point, uint32_t threads_x, uint32_t threads_y, uint32_t threads_z, void **out_pipeline, char *log, int32_t log_capacity) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_api *gl = &device->gl;
    if (log != NULL && log_capacity > 0) {
        log[0] = '\0';
    }
    if (entry_point == NULL || strcmp(entry_point, "main") != 0) {
        cs_set_error(log, log_capacity, "OpenGL compute shaders must use the entry point main, not %s", entry_point != NULL ? entry_point : "(null)");
        return 1;
    }
    cs_gl_clear_errors(device);
    GLuint shader = gl->CreateShader(GL_COMPUTE_SHADER);
    gl->ShaderSource(shader, 1, &source, NULL);
    gl->CompileShader(shader);
    GLint status = 0;
    gl->GetShaderiv(shader, GL_COMPILE_STATUS, &status);
    GLint length = 0;
    gl->GetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length > 1 && log != NULL && log_capacity > 0) {
        gl->GetShaderInfoLog(shader, log_capacity, NULL, log);
    }
    if (!status) {
        if (length <= 1) {
            cs_set_error(log, log_capacity, "GLSL compilation failed without a log");
        }
        gl->DeleteShader(shader);
        return 2;
    }
    GLuint program = gl->CreateProgram();
    gl->AttachShader(program, shader);
    gl->LinkProgram(program);
    gl->DeleteShader(shader);
    gl->GetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        length = 0;
        gl->GetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        if (length > 1 && log != NULL && log_capacity > 0) {
            gl->GetProgramInfoLog(program, log_capacity, NULL, log);
        } else {
            cs_set_error(log, log_capacity, "GLSL program link failed without a log");
        }
        gl->DeleteProgram(program);
        return 3;
    }
    GLint size[3] = {0, 0, 0};
    gl->GetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, size);
    if ((uint32_t)size[0] != threads_x || (uint32_t)size[1] != threads_y || (uint32_t)size[2] != threads_z) {
        cs_set_error(log, log_capacity, "the program declares a %dx%dx%d work group but the shader requires %ux%ux%u",
                     size[0], size[1], size[2], threads_x, threads_y, threads_z);
        gl->DeleteProgram(program);
        return 4;
    }
    if (cs_gl_check(device, "pipeline creation", log, log_capacity) != 0) {
        gl->DeleteProgram(program);
        return 5;
    }
    cs_gl_pipeline *pipeline = (cs_gl_pipeline *)calloc(1, sizeof(cs_gl_pipeline));
    if (pipeline == NULL) {
        gl->DeleteProgram(program);
        cs_set_error(log, log_capacity, "out of host memory");
        return 6;
    }
    pipeline->program = program;
    pipeline->threads[0] = threads_x;
    pipeline->threads[1] = threads_y;
    pipeline->threads[2] = threads_z;
    *out_pipeline = pipeline;
    return 0;
}

void cs_pipeline_destroy(void *handle, void *pipeline_handle) {
    cs_gl_device *device = (cs_gl_device *)handle;
    cs_gl_pipeline *pipeline = (cs_gl_pipeline *)pipeline_handle;
    if (pipeline == NULL) {
        return;
    }
    if (device != NULL) {
        device->gl.DeleteProgram(pipeline->program);
    }
    free(pipeline);
}

/* ------------------------------------------------------------------------- */
/* Command lists                                                              */
/* ------------------------------------------------------------------------- */

int cs_list_create(void *handle, void **out_list, char *error, int32_t error_capacity) {
    cs_gl_list *list = (cs_gl_list *)calloc(1, sizeof(cs_gl_list));
    if (list == NULL) {
        cs_set_error(error, error_capacity, "out of host memory");
        return 1;
    }
    list->device = (cs_gl_device *)handle;
    *out_list = list;
    return 0;
}

static cs_gl_op *cs_list_append(cs_gl_list *list, uint32_t kind, char *error, int32_t error_capacity) {
    if (list->submitted) {
        cs_set_error(error, error_capacity, "command list already submitted");
        return NULL;
    }
    if (list->count == list->capacity) {
        size_t capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        cs_gl_op *ops = (cs_gl_op *)realloc(list->ops, capacity * sizeof(cs_gl_op));
        if (ops == NULL) {
            cs_set_error(error, error_capacity, "out of host memory");
            return NULL;
        }
        list->ops = ops;
        list->capacity = capacity;
    }
    cs_gl_op *op = &list->ops[list->count++];
    memset(op, 0, sizeof(*op));
    op->kind = kind;
    return op;
}

int cs_list_dispatch(void *list_handle, void *pipeline_handle, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, const cs_binding *bindings, uint32_t binding_count, char *error, int32_t error_capacity) {
    cs_gl_list *list = (cs_gl_list *)list_handle;
    for (uint32_t i = 0; i < binding_count; i++) {
        uint32_t kind = bindings[i].kind;
        if (kind < CS_BIND_BUFFER || kind > CS_BIND_UNIFORM_BUFFER) {
            cs_set_error(error, error_capacity, "invalid binding kind %u", kind);
            return 2;
        }
        if (kind != CS_BIND_CONSTANT_BYTES && bindings[i].object == NULL) {
            cs_set_error(error, error_capacity, "binding %u of kind %u has no resource", i, kind);
            return 3;
        }
    }
    cs_gl_op *op = cs_list_append(list, CS_OP_DISPATCH, error, error_capacity);
    if (op == NULL) {
        return 1;
    }
    op->pipeline = (cs_gl_pipeline *)pipeline_handle;
    op->groups[0] = groups_x;
    op->groups[1] = groups_y;
    op->groups[2] = groups_z;
    if (binding_count > 0) {
        size_t constant_bytes = 0;
        for (uint32_t i = 0; i < binding_count; i++) {
            if (bindings[i].kind == CS_BIND_CONSTANT_BYTES) {
                constant_bytes += (size_t)bindings[i].size;
            }
        }
        /* One allocation holds the binding table followed by the copied constant data. */
        uint8_t *storage = (uint8_t *)malloc(binding_count * sizeof(cs_binding) + constant_bytes);
        if (storage == NULL) {
            list->count--;
            cs_set_error(error, error_capacity, "out of host memory");
            return 4;
        }
        cs_binding *copy = (cs_binding *)storage;
        uint8_t *constants = storage + binding_count * sizeof(cs_binding);
        memcpy(copy, bindings, binding_count * sizeof(cs_binding));
        for (uint32_t i = 0; i < binding_count; i++) {
            if (copy[i].kind == CS_BIND_CONSTANT_BYTES) {
                memcpy(constants, bindings[i].data, (size_t)bindings[i].size);
                copy[i].data = constants;
                constants += bindings[i].size;
            }
        }
        op->bindings = copy;
        op->binding_count = binding_count;
    }
    return 0;
}

int cs_list_barrier(void *list_handle, char *error, int32_t error_capacity) {
    return cs_list_append((cs_gl_list *)list_handle, CS_OP_BARRIER, error, error_capacity) == NULL ? 1 : 0;
}

int cs_list_copy_buffer(void *list_handle, void *source, uint64_t source_offset, void *destination, uint64_t destination_offset, uint64_t size, char *error, int32_t error_capacity) {
    cs_gl_buffer *src = (cs_gl_buffer *)source;
    cs_gl_buffer *dst = (cs_gl_buffer *)destination;
    if (source_offset + size > src->capacity || destination_offset + size > dst->capacity) {
        cs_set_error(error, error_capacity, "buffer copy out of range");
        return 1;
    }
    cs_gl_op *op = cs_list_append((cs_gl_list *)list_handle, CS_OP_COPY_BUFFER, error, error_capacity);
    if (op == NULL) {
        return 2;
    }
    op->source = src;
    op->destination = dst;
    op->source_offset = source_offset;
    op->destination_offset = destination_offset;
    op->size = size;
    return 0;
}

int cs_list_fill_buffer(void *list_handle, void *buffer, uint64_t offset, uint64_t size, uint32_t value, char *error, int32_t error_capacity) {
    cs_gl_buffer *target = (cs_gl_buffer *)buffer;
    if (offset + size > target->capacity) {
        cs_set_error(error, error_capacity, "buffer fill out of range");
        return 1;
    }
    if (value > 0xFFu) {
        cs_set_error(error, error_capacity, "buffer fills take a byte value");
        return 2;
    }
    cs_gl_op *op = cs_list_append((cs_gl_list *)list_handle, CS_OP_FILL_BUFFER, error, error_capacity);
    if (op == NULL) {
        return 3;
    }
    op->destination = target;
    op->destination_offset = offset;
    op->size = size;
    op->value = value;
    return 0;
}

int cs_list_copy_buffer_to_texture(void *list_handle, void *buffer, uint64_t offset, uint64_t row_pitch, uint64_t slice_pitch, void *texture_handle, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity) {
    cs_gl_texture *texture = (cs_gl_texture *)texture_handle;
    cs_texture_region(texture, &y, &z, &height, &depth);
    if (!cs_region_in_bounds(texture, x, y, z, width, height, depth)) {
        cs_set_error(error, error_capacity, "texture upload out of range");
        return 1;
    }
    uint32_t texel = texture->format->texel_size;
    if (row_pitch % texel != 0 || (height > 1 && row_pitch < (uint64_t)width * texel) ||
        (depth > 1 && (row_pitch == 0 || slice_pitch % row_pitch != 0))) {
        cs_set_error(error, error_capacity, "texture upload pitches %llu/%llu are not whole texel rows",
                     (unsigned long long)row_pitch, (unsigned long long)slice_pitch);
        return 2;
    }
    cs_gl_op *op = cs_list_append((cs_gl_list *)list_handle, CS_OP_COPY_BUFFER_TO_TEXTURE, error, error_capacity);
    if (op == NULL) {
        return 3;
    }
    op->source = buffer;
    op->source_offset = offset;
    op->row_pitch = row_pitch;
    op->slice_pitch = slice_pitch;
    op->destination = texture;
    op->destination_origin[0] = x;
    op->destination_origin[1] = y;
    op->destination_origin[2] = z;
    op->extent[0] = width;
    op->extent[1] = height;
    op->extent[2] = depth;
    return 0;
}

int cs_list_copy_texture(void *list_handle, void *source, uint32_t source_x, uint32_t source_y, uint32_t source_z, void *destination, uint32_t destination_x, uint32_t destination_y, uint32_t destination_z, uint32_t width, uint32_t height, uint32_t depth, char *error, int32_t error_capacity) {
    cs_gl_texture *src = (cs_gl_texture *)source;
    cs_gl_texture *dst = (cs_gl_texture *)destination;
    cs_texture_region(src, &source_y, &source_z, &height, &depth);
    if (!cs_region_in_bounds(src, source_x, source_y, source_z, width, height, depth) ||
        !cs_region_in_bounds(dst, destination_x, destination_y, destination_z, width, height, depth)) {
        cs_set_error(error, error_capacity, "texture copy out of range");
        return 1;
    }
    cs_gl_op *op = cs_list_append((cs_gl_list *)list_handle, CS_OP_COPY_TEXTURE, error, error_capacity);
    if (op == NULL) {
        return 2;
    }
    op->source = src;
    op->destination = dst;
    op->source_origin[0] = source_x;
    op->source_origin[1] = source_y;
    op->source_origin[2] = source_z;
    op->destination_origin[0] = destination_x;
    op->destination_origin[1] = destination_y;
    op->destination_origin[2] = destination_z;
    op->extent[0] = width;
    op->extent[1] = height;
    op->extent[2] = depth;
    return 0;
}

static uint64_t cs_align(uint64_t value, uint64_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

static int cs_execute_dispatch(cs_gl_device *device, cs_gl_op *op, GLuint constants, uint64_t *constant_cursor, char *error, int32_t error_capacity) {
    cs_gl_api *gl = &device->gl;
    gl->UseProgram(op->pipeline->program);
    for (uint32_t i = 0; i < op->binding_count; i++) {
        const cs_binding *binding = &op->bindings[i];
        switch (binding->kind) {
        case CS_BIND_BUFFER:
        case CS_BIND_UNIFORM_BUFFER: {
            cs_gl_buffer *buffer = (cs_gl_buffer *)binding->object;
            uint64_t size = binding->size != 0 ? binding->size : buffer->size - binding->offset;
            if (size == 0 || binding->offset + size > buffer->capacity) {
                size = buffer->capacity - binding->offset;
            }
            GLenum target = binding->kind == CS_BIND_BUFFER ? GL_SHADER_STORAGE_BUFFER : GL_UNIFORM_BUFFER;
            gl->BindBufferRange(target, binding->slot, buffer->name, (GLintptr)binding->offset, (GLsizeiptr)size);
            break;
        }
        case CS_BIND_CONSTANT_BYTES: {
            uint64_t offset = cs_align(*constant_cursor, (uint64_t)device->uniform_offset_alignment);
            gl->BindBufferRange(GL_UNIFORM_BUFFER, binding->slot, constants, (GLintptr)offset, (GLsizeiptr)binding->size);
            *constant_cursor = offset + binding->size;
            break;
        }
        case CS_BIND_TEXTURE: {
            cs_gl_texture *texture = (cs_gl_texture *)binding->object;
            gl->BindTextureUnit(binding->slot, texture->name);
            gl->BindSampler(binding->slot, 0);
            break;
        }
        case CS_BIND_IMAGE: {
            cs_gl_texture *texture = (cs_gl_texture *)binding->object;
            GLenum access = GL_READ_WRITE;
            if ((binding->access & CS_USAGE_SHADER_WRITE) == 0) {
                access = GL_READ_ONLY;
            }
            gl->BindImageTexture(binding->slot, texture->name, 0, texture->dimension == CS_TEXTURE_3D ? GL_TRUE : GL_FALSE,
                                 0, access, texture->format->internal_format);
            break;
        }
        case CS_BIND_SAMPLER: {
            cs_gl_sampler *sampler = (cs_gl_sampler *)binding->object;
            gl->BindSampler(binding->slot, sampler->name);
            break;
        }
        default:
            cs_set_error(error, error_capacity, "invalid binding kind %u", binding->kind);
            return 1;
        }
    }
    gl->DispatchCompute(op->groups[0], op->groups[1], op->groups[2]);
    gl->MemoryBarrier(GL_ALL_BARRIER_BITS);
    return cs_gl_check(device, "dispatch", error, error_capacity) != 0 ? 2 : 0;
}

int cs_list_submit(void *list_handle, char *error, int32_t error_capacity) {
    cs_gl_list *list = (cs_gl_list *)list_handle;
    cs_gl_device *device = list->device;
    cs_gl_api *gl = &device->gl;
    if (list->submitted) {
        cs_set_error(error, error_capacity, "command list already submitted");
        return 1;
    }
    list->submitted = 1;
    cs_gl_clear_errors(device);

    /* Stage the constant data of every dispatch in one uniform buffer. */
    uint64_t alignment = (uint64_t)device->uniform_offset_alignment;
    uint64_t constant_size = 0;
    for (size_t i = 0; i < list->count; i++) {
        cs_gl_op *op = &list->ops[i];
        for (uint32_t j = 0; op->kind == CS_OP_DISPATCH && j < op->binding_count; j++) {
            if (op->bindings[j].kind == CS_BIND_CONSTANT_BYTES) {
                constant_size = cs_align(constant_size, alignment) + op->bindings[j].size;
            }
        }
    }
    GLuint constants = 0;
    if (constant_size > 0) {
        uint8_t *staging = (uint8_t *)calloc(1, (size_t)constant_size);
        if (staging == NULL) {
            cs_set_error(error, error_capacity, "out of host memory");
            return 2;
        }
        uint64_t cursor = 0;
        for (size_t i = 0; i < list->count; i++) {
            cs_gl_op *op = &list->ops[i];
            for (uint32_t j = 0; op->kind == CS_OP_DISPATCH && j < op->binding_count; j++) {
                if (op->bindings[j].kind == CS_BIND_CONSTANT_BYTES) {
                    cursor = cs_align(cursor, alignment);
                    memcpy(staging + cursor, op->bindings[j].data, (size_t)op->bindings[j].size);
                    cursor += op->bindings[j].size;
                }
            }
        }
        gl->CreateBuffers(1, &constants);
        gl->NamedBufferStorage(constants, (GLsizeiptr)constant_size, staging, 0);
        free(staging);
    }

    int status = 0;
    uint64_t constant_cursor = 0;
    for (size_t i = 0; i < list->count && status == 0; i++) {
        cs_gl_op *op = &list->ops[i];
        switch (op->kind) {
        case CS_OP_DISPATCH:
            status = cs_execute_dispatch(device, op, constants, &constant_cursor, error, error_capacity);
            break;
        case CS_OP_BARRIER:
            gl->MemoryBarrier(GL_ALL_BARRIER_BITS);
            break;
        case CS_OP_COPY_BUFFER: {
            cs_gl_buffer *src = (cs_gl_buffer *)op->source;
            cs_gl_buffer *dst = (cs_gl_buffer *)op->destination;
            if (op->size > 0) {
                gl->CopyNamedBufferSubData(src->name, dst->name, (GLintptr)op->source_offset, (GLintptr)op->destination_offset, (GLsizeiptr)op->size);
            }
            break;
        }
        case CS_OP_FILL_BUFFER: {
            cs_gl_buffer *dst = (cs_gl_buffer *)op->destination;
            uint8_t value = (uint8_t)op->value;
            if (op->size > 0) {
                gl->ClearNamedBufferSubData(dst->name, GL_R8UI, (GLintptr)op->destination_offset, (GLsizeiptr)op->size, GL_RED_INTEGER, GL_UNSIGNED_BYTE, &value);
            }
            break;
        }
        case CS_OP_COPY_BUFFER_TO_TEXTURE: {
            cs_gl_buffer *src = (cs_gl_buffer *)op->source;
            cs_gl_texture *dst = (cs_gl_texture *)op->destination;
            uint32_t texel = dst->format->texel_size;
            GLint row_length = op->row_pitch != 0 ? (GLint)(op->row_pitch / texel) : 0;
            GLint image_height = op->row_pitch != 0 && op->slice_pitch != 0 ? (GLint)(op->slice_pitch / op->row_pitch) : 0;
            gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, src->name);
            cs_set_pixel_store(device, GL_UNPACK_ROW_LENGTH, GL_UNPACK_IMAGE_HEIGHT,
                               dst->dimension == CS_TEXTURE_1D ? 0 : row_length,
                               dst->dimension == CS_TEXTURE_3D ? image_height : 0);
            cs_upload_region(device, dst, op->destination_origin[0], op->destination_origin[1], op->destination_origin[2],
                             op->extent[0], op->extent[1], op->extent[2], (const void *)(uintptr_t)op->source_offset);
            gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            cs_set_pixel_store(device, GL_UNPACK_ROW_LENGTH, GL_UNPACK_IMAGE_HEIGHT, 0, 0);
            break;
        }
        case CS_OP_COPY_TEXTURE: {
            cs_gl_texture *src = (cs_gl_texture *)op->source;
            cs_gl_texture *dst = (cs_gl_texture *)op->destination;
            gl->CopyImageSubData(src->name, src->target, 0, (GLint)op->source_origin[0], (GLint)op->source_origin[1], (GLint)op->source_origin[2],
                                 dst->name, dst->target, 0, (GLint)op->destination_origin[0], (GLint)op->destination_origin[1], (GLint)op->destination_origin[2],
                                 (GLsizei)op->extent[0], (GLsizei)op->extent[1], (GLsizei)op->extent[2]);
            break;
        }
        default:
            cs_set_error(error, error_capacity, "invalid recorded operation %u", op->kind);
            status = 3;
            break;
        }
        if (status == 0 && op->kind != CS_OP_DISPATCH) {
            gl->MemoryBarrier(GL_ALL_BARRIER_BITS);
            if (cs_gl_check(device, "recorded copy", error, error_capacity) != 0) {
                status = 4;
            }
        }
    }
    gl->Finish();
    if (constants != 0) {
        gl->DeleteBuffers(1, &constants);
    }
    if (status == 0 && cs_gl_check(device, "submission", error, error_capacity) != 0) {
        status = 5;
    }
    return status;
}

void cs_list_destroy(void *list_handle) {
    cs_gl_list *list = (cs_gl_list *)list_handle;
    if (list == NULL) {
        return;
    }
    for (size_t i = 0; i < list->count; i++) {
        free(list->ops[i].bindings);
    }
    free(list->ops);
    free(list);
}
