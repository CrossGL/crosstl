#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

enum MathError { DivisionByZero, InvalidInput, Overflow, Underflow };

enum ShaderError { MathError, TextureError, BufferError, InvalidState };

struct RenderState {};

enum LightingModel { Phong, PBR, Toon };

OptionType variant;
ResultType variant;
T y;
T z;
Vec3 row1;
Vec3 row2;