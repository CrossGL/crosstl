
static const int MathError_DivisionByZero = 0;
static const int MathError_InvalidInput = 1;
static const int MathError_Overflow = 2;
static const int MathError_Underflow = 3;
static const int VectorOp_Add = 0;
static const int VectorOp_Multiply = 1;
static const int VectorOp_Cross = 2;
static const int VectorOp_Normalize = 3;
static const int ShaderError_MathError = 0;
static const int ShaderError_TextureError = 1;
static const int ShaderError_BufferError = 2;
static const int ShaderError_InvalidState = 3;
static const int LightingModel_Phong = 0;
static const int LightingModel_PBR = 1;
static const int LightingModel_Toon = 2;
static const int RenderCommand_Draw = 0;
static const int RenderCommand_Clear = 1;
static const int RenderCommand_SetState = 2;
static const int RenderOutput_Success = 0;
static const int RenderOutput_Clear = 1;
static const int RenderOutput_StateSet = 2;

static const int Option_Some = 0;
static const int Option_None = 1;
static const int Result_Ok = 0;
static const int Result_Err = 1;

struct Vec3_float
{
    float x;
    float y;
    float z;
};

struct ShaderError
{
    int variant;
    int MathError_0;
    int TextureError_0;
    int BufferError_0;
};

struct LightingModel
{
    int variant;
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float shininess;
    float3 albedo;
    float metallic;
    float roughness;
    float ao;
    float3 base_color;
    int levels;
    float smoothing;
};

struct Option_int
{
    int variant;
    int Some_0;
};

struct Result_float_MathError
{
    int variant;
    float Ok_0;
    int Err_0;
};

struct Numeric
{
    float add;
    float mul;
    float zero;
    float one;
};
struct VectorOps
{
    float dot;
    float cross;
    float magnitude;
    float normalize;
};
struct Geometry
{
    float3 center;
    float3 normal;
};
struct Material
{
    float3 albedo;
    float roughness;
};
struct LightingEnvironment
{
    float3 direction;
    float intensity;
};
struct RenderedFrame
{
    float4 color;
    float depth;
};
struct VertexInput
{
    float3 position : POSITION;
    float3 normal : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float4 color : TEXCOORD2;
};
struct VertexOutput
{
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 color : TEXCOORD3;
};
struct FragmentInput
{
    float3 world_position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 color : TEXCOORD3;
};
struct Matrix3x3_float
{
    Vec3_float row0;
    Vec3_float row1;
    Vec3_float row2;
};

struct RenderOutput
{
    int variant;
    RenderedFrame Success_0;
    float4 color;
    float depth;
};

struct Option_Material
{
    int variant;
    Material Some_0;
};

struct Result_RenderedFrame_MathError
{
    int variant;
    RenderedFrame Ok_0;
    int Err_0;
};

struct Result_int_ShaderError
{
    int variant;
    int Ok_0;
    ShaderError Err_0;
};

struct Result_Vec3_float_MathError
{
    int variant;
    Vec3_float Ok_0;
    int Err_0;
};

struct RenderState_float
{
    Matrix3x3_float transform;
    Vec3_float position;
    Vec3_float color;
    Option_int material_id;
    LightingModel lighting_model;
};

struct Result_RenderOutput_ShaderError
{
    int variant;
    RenderOutput Ok_0;
    ShaderError Err_0;
};

struct RenderCommand
{
    int variant;
    Geometry geometry;
    Option_Material material;
    float4x4 transform;
    LightingEnvironment lighting;
    float4 color;
    float depth;
    RenderState_float state;
};

ShaderError ShaderError_MathError_make(int payload0)
{
    ShaderError result;
    result.variant = ShaderError_MathError;
    result.TextureError_0 = int(0);
    result.BufferError_0 = int(0);
    result.MathError_0 = payload0;
    return result;
}

ShaderError ShaderError_TextureError_make(int payload0)
{
    ShaderError result;
    result.variant = ShaderError_TextureError;
    result.MathError_0 = int(0);
    result.BufferError_0 = int(0);
    result.TextureError_0 = payload0;
    return result;
}

ShaderError ShaderError_BufferError_make(int payload0)
{
    ShaderError result;
    result.variant = ShaderError_BufferError;
    result.MathError_0 = int(0);
    result.TextureError_0 = int(0);
    result.BufferError_0 = payload0;
    return result;
}

ShaderError ShaderError_InvalidState_make()
{
    ShaderError result;
    result.variant = ShaderError_InvalidState;
    result.MathError_0 = int(0);
    result.TextureError_0 = int(0);
    result.BufferError_0 = int(0);
    return result;
}

LightingModel LightingModel_Phong_make(float3 payload0, float3 payload1, float3 payload2, float payload3)
{
    LightingModel result;
    result.variant = LightingModel_Phong;
    result.albedo = float3(0);
    result.metallic = float(0);
    result.roughness = float(0);
    result.ao = float(0);
    result.base_color = float3(0);
    result.levels = int(0);
    result.smoothing = float(0);
    result.ambient = payload0;
    result.diffuse = payload1;
    result.specular = payload2;
    result.shininess = payload3;
    return result;
}

LightingModel LightingModel_PBR_make(float3 payload0, float payload1, float payload2, float payload3)
{
    LightingModel result;
    result.variant = LightingModel_PBR;
    result.ambient = float3(0);
    result.diffuse = float3(0);
    result.specular = float3(0);
    result.shininess = float(0);
    result.base_color = float3(0);
    result.levels = int(0);
    result.smoothing = float(0);
    result.albedo = payload0;
    result.metallic = payload1;
    result.roughness = payload2;
    result.ao = payload3;
    return result;
}

LightingModel LightingModel_Toon_make(float3 payload0, int payload1, float payload2)
{
    LightingModel result;
    result.variant = LightingModel_Toon;
    result.ambient = float3(0);
    result.diffuse = float3(0);
    result.specular = float3(0);
    result.shininess = float(0);
    result.albedo = float3(0);
    result.metallic = float(0);
    result.roughness = float(0);
    result.ao = float(0);
    result.base_color = payload0;
    result.levels = payload1;
    result.smoothing = payload2;
    return result;
}

RenderCommand RenderCommand_Draw_make(Geometry payload0, Option_Material payload1, float4x4 payload2,
                                      LightingEnvironment payload3)
{
    RenderCommand result;
    result.variant = RenderCommand_Draw;
    result.color = float4(0);
    result.depth = float(0);
    result.state = RenderState_float(0);
    result.geometry = payload0;
    result.material = payload1;
    result.transform = payload2;
    result.lighting = payload3;
    return result;
}

RenderCommand RenderCommand_Clear_make(float4 payload0, float payload1)
{
    RenderCommand result;
    result.variant = RenderCommand_Clear;
    result.geometry = Geometry(0);
    result.material = Option_Material(0);
    result.transform = float4x4(0);
    result.lighting = LightingEnvironment(0);
    result.state = RenderState_float(0);
    result.color = payload0;
    result.depth = payload1;
    return result;
}

RenderCommand RenderCommand_SetState_make(RenderState_float payload0)
{
    RenderCommand result;
    result.variant = RenderCommand_SetState;
    result.geometry = Geometry(0);
    result.material = Option_Material(0);
    result.transform = float4x4(0);
    result.lighting = LightingEnvironment(0);
    result.color = float4(0);
    result.depth = float(0);
    result.state = payload0;
    return result;
}

RenderOutput RenderOutput_Success_make(RenderedFrame payload0)
{
    RenderOutput result;
    result.variant = RenderOutput_Success;
    result.color = float4(0);
    result.depth = float(0);
    result.Success_0 = payload0;
    return result;
}

RenderOutput RenderOutput_Clear_make(float4 payload0, float payload1)
{
    RenderOutput result;
    result.variant = RenderOutput_Clear;
    result.Success_0 = RenderedFrame(0);
    result.color = payload0;
    result.depth = payload1;
    return result;
}

RenderOutput RenderOutput_StateSet_make()
{
    RenderOutput result;
    result.variant = RenderOutput_StateSet;
    result.Success_0 = RenderedFrame(0);
    result.color = float4(0);
    result.depth = float(0);
    return result;
}

Option_Material Option_Material_Some_make(Material payload0)
{
    Option_Material result;
    result.variant = Option_Some;
    result.Some_0 = payload0;
    return result;
}

Option_Material Option_Material_None_make()
{
    Option_Material result;
    result.variant = Option_None;
    result.Some_0 = Material(0);
    return result;
}

Option_int Option_int_Some_make(int payload0)
{
    Option_int result;
    result.variant = Option_Some;
    result.Some_0 = payload0;
    return result;
}

Option_int Option_int_None_make()
{
    Option_int result;
    result.variant = Option_None;
    result.Some_0 = int(0);
    return result;
}

Result_RenderOutput_ShaderError Result_RenderOutput_ShaderError_Ok_make(RenderOutput payload0)
{
    Result_RenderOutput_ShaderError result;
    result.variant = Result_Ok;
    result.Err_0 = ShaderError(0);
    result.Ok_0 = payload0;
    return result;
}

Result_RenderOutput_ShaderError Result_RenderOutput_ShaderError_Err_make(ShaderError payload0)
{
    Result_RenderOutput_ShaderError result;
    result.variant = Result_Err;
    result.Ok_0 = RenderOutput(0);
    result.Err_0 = payload0;
    return result;
}

Result_RenderedFrame_MathError Result_RenderedFrame_MathError_Ok_make(RenderedFrame payload0)
{
    Result_RenderedFrame_MathError result;
    result.variant = Result_Ok;
    result.Err_0 = int(0);
    result.Ok_0 = payload0;
    return result;
}

Result_RenderedFrame_MathError Result_RenderedFrame_MathError_Err_make(int payload0)
{
    Result_RenderedFrame_MathError result;
    result.variant = Result_Err;
    result.Ok_0 = RenderedFrame(0);
    result.Err_0 = payload0;
    return result;
}

Result_int_ShaderError Result_int_ShaderError_Ok_make(int payload0)
{
    Result_int_ShaderError result;
    result.variant = Result_Ok;
    result.Err_0 = ShaderError(0);
    result.Ok_0 = payload0;
    return result;
}

Result_int_ShaderError Result_int_ShaderError_Err_make(ShaderError payload0)
{
    Result_int_ShaderError result;
    result.variant = Result_Err;
    result.Ok_0 = int(0);
    result.Err_0 = payload0;
    return result;
}

Result_Vec3_float_MathError Result_Vec3_float_MathError_Ok_make(Vec3_float payload0)
{
    Result_Vec3_float_MathError result;
    result.variant = Result_Ok;
    result.Err_0 = int(0);
    result.Ok_0 = payload0;
    return result;
}

Result_Vec3_float_MathError Result_Vec3_float_MathError_Err_make(int payload0)
{
    Result_Vec3_float_MathError result;
    result.variant = Result_Err;
    result.Ok_0 = Vec3_float(0);
    result.Err_0 = payload0;
    return result;
}

Result_float_MathError Result_float_MathError_Ok_make(float payload0)
{
    Result_float_MathError result;
    result.variant = Result_Ok;
    result.Err_0 = int(0);
    result.Ok_0 = payload0;
    return result;
}

Result_float_MathError Result_float_MathError_Err_make(int payload0)
{
    Result_float_MathError result;
    result.variant = Result_Err;
    result.Ok_0 = float(0);
    result.Err_0 = payload0;
    return result;
}

float4x4 mvp_matrix;
float4x4 model_matrix;
float3x3 normal_matrix;
LightingModel lighting_model;
float3 light_direction;
float3 camera_position;
Texture2D main_texture : register(t0);
SamplerState main_textureSampler : register(s0);
Result_float_MathError safe_divide_float(float a, float b)
{
    {
        float zero = b;
        if ((zero == 0.0))
        {
            return Result_float_MathError_Err_make(MathError_DivisionByZero);
        }
        else
        {
            {
                return Result_float_MathError_Ok_make((a / b));
            }
        }
    }
    return Result_float_MathError(int(0), float(0), int(0));
}

Result_Vec3_float_MathError vector_operation_float(Vec3_float v1, Vec3_float v2, int op)
{
    if ((op == VectorOp_Add))
    {
        return Result_Vec3_float_MathError_Ok_make(Vec3_float((v1.x + v2.x), (v1.y + v2.y), (v1.z + v2.z)));
    }
    else if ((op == VectorOp_Multiply))
    {
        return Result_Vec3_float_MathError_Ok_make(Vec3_float((v1.x * v2.x), (v1.y * v2.y), (v1.z * v2.z)));
    }
    else if ((op == VectorOp_Cross))
    {
        Vec3_float cross_result = Vec3_float(((v1.y * v2.z) - (v1.z * v2.y)), ((v1.z * v2.x) - (v1.x * v2.z)),
                                             ((v1.x * v2.y) - (v1.y * v2.x)));
        return Result_Vec3_float_MathError_Ok_make(cross_result);
    }
    else if ((op == VectorOp_Normalize))
    {
        float mag_squared = (((v1.x * v1.x) + (v1.y * v1.y)) + (v1.z * v1.z));
        Result_float_MathError __crossgl_match_subject_0 = safe_divide_float(1.0, sqrt(mag_squared));
        if ((__crossgl_match_subject_0.variant == Result_Ok))
        {
            float inv_mag = __crossgl_match_subject_0.Ok_0;
            return Result_Vec3_float_MathError_Ok_make(
                Vec3_float((v1.x * inv_mag), (v1.y * inv_mag), (v1.z * inv_mag)));
        }
        else if ((__crossgl_match_subject_0.variant == Result_Err))
        {
            int e = __crossgl_match_subject_0.Err_0;
            return Result_Vec3_float_MathError_Err_make(e);
        }
    }
    return Result_Vec3_float_MathError(int(0), Vec3_float(float(0), float(0), float(0)), int(0));
}

float3 process_lighting_model(LightingModel model, float3 light_dir, float3 view_dir, float3 normal)
{
    if ((model.variant == LightingModel_Phong))
    {
        float3 ambient = model.ambient;
        float3 diffuse = model.diffuse;
        float3 specular = model.specular;
        float shininess = model.shininess;
        float n_dot_l = max(0.0, dot(normal, light_dir));
        float3 reflect_dir = reflect(-light_dir, normal);
        float r_dot_v = max(0.0, dot(reflect_dir, view_dir));
        return ((ambient + (diffuse * n_dot_l)) + (specular * pow(r_dot_v, shininess)));
    }
    else if ((model.variant == LightingModel_PBR))
    {
        float3 albedo = model.albedo;
        float metallic = model.metallic;
        float roughness = model.roughness;
        float ao = model.ao;
        float3 half_vector = normalize((light_dir + view_dir));
        float n_dot_l = max(0.0, dot(normal, light_dir));
        float n_dot_v = max(0.0, dot(normal, view_dir));
        float n_dot_h = max(0.0, dot(normal, half_vector));
        float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
        float3 fresnel = (f0 + ((1.0 - f0) * pow((1.0 - n_dot_v), 5.0)));
        float alpha = (roughness * roughness);
        float distribution = (alpha / (3.14159 * pow((((n_dot_h * n_dot_h) * (alpha - 1.0)) + 1.0), 2.0)));
        float geometry = ((n_dot_l * n_dot_v) / (((n_dot_l + n_dot_v) - (n_dot_l * n_dot_v)) + 0.001));
        float3 brdf = (((distribution * geometry) * fresnel) / (((4.0 * n_dot_l) * n_dot_v) + 0.001));
        float3 diffuse_contribution = ((((1.0 - fresnel) * (1.0 - metallic)) * albedo) / 3.14159);
        return (((diffuse_contribution + brdf) * n_dot_l) * ao);
    }
    else if ((model.variant == LightingModel_Toon))
    {
        float3 base_color = model.base_color;
        int levels = model.levels;
        float smoothing = model.smoothing;
        if ((levels > 0))
        {
            float n_dot_l = dot(normal, light_dir);
            float toon_level = (floor((n_dot_l * float(levels))) / float(levels));
            float smooth_factor = smoothstep((toon_level - smoothing), (toon_level + smoothing), n_dot_l);
            return (base_color * smooth_factor);
        }
        else
        {
            if ((model.variant == LightingModel_Toon))
            {
                float3 base_color = model.base_color;
                return (base_color * 0.5);
            }
        }
    }
    else
    {
        if ((model.variant == LightingModel_Toon))
        {
            float3 base_color = model.base_color;
            return (base_color * 0.5);
        }
    }
    return float3(0);
}

Result_int_ShaderError validate_material(Material material)
{
    if (((material.roughness >= 0.0) == true))
    {
        return Result_int_ShaderError_Ok_make(1);
    }
    else if (((material.roughness >= 0.0) == false))
    {
        return Result_int_ShaderError_Err_make(ShaderError_InvalidState_make());
    }
    return Result_int_ShaderError(int(0), int(0), ShaderError(0));
}

Geometry transform_geometry(Geometry shape, float4x4 transform)
{
    float4 transformed_center = mul(transform, float4(shape.center, 1.0));
    return Geometry(transformed_center.xyz, shape.normal);
    return Geometry(float3(0), float3(0));
}

Result_RenderedFrame_MathError apply_lighting(Geometry shape, Material material, LightingEnvironment lighting)
{
    float light_strength = (max(0.0, dot(shape.normal, normalize(lighting.direction))) * lighting.intensity);
    float3 shaded_color = (material.albedo * light_strength);
    return Result_RenderedFrame_MathError_Ok_make(RenderedFrame(float4(shaded_color, 1.0), 1.0));
    return Result_RenderedFrame_MathError(int(0), RenderedFrame(float4(0), float(0)), int(0));
}

bool validate_state(RenderState_float state)
{
    if ((state.material_id.variant == Option_Some))
    {
        int id = state.material_id.Some_0;
        return (id >= 0);
    }
    else if ((state.material_id.variant == Option_None))
    {
        return true;
    }
    return false;
}

Result_RenderOutput_ShaderError process_render_command(RenderCommand command)
{
    if (((command.variant == RenderCommand_Draw)) && ((command.material.variant == Option_Some)))
    {
        Geometry geometry = command.geometry;
        Material material = command.material.Some_0;
        float4x4 transform = command.transform;
        LightingEnvironment lighting = command.lighting;
        Result_int_ShaderError __crossgl_match_subject_1 = validate_material(material);
        if ((__crossgl_match_subject_1.variant == Result_Ok))
        {
            Geometry transformed_geometry = transform_geometry(geometry, transform);
            Result_RenderedFrame_MathError lit_result = apply_lighting(transformed_geometry, material, lighting);
            if ((lit_result.variant == Result_Ok))
            {
                RenderedFrame output = lit_result.Ok_0;
                return Result_RenderOutput_ShaderError_Ok_make(RenderOutput_Success_make(output));
            }
            else if ((lit_result.variant == Result_Err))
            {
                int lighting_error = lit_result.Err_0;
                return Result_RenderOutput_ShaderError_Err_make(ShaderError_MathError_make(lighting_error));
            }
        }
        else if ((__crossgl_match_subject_1.variant == Result_Err))
        {
            ShaderError validation_error = __crossgl_match_subject_1.Err_0;
            return Result_RenderOutput_ShaderError_Err_make(ShaderError_InvalidState_make());
        }
    }
    else if (((command.variant == RenderCommand_Draw)) && ((command.material.variant == Option_None)))
    {
        Geometry geometry = command.geometry;
        return Result_RenderOutput_ShaderError_Err_make(ShaderError_InvalidState_make());
    }
    else if ((command.variant == RenderCommand_Clear))
    {
        float4 color = command.color;
        float depth = command.depth;
        return Result_RenderOutput_ShaderError_Ok_make(RenderOutput_Clear_make(color, depth));
    }
    else if ((command.variant == RenderCommand_SetState))
    {
        RenderState_float state = command.state;
        bool __crossgl_match_subject_2 = validate_state(state);
        if ((__crossgl_match_subject_2 == true))
        {
            return Result_RenderOutput_ShaderError_Ok_make(RenderOutput_StateSet_make());
        }
        else if ((__crossgl_match_subject_2 == false))
        {
            return Result_RenderOutput_ShaderError_Err_make(ShaderError_InvalidState_make());
        }
    }
    return Result_RenderOutput_ShaderError(int(0), RenderOutput(0), ShaderError(0));
}

// Vertex Shader
VertexOutput VSMain(VertexInput input)
{
    float4 world_pos = mul(model_matrix, float4(input.position, 1.0));
    float4 clip_pos = mul(mvp_matrix, float4(input.position, 1.0));
    float3 world_normal = normalize(mul(normal_matrix, input.normal));
    Vec3_float normal_vec3 = Vec3_float(world_normal.x, world_normal.y, world_normal.z);
    float3 processed_normal;
    Result_Vec3_float_MathError __crossgl_match_subject_3 =
        vector_operation_float(normal_vec3, normal_vec3, VectorOp_Normalize);
    if ((__crossgl_match_subject_3.variant == Result_Ok))
    {
        Vec3_float normalized = __crossgl_match_subject_3.Ok_0;
        processed_normal = float3(normalized.x, normalized.y, normalized.z);
    }
    else if ((__crossgl_match_subject_3.variant == Result_Err))
    {
        processed_normal = world_normal;
    }
    return VertexOutput(clip_pos, world_pos.xyz, processed_normal, input.uv, input.color);
    return VertexOutput(float4(0), float3(0), float3(0), float2(0), float4(0));
}

// Fragment Shader
float4 PSMain(FragmentInput input) : SV_TARGET
{
    float3 view_dir = normalize((camera_position - input.world_position));
    float3 normal = normalize(input.normal);
    float3 light_dir = normalize(-light_direction);
    float4 tex_color = main_texture.Sample(main_textureSampler, input.uv);
    float3 lighting_contribution = process_lighting_model(lighting_model, light_dir, view_dir, normal);
    float3 final_color = ((tex_color.rgb * lighting_contribution) * input.color.rgb);
    return float4(final_color, (tex_color.a * input.color.a));
    return float4(0);
}
