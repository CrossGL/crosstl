float4x4 _crossgl_inverse_float4x4(float4x4 m)
{
    float4x4 a = m;
    float4x4 inv = float4x4(1.0);
    for (int i = 0; i < 4; i++)
    {
        int pivotRow = i;
        float pivotAbs = abs(a[i][i]);
        for (int row = i + 1; row < 4; row++)
        {
            float candidateAbs = abs(a[row][i]);
            if (candidateAbs > pivotAbs)
            {
                pivotAbs = candidateAbs;
                pivotRow = row;
            }
        }
        if (pivotRow != i)
        {
            float4 tmpA = a[i];
            a[i] = a[pivotRow];
            a[pivotRow] = tmpA;
            float4 tmpInv = inv[i];
            inv[i] = inv[pivotRow];
            inv[pivotRow] = tmpInv;
        }
        float invPivot = 1.0 / a[i][i];
        a[i] = a[i] * invPivot;
        inv[i] = inv[i] * invPivot;
        for (int row = 0; row < 4; row++)
        {
            if (row != i)
            {
                float factor = a[row][i];
                a[row] = a[row] - factor * a[i];
                inv[row] = inv[row] - factor * inv[i];
            }
        }
    }
    return inv;
}

static const float PI = 3.14159265359;
static const float EPSILON = 0.0001;
static const int MAX_ITERATIONS = 64;
static const float3 UP_VECTOR = float3(0.0, 1.0, 0.0);

struct Material
{
    float3 albedo;
    float roughness;
    float metallic;
    float3 emissive;
    float opacity;
    bool hasNormalMap;
};

struct Light
{
    float3 position;
    float3 color;
    float intensity;
    float radius;
    bool castShadows;
    float4x4 viewProjection;
};

struct Scene
{
    Material materials[4];
    Light lights[8];
    float3 ambientLight;
    float time;
    float elapsedTime;
    int activeLightCount;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
};

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : TEXCOORD0;
    float3 tangent : TEXCOORD1;
    float3 bitangent : TEXCOORD2;
    float2 texCoord0 : TEXCOORD3;
    float2 texCoord1 : TEXCOORD4;
    float4 color : TEXCOORD5;
    int materialIndex : TEXCOORD6;
};

struct VertexOutput
{
    float3 worldPosition : TEXCOORD0;
    float3 worldNormal : TEXCOORD1;
    float3 worldTangent : TEXCOORD2;
    float3 worldBitangent : TEXCOORD3;
    float2 texCoord0 : TEXCOORD4;
    float2 texCoord1 : TEXCOORD5;
    float4 color : TEXCOORD6;
    float3x3 TBN : TEXCOORD7;
    int materialIndex : TEXCOORD8;
    float4 clipPosition : SV_Position;
};

struct FragmentOutput
{
    float4 color : SV_Target;
    float4 normalBuffer : SV_Target1;
    float4 positionBuffer : SV_Target2;
    float depth : SV_Depth;
};

struct GlobalUniforms
{
    Scene scene;
    float3 cameraPosition;
    float globalRoughness;
    float2 screenSize;
    float nearPlane;
    float farPlane;
    int frameCount;
    float noiseValues[1024];
};

[[vk::binding(0, 0)]] Sampler2D<float4> Material_albedoMap : register(t0);
[[vk::binding(1, 0)]] Sampler2D<float4> Material_normalMap : register(t1);
[[vk::binding(2, 0)]] Sampler2D<float4> Material_metallicRoughnessMap : register(t2);
float distributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return num / max(denom, EPSILON);
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / max(denom, EPSILON);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float noise3D(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = frac(sin(dot(i, float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n001 = frac(sin(dot(i + float3(0.0, 0.0, 1.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n010 = frac(sin(dot(i + float3(0.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n011 = frac(sin(dot(i + float3(0.0, 1.0, 1.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n100 = frac(sin(dot(i + float3(1.0, 0.0, 0.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n101 = frac(sin(dot(i + float3(1.0, 0.0, 1.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n110 = frac(sin(dot(i + float3(1.0, 1.0, 0.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n111 = frac(sin(dot(i + float3(1.0, 1.0, 1.0), float3(13.534, 43.5234, 243.32))) * 4453.0);
    float n00 = lerp(n000, n001, u.z);
    float n01 = lerp(n010, n011, u.z);
    float n10 = lerp(n100, n101, u.z);
    float n11 = lerp(n110, n111, u.z);
    float n0 = lerp(n00, n01, u.y);
    float n1 = lerp(n10, n11, u.y);
    return lerp(n0, n1, u.x);
}

float fbm(float3 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        if (i >= MAX_ITERATIONS)
        {
            break;
        }
        sum += amplitude * noise3D(p * frequency);
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return sum;
}

float4 samplePlanarProjection(Sampler2D<float4> tex, float3 worldPos, float3 normal)
{
    float3 absNormal = abs(normal);
    bool useX = absNormal.x >= absNormal.y && absNormal.x >= absNormal.z;
    bool useY = !useX && absNormal.y >= absNormal.z;
    float2 uv;
    if (useX)
    {
        uv = worldPos.zy * 0.5 + 0.5;
        if (normal.x < 0.0)
        {
            uv.x = 1.0 - uv.x;
        }
    }
    else
    {
        if (useY)
        {
            uv = worldPos.xz * 0.5 + 0.5;
            if (normal.y < 0.0)
            {
                uv.y = 1.0 - uv.y;
            }
        }
        else
        {
            uv = worldPos.xy * 0.5 + 0.5;
            if (normal.z < 0.0)
            {
                uv.x = 1.0 - uv.x;
            }
        }
    }
    return tex.Sample(uv);
}

// Vertex Shader
uniform GlobalUniforms globals;
[shader("vertex")] VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    float4x4 modelMatrix = float4x4(1.0);
    float4x4 viewMatrix = globals.scene.viewMatrix;
    float4x4 projectionMatrix = globals.scene.projectionMatrix;
    float4x4 modelViewMatrix = mul(viewMatrix, modelMatrix);
    float4x4 modelViewProjectionMatrix = mul(projectionMatrix, modelViewMatrix);
    float3x3 normalMatrix = float3x3(transpose(_crossgl_inverse_float4x4(modelMatrix)));
    float4 worldPosition = mul(modelMatrix, float4(input.position, 1.0));
    float3 worldNormal = normalize(mul(normalMatrix, input.normal));
    float3 worldTangent = normalize(mul(normalMatrix, input.tangent));
    float3 worldBitangent = normalize(mul(normalMatrix, input.bitangent));
    float3x3 TBN = float3x3(worldTangent, worldBitangent, worldNormal);
    float displacement = fbm(worldPosition.xyz + globals.scene.time * 0.1, 4, 2.0, 0.5) * 0.1;
    if (input.materialIndex > 0)
    {
        worldPosition.xyz += worldNormal * displacement;
    }
    float3 viewDir = normalize(globals.cameraPosition - worldPosition.xyz);
    float fresnel = pow(1.0 - max(0.0, dot(worldNormal, viewDir)), 5.0);
    if (input.materialIndex < globals.scene.activeLightCount)
    {
        output.color = input.color * float4(1.0, 1.0, 1.0, 1.0);
        for (int i = 0; i < 4; i++)
        {
            if (i >= globals.frameCount % 5)
            {
                break;
            }
            Light light = globals.scene.lights[i];
            float3 lightDir = normalize(light.position - worldPosition.xyz);
            float lightDistance = length(light.position - worldPosition.xyz);
            float attenuation = 1.0 / (1.0 + lightDistance * lightDistance);
            float lightIntensity = light.intensity * attenuation;
            output.color.rgb += light.color * lightIntensity * max(0.0, dot(worldNormal, lightDir)) * 0.025;
        }
    }
    else
    {
        output.color = input.color;
        if (globals.globalRoughness > 0.5)
        {
            if (fresnel > 0.7)
            {
                output.color.a *= 0.8;
            }
            else
            {
                output.color.a *= 0.9;
            }
        }
    }
    output.worldPosition = worldPosition.xyz;
    output.worldNormal = worldNormal;
    output.worldTangent = worldTangent;
    output.worldBitangent = worldBitangent;
    output.texCoord0 = input.texCoord0;
    output.texCoord1 = input.texCoord1;
    output.TBN = TBN;
    output.materialIndex = input.materialIndex;
    output.clipPosition = mul(modelViewProjectionMatrix, float4(input.position, 1.0));
    return output;
}

// Fragment Shader
[[vk::binding(3, 0)]] uniform Sampler2D<float4> shadowMap : register(t3);
float shadowCalculation(float4 fragPosLightSpace, int iteration, VertexOutput input)
{
    if (iteration > 3)
    {
        return 0.0;
    }
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = shadowMap.Sample(projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias =
        max(0.05 * (1.0 - dot(input.worldNormal, normalize(globals.cameraPosition - input.worldPosition))), 0.005);
    float shadow = (currentDepth - bias > closestDepth ? 1.0 : 0.0);
    float pcfDepth = 0.0;
    float2 texelSize = 1.0 / float2(globals.screenSize);
    float offset = globals.noiseValues[((iteration * 4) % 16)] * 0.001;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            float pcfDepth = shadowMap.Sample(projCoords.xy + float2(x, y) * texelSize + float2(offset)).r;
            shadow += (currentDepth - bias > pcfDepth ? 1.0 : 0.0);
        }
    }
    shadow /= 9.0;
    if (projCoords.z > 1.0)
    {
        shadow = 0.0;
    }
    return shadow;
}

[shader("fragment")] FragmentOutput PSMain(VertexOutput input) {
    FragmentOutput output;
    Material material = globals.scene.materials[input.materialIndex];
    float4 albedoValue = Material_albedoMap.Sample(input.texCoord0);
    float4 normalValue = Material_normalMap.Sample(input.texCoord0);
    float4 metallicRoughnessValue = Material_metallicRoughnessMap.Sample(input.texCoord0);
    float3 normal = normalValue.xyz * 2.0 - 1.0;
    float3 worldNormal = normalize(mul(input.TBN, normal));
    float3 albedo = albedoValue.rgb * material.albedo;
    float metallic = metallicRoughnessValue.b * material.metallic;
    float roughness = metallicRoughnessValue.g * material.roughness;
    float ao = metallicRoughnessValue.r;
    float3 viewDir = normalize(globals.cameraPosition - input.worldPosition);
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 Lo = float3(0.0);
    for (int i = 0; i < globals.scene.activeLightCount; i++)
    {
        if (i >= 8)
        {
            break;
        }
        Light light = globals.scene.lights[i];
        float3 lightDir = normalize(light.position - input.worldPosition);
        float3 halfway = normalize(viewDir + lightDir);
        float distance = length(light.position - input.worldPosition);
        float attenuation = 1.0 / (distance * distance);
        float3 radiance = light.color * light.intensity * attenuation;
        float NDF = distributionGGX(worldNormal, halfway, roughness);
        float G = geometrySmith(worldNormal, viewDir, lightDir, roughness);
        float3 F = fresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);
        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= 1.0 - metallic;
        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(worldNormal, viewDir), 0.0) * max(dot(worldNormal, lightDir), 0.0) + EPSILON;
        float3 specular = numerator / denominator;
        float NdotL = max(dot(worldNormal, lightDir), 0.0);
        float shadow = 0.0;
        if (light.castShadows)
        {
            float4 fragPosLightSpace = mul(light.viewProjection, float4(input.worldPosition, 1.0));
            shadow = shadowCalculation(fragPosLightSpace, 0, input);
            for (int s = 0; s < 4; s++)
            {
                if (s >= globals.frameCount % 3)
                {
                    continue;
                }
                shadow += shadowCalculation(
                    fragPosLightSpace + float4(globals.noiseValues[(s % 16)] * 0.001, 0.0, 0.0, 0.0), s + 1, input);
            }
            shadow /= 5.0;
        }
        Lo += (1.0 - shadow) * (kD * albedo / PI + specular) * radiance * NdotL;
    }
    float3 ambient = globals.scene.ambientLight * albedo * ao;
    float3 color = ambient + Lo;
    color = color / (color + float3(1.0));
    color = pow(color, float3(1.0 / 2.2));
    output.color = float4(color, material.opacity * albedoValue.a);
    output.normalBuffer = float4(worldNormal * 0.5 + 0.5, 1.0);
    output.positionBuffer = float4(input.worldPosition, 1.0);
    output.depth = input.clipPosition.z / input.clipPosition.w;
    return output;
}

// Compute Shader
[[vk::binding(4, 0)]] RWTexture2D<float4> outputImage : register(u0);
[numthreads(16, 16, 1)][shader("compute")] void CSMain(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    int2 texCoord = int2(gl_GlobalInvocationID.xy);
    float2 screenSize = globals.screenSize;
    if (texCoord.x >= int(screenSize.x) || texCoord.y >= int(screenSize.y))
    {
        return;
    }
    float2 uv = float2(texCoord) / screenSize;
    float4 color = float4(0.0);
    float totalWeight = 0.0;
    float2 direction = float2(0.5) - uv;
    float len = length(direction);
    direction = normalize(direction);
    for (int i = 0; i < 32; i++)
    {
        if (i >= MAX_ITERATIONS)
        {
            break;
        }
        float t = float(i) / 32.0;
        float2 pos = uv + direction * t * len * 0.1;
        float noise = fbm(float3(pos * 10.0, globals.scene.time * 0.05), 4, 2.0, 0.5);
        float weight = 1.0 - t;
        weight = weight * weight;
        float3 noiseColor = float3(0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 0.0),
                                   0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 2.0),
                                   0.5 + 0.5 * sin(noise * 5.0 + globals.scene.time + 4.0));
        color.rgb += noiseColor * weight;
        totalWeight += weight;
        direction = mul(float2x2(cos(t * 3.0), -sin(t * 3.0), sin(t * 3.0), cos(t * 3.0)), direction);
    }
    color.rgb /= totalWeight;
    color.a = 1.0;
    float vignette = 1.0 - smoothstep(0.5, 1.0, length(uv - 0.5) * 1.5);
    color.rgb *= vignette;
    outputImage[texCoord] = color;
}
