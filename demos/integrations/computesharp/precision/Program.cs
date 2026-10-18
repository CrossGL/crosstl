using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ComputeSharp.IntrinsicProbe;

/// <summary>
/// Evaluates single-precision HLSL intrinsics on one device over a fixed input set.
/// </summary>
/// <remarks>
/// <c>export &lt;dir&gt;</c> writes the generated HLSL with its descriptors in the layout of
/// the discovery export, so <c>translate.py</c> can translate it. <c>run &lt;warp|default&gt;
/// &lt;dir&gt;</c> dispatches every probe and writes <c>inputs.bin</c>, one little-endian
/// float32 file per intrinsic and <c>probe.json</c> with the device and content hashes.
/// </remarks>
internal static class Program
{
    private const BindingFlags StaticProperties = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    private static readonly Probe[] Probes =
    [
        new("sin", "sin(x)", typeof(SinProbe), static (d, i, o) => d.For(i.Length, new SinProbe(i, o))),
        new("cos", "cos(x)", typeof(CosProbe), static (d, i, o) => d.For(i.Length, new CosProbe(i, o))),
        new("tan", "tan(x)", typeof(TanProbe), static (d, i, o) => d.For(i.Length, new TanProbe(i, o))),
        new("exp", "exp(x)", typeof(ExpProbe), static (d, i, o) => d.For(i.Length, new ExpProbe(i, o))),
        new("exp2", "exp2(x)", typeof(Exp2Probe), static (d, i, o) => d.For(i.Length, new Exp2Probe(i, o))),
        new("log", "log(x)", typeof(LogProbe), static (d, i, o) => d.For(i.Length, new LogProbe(i, o))),
        new("log2", "log2(x)", typeof(Log2Probe), static (d, i, o) => d.For(i.Length, new Log2Probe(i, o))),
        new("sqrt", "sqrt(x)", typeof(SqrtProbe), static (d, i, o) => d.For(i.Length, new SqrtProbe(i, o))),
        new("rsqrt", "rsqrt(x)", typeof(RsqrtProbe), static (d, i, o) => d.For(i.Length, new RsqrtProbe(i, o))),
        new("rcp", "rcp(x)", typeof(RcpProbe), static (d, i, o) => d.For(i.Length, new RcpProbe(i, o))),
        new("div", "1.0 / x", typeof(DivideProbe), static (d, i, o) => d.For(i.Length, new DivideProbe(i, o))),
        new("frac", "frac(x)", typeof(FracProbe), static (d, i, o) => d.For(i.Length, new FracProbe(i, o))),
        new("pow-2.2", "pow(abs(x), 2.2)", typeof(PowGammaProbe), static (d, i, o) => d.For(i.Length, new PowGammaProbe(i, o))),
        new("pow-0.4545", "pow(abs(x), 0.4545)", typeof(PowInverseGammaProbe), static (d, i, o) => d.For(i.Length, new PowInverseGammaProbe(i, o))),
        new("atan", "atan(x)", typeof(AtanProbe), static (d, i, o) => d.For(i.Length, new AtanProbe(i, o))),
        new("dot2", "dot(float2(x, y), float2(27.619, 57.583))", typeof(Dot2Probe), static (d, i, o) => d.For(i.Length, new Dot2Probe(i, o))),
        new("dot3", "dot(float3(x, y, z), float3(0.2126, 0.7152, 0.0722))", typeof(Dot3Probe), static (d, i, o) => d.For(i.Length, new Dot3Probe(i, o))),
        new("multiply-add", "x * 27.619 + y", typeof(MultiplyAddProbe), static (d, i, o) => d.For(i.Length, new MultiplyAddProbe(i, o))),
        new("mad", "mad(x, 27.619, y)", typeof(MadProbe), static (d, i, o) => d.For(i.Length, new MadProbe(i, o))),
        new("lerp", "lerp(x, y, frac(z))", typeof(LerpProbe), static (d, i, o) => d.For(i.Length, new LerpProbe(i, o))),
        new("atan2", "atan2(x, y)", typeof(Atan2Probe), static (d, i, o) => d.For(i.Length, new Atan2Probe(i, o))),
        new("length2", "length(float2(x, y))", typeof(Length2Probe), static (d, i, o) => d.For(i.Length, new Length2Probe(i, o))),
        new("normalize3", "normalize(float3(x, y, z)).x", typeof(Normalize3Probe), static (d, i, o) => d.For(i.Length, new Normalize3Probe(i, o))),
        new("smoothstep", "smoothstep(0.0, 1.0, frac(x))", typeof(SmoothStepProbe), static (d, i, o) => d.For(i.Length, new SmoothStepProbe(i, o))),
        new("pow", "pow(abs(x), frac(y) * 4.0)", typeof(PowProbe), static (d, i, o) => d.For(i.Length, new PowProbe(i, o))),
    ];

    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 2 && args[0] == "export")
            {
                Export(Path.GetFullPath(args[1]));
                return 0;
            }

            if (args.Length == 3 && args[0] == "run" && args[1] is "warp" or "default")
            {
                Run(args[1], Path.GetFullPath(args[2]));
                return 0;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }

        Console.Error.WriteLine("Usage: ComputeSharp.IntrinsicProbe export <new-dir> | run <warp|default> <new-dir>");
        return 2;
    }

    private static void Export(string output)
    {
        CreateNew(output);
        _ = Directory.CreateDirectory(Path.Combine(output, "hlsl"));

        List<object> shaders = [];
        foreach (Probe probe in Probes)
        {
            byte[] source = Encoding.UTF8.GetBytes(HlslSource(probe.Shader));
            string id = Hash(Encoding.UTF8.GetBytes(probe.Shader.FullName!));
            File.WriteAllBytes(Path.Combine(output, "hlsl", id + ".hlsl"), source);
            object ranges = Property(probe.Shader, "ResourceDescriptorRanges");
            object rangeArray = ranges.GetType().GetMethod("ToArray", Type.EmptyTypes)!.Invoke(ranges, null)!;
            shaders.Add(new
            {
                id,
                type = probe.Shader.FullName,
                path = "hlsl/" + id + ".hlsl",
                sha256 = Hash(source),
                threads = new[] { (int)Property(probe.Shader, "ThreadsX"), (int)Property(probe.Shader, "ThreadsY"), (int)Property(probe.Shader, "ThreadsZ") },
                constantBufferSize = (int)Property(probe.Shader, "ConstantBufferSize"),
                staticSamplerRequired = (bool)Property(probe.Shader, "IsStaticSamplerRequired"),
                resourceDescriptorRanges = JsonSerializer.SerializeToNode(rangeArray, rangeArray.GetType(), JsonOptions)
            });
        }

        WriteJson(Path.Combine(output, "discovery.json"), new
        {
            schemaVersion = 1,
            kind = "computesharp.intrinsic-probe-shaders",
            shaderCount = shaders.Count,
            shaders
        });
        Console.WriteLine($"Exported {shaders.Count} probe shaders");
    }

    private static void Run(string deviceKind, string output)
    {
        GraphicsDevice device = deviceKind == "warp"
            ? GraphicsDevice.QueryDevices(static info => !info.IsHardwareAccelerated).FirstOrDefault()
                ?? throw new InvalidOperationException("No WARP device is available")
            : GraphicsDevice.GetDefault();

        float[] inputs = Inputs(out List<object> sets);
        CreateNew(output);
        byte[] inputBytes = MemoryMarshal.AsBytes(inputs.AsSpan()).ToArray();
        File.WriteAllBytes(Path.Combine(output, "inputs.bin"), inputBytes);

        List<object> functions = [];
        using ReadOnlyBuffer<float> input = device.AllocateReadOnlyBuffer(inputs);
        using ReadWriteBuffer<float> results = device.AllocateReadWriteBuffer<float>(inputs.Length);
        foreach (Probe probe in Probes)
        {
            probe.Dispatch(device, input, results);
            float[] values = results.ToArray();
            byte[] bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
            string file = probe.Name + ".bin";
            File.WriteAllBytes(Path.Combine(output, file), bytes);
            functions.Add(new
            {
                name = probe.Name,
                expression = probe.Expression,
                shader = probe.Shader.FullName,
                hlslSha256 = Hash(Encoding.UTF8.GetBytes(HlslSource(probe.Shader))),
                file,
                sha256 = Hash(bytes)
            });
        }

        WriteJson(Path.Combine(output, "probe.json"), new
        {
            schemaVersion = 1,
            kind = "computesharp.intrinsic-probe",
            device = new
            {
                name = device.Name,
                hardware = device.IsHardwareAccelerated,
                runtime = typeof(GraphicsDevice).Assembly.GetName().Version?.ToString(),
                os = RuntimeInformation.OSDescription,
                architecture = RuntimeInformation.ProcessArchitecture.ToString()
            },
            inputs = new { file = "inputs.bin", count = inputs.Length, sha256 = Hash(inputBytes), sets },
            functions
        });
        Console.WriteLine($"Evaluated {Probes.Length} intrinsics over {inputs.Length} inputs on {device.Name}");
    }

    /// <summary>
    /// Builds the fixed input set: special values, integers and half-integers (the operands
    /// of sin-based hashes), a dense grid, runs of consecutive floats near anchors and
    /// log-uniform values from a fixed xorshift sequence.
    /// </summary>
    private static float[] Inputs(out List<object> sets)
    {
        List<float> values = [];
        List<object> described = [];

        void Add(string name, IEnumerable<float> items)
        {
            int start = values.Count;
            values.AddRange(items);
            described.Add(new { name, offset = start, count = values.Count - start });
        }

        Add("special", [0f, -0f, float.Epsilon, -float.Epsilon, 1.17549435e-38f, float.MaxValue, float.MinValue, float.PositiveInfinity, float.NegativeInfinity, float.NaN]);
        Add("integers", Enumerable.Range(-32768, 65537).Select(static i => (float)i));
        Add("half-integers", Enumerable.Range(-8192, 16384).Select(static i => i + 0.5f));
        Add("grid", Enumerable.Range(-16384, 32769).Select(static i => i / 2048f));
        foreach (float anchor in (float[])[0.5f, 1f, 3f, 10f, 100f, 1000f, 10000f, 43758.547f])
        {
            int bits = BitConverter.SingleToInt32Bits(anchor);
            Add("consecutive-" + anchor.ToString("R", System.Globalization.CultureInfo.InvariantCulture), Enumerable.Range(0, 1024).Select(i => BitConverter.Int32BitsToSingle(bits + i)));
        }

        uint state = 0x9E3779B9u;
        uint Next()
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }

        // Built from raw bits rather than MathF.Pow, whose rounding differs between C runtimes:
        // a uniform exponent in [-10, 17), a uniform 23-bit mantissa and a sign bit.
        Add("log-uniform", Enumerable.Range(0, 32768).Select(_ =>
        {
            uint random = Next();
            uint exponent = (uint)(127 - 10 + (int)((random >> 24) % 27));
            uint bits = ((random & 1) << 31) | (exponent << 23) | ((random >> 1) & 0x7FFFFF);
            return BitConverter.UInt32BitsToSingle(bits);
        }).ToArray());

        sets = described;
        return values.ToArray();
    }

    private static string HlslSource(Type shader) => (string)Property(shader, "HlslSource");

    private static object Property(Type type, string name) =>
        type.GetProperties(StaticProperties).Single(p => p.Name == name || p.Name.EndsWith("." + name, StringComparison.Ordinal)).GetValue(null)
        ?? throw new InvalidOperationException("Missing descriptor property: " + name);

    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static void CreateNew(string directory)
    {
        if (Path.Exists(directory))
        {
            throw new InvalidOperationException("Output already exists: " + directory);
        }

        _ = Directory.CreateDirectory(directory);
    }

    private static void WriteJson(string path, object value)
    {
        File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOptions) + "\n", new UTF8Encoding(false));
    }

    private sealed record Probe(string Name, string Expression, Type Shader, Action<GraphicsDevice, ReadOnlyBuffer<float>, ReadWriteBuffer<float>> Dispatch);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SinProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Sin(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct CosProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Cos(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct TanProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Tan(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct ExpProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Exp(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Exp2Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Exp2(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct LogProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Log(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Log2Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Log2(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SqrtProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Sqrt(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct RsqrtProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Rsqrt(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct RcpProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Rcp(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct DivideProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = 1.0f / input[ThreadIds.X];
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct FracProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Frac(input[ThreadIds.X]);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct PowGammaProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Pow(Hlsl.Abs(input[ThreadIds.X]), 2.2f);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct PowInverseGammaProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Pow(Hlsl.Abs(input[ThreadIds.X]), 0.4545f);
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct AtanProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute() => output[ThreadIds.X] = Hlsl.Atan(input[ThreadIds.X]);
}

// The two- and three-operand probes read x = input[i], y = input[i + 1] and
// z = input[i + 2] (wrapping), so neighbouring inputs of each set are combined.

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Dot2Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 p = new(input[i], input[(i + 1) % input.Length]);
        output[i] = Hlsl.Dot(p, new float2(27.619f, 57.583f));
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Dot3Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 p = new(input[i], input[(i + 1) % input.Length], input[(i + 2) % input.Length]);
        output[i] = Hlsl.Dot(p, new float3(0.2126f, 0.7152f, 0.0722f));
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct MultiplyAddProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = (input[i] * 27.619f) + input[(i + 1) % input.Length];
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct MadProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.Mad(input[i], 27.619f, input[(i + 1) % input.Length]);
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct LerpProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.Lerp(input[i], input[(i + 1) % input.Length], Hlsl.Frac(input[(i + 2) % input.Length]));
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Atan2Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.Atan2(input[i], input[(i + 1) % input.Length]);
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Length2Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.Length(new float2(input[i], input[(i + 1) % input.Length]));
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct Normalize3Probe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 p = new(input[i], input[(i + 1) % input.Length], input[(i + 2) % input.Length]);
        output[i] = Hlsl.Normalize(p).X;
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SmoothStepProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.SmoothStep(0.0f, 1.0f, Hlsl.Frac(input[i]));
    }
}

[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct PowProbe(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Hlsl.Pow(Hlsl.Abs(input[i]), Hlsl.Frac(input[(i + 1) % input.Length]) * 4.0f);
    }
}
