// A top-level program with one compute shader descriptor, like the upstream
// GlobalStatements project. PROGRAM_FIXTURE_MODE=fail makes its check throw.
string program = Environment.GetEnvironmentVariable("CROSSTL_COMPUTESHARP_PROGRAM") ?? "";
Console.WriteLine("program " + program);
if (Environment.GetEnvironmentVariable("PROGRAM_FIXTURE_MODE") == "fail")
{
    throw new Exception("The processed buffer does not match the expected results.");
}

Console.WriteLine("Test passed successfully!");

internal readonly struct Shader : ComputeSharp.Descriptors.IComputeShaderDescriptor<Shader>
{
    public static int ThreadsX => 64;
    public static int ThreadsY => 1;
    public static int ThreadsZ => 1;
    public static int ConstantBufferSize => 16;
    public static bool IsStaticSamplerRequired => false;
    public static ReadOnlyMemory<Range> ResourceDescriptorRanges => new Range[] { new(1, 0) };
    static string ComputeSharp.Descriptors.IComputeShaderDescriptor<Shader>.HlslSource =>
        "RWStructuredBuffer<float> values : register(u0);\n[numthreads(64,1,1)]\nvoid Execute(uint3 id : SV_DispatchThreadID) { values[id.x] += 0.5; }\n";
}

internal readonly record struct Range(int Type, uint Register);

namespace ComputeSharp.Descriptors
{
    // A descriptor fixture only; no source generator or native execution is simulated.
    public interface IComputeShaderDescriptor<T> where T : struct { static abstract string HlslSource { get; } }
}
