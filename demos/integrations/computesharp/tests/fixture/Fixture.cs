using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DiscoveryFixture
{
    [TestClass]
    public sealed class Cases
    {
        [TestMethod]
        public void Plain() { }

        [DataTestMethod]
        [DataRow(-0.0f, typeof(int))]
        [DataRow(1.5f, typeof(uint))]
        [DataRow(1.5f, typeof(uint))]
        public void Values(float value, Type type) { }

        [Rows]
        public void Expanded(object value) { }

        [DataTestMethod]
        [DataRow(ComputeSharp.Tests.Extensions.Device.Discrete)]
        [DataRow(ComputeSharp.Tests.Extensions.Device.Warp)]
        public void Device(ComputeSharp.Tests.Extensions.Device device)
        {
            if (device == ComputeSharp.Tests.Extensions.Device.Discrete)
                Assert.Inconclusive("Fixture optional hardware is absent");
        }
    }

    [AttributeUsage(AttributeTargets.Method)]
    public sealed class RowsAttribute : TestMethodAttribute, ITestDataSource
    {
        public IEnumerable<object?[]> GetData(MethodInfo method)
        {
            string? mode = Environment.GetEnvironmentVariable("DISCOVERY_FIXTURE_MODE");
            if (mode == "empty") yield break;
            if (mode == "throw") throw new InvalidOperationException("Fixture discovery failure");
            if (mode == "arity") { yield return []; yield break; }
            if (mode == "unsupported") { yield return [Guid.Empty]; yield break; }
            yield return [17];
            if (mode == "duplicate") yield return [17];
        }

        public string GetDisplayName(MethodInfo method, object?[]? data) => method.Name + " expanded";
    }

    internal readonly struct Shader : ComputeSharp.Descriptors.IComputeShaderDescriptor<Shader>
    {
        public static int ThreadsX => 32;
        public static int ThreadsY => 1;
        public static int ThreadsZ => 1;
        public static int ConstantBufferSize => 16;
        public static bool IsStaticSamplerRequired => false;
        public static ReadOnlyMemory<Range> ResourceDescriptorRanges => new Range[] { new(1, 2) };
        static string ComputeSharp.Descriptors.IComputeShaderDescriptor<Shader>.HlslSource => Environment.GetEnvironmentVariable("DISCOVERY_FIXTURE_MODE") == "blank" ? "" :
            "RWStructuredBuffer<uint> output : register(u2);\n[numthreads(32,1,1)]\nvoid Execute(uint3 id : SV_DispatchThreadID) { output[id.x] = id.x; }\n";
    }

    internal readonly record struct Range(int Type, uint Register);
}

namespace ComputeSharp.Descriptors
{
    // A descriptor fixture only; no source generator or native execution is simulated.
    public interface IComputeShaderDescriptor<T> where T : struct { static abstract string HlslSource { get; } }
}

namespace ComputeSharp.Tests.Extensions
{
    public enum Device { Discrete, Warp }
}
