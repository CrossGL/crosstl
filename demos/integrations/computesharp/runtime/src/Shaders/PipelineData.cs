using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ComputeSharp.Descriptors;
using ComputeSharp.Interop;
using ComputeSharp.Portable.Diagnostics;
using ComputeSharp.Portable.Native;
using ComputeSharp.Portable.Shaders;
using ComputeSharp.Resources.Interop;

namespace ComputeSharp.Shaders.Loading
{
    /// <summary>A compiled translated pipeline and the plan that maps descriptor ranges to target bindings.</summary>
    internal sealed class PipelineData : ReferenceTrackedObject
    {
        private readonly GraphicsDevice device;
        private IntPtr pipeline;

        public PipelineData(GraphicsDevice device, IntPtr pipeline, TranslatedShader shader, DispatchPlan plan, string typeName)
        {
            using ReferenceTracker.Lease _0 = GetReferenceTracker().GetLease();

            this.device = device;
            this.pipeline = pipeline;
            Shader = shader;
            Plan = plan;
            TypeName = typeName;
        }

        public IntPtr Pipeline => this.pipeline;

        public TranslatedShader Shader { get; }

        public DispatchPlan Plan { get; }

        public string TypeName { get; }

        protected override void DangerousOnDispose()
        {
            this.device.Native.DestroyPipeline(this.pipeline);
            this.pipeline = IntPtr.Zero;
        }
    }

    /// <summary>Creates and caches the translated pipeline for a shader type, per device.</summary>
    internal static class PipelineDataLoader<T>
        where T : struct, IComputeShaderDescriptor<T>
    {
        private static readonly ConditionalWeakTable<GraphicsDevice, PipelineData> CachedPipelines = [];

        public static PipelineData GetPipelineData(GraphicsDevice device)
        {
            return CachedPipelines.GetValue(device, CreatePipelineData);
        }

        private static PipelineData CreatePipelineData(GraphicsDevice device)
        {
            string typeName = typeof(T).FullName ?? typeof(T).Name;
            string hlslHash = TranslatedShaderLibrary.HashHlsl(T.HlslSource);
            TranslatedShader shader = device.ShaderLibrary.Find(hlslHash, typeName);
            string source = shader.LoadSource();
            DispatchPlan plan = DispatchPlan.Create(shader, T.ResourceDescriptorRanges.Span, T.IsStaticSamplerRequired, T.ConstantBufferSize);
            IntPtr pipeline = device.Native.CreatePipeline(source, shader.EntryPoint ?? "main", T.ThreadsX, T.ThreadsY, T.ThreadsZ, out string log);

            EvidenceLog.Write("pipeline", json =>
            {
                json.WriteString("shader", typeName);
                json.WriteString("hlslSha256", hlslHash);
                json.WriteString("artifact", shader.ArtifactPath);
                json.WriteString("artifactSha256", shader.ArtifactSha256);
                json.WriteString("entryPoint", shader.EntryPoint);
                json.WriteStartArray("threads");
                json.WriteNumberValue(T.ThreadsX);
                json.WriteNumberValue(T.ThreadsY);
                json.WriteNumberValue(T.ThreadsZ);
                json.WriteEndArray();
                if (log.Length > 0)
                {
                    json.WriteString("compilerLog", log);
                }
            });

            PipelineData pipelineData = new(device, pipeline, shader, plan, typeName);

            CachedPipelines.Add(device, pipelineData);

            device.RegisterPipelineData(pipelineData);

            return pipelineData;
        }
    }
}

namespace ComputeSharp.Portable.Shaders
{
    /// <summary>The target binding of one descriptor range.</summary>
    internal sealed record RangeBinding(int RangeIndex, ResourceDescriptorRangeType Type, string Register, TranslatedBinding Target, TranslatedBinding? Length);

    /// <summary>Maps the upstream root signature (root constants and descriptor ranges) to target slots.</summary>
    internal sealed class DispatchPlan
    {
        private DispatchPlan(TranslatedBinding? constants, RangeBinding[] ranges, TranslatedBinding[] samplers, int constantBufferSize)
        {
            Constants = constants;
            Ranges = ranges;
            Samplers = samplers;
            ConstantBufferSize = constantBufferSize;
        }

        /// <summary>Gets the binding of the <c>b0</c> root constants, if the artifact uses them.</summary>
        public TranslatedBinding? Constants { get; }

        public RangeBinding[] Ranges { get; }

        public TranslatedBinding[] Samplers { get; }

        public int ConstantBufferSize { get; }

        public int MaximumBindingCount => 1 + (Ranges.Length * 2) + Samplers.Length;

        public static DispatchPlan Create(TranslatedShader shader, ReadOnlySpan<ResourceDescriptorRange> ranges, bool isStaticSamplerRequired, int constantBufferSize)
        {
            Dictionary<string, TranslatedBinding> byRegister = new(StringComparer.Ordinal);
            foreach (TranslatedBinding binding in shader.Bindings)
            {
                if (binding.Register is string register && !byRegister.TryAdd(register, binding))
                {
                    throw new InvalidOperationException($"CrossTL reported two bindings for register {register} in {shader.Type}.");
                }
            }

            TranslatedBinding? constants = byRegister.GetValueOrDefault("b0");
            RangeBinding[] result = new RangeBinding[ranges.Length];
            for (int i = 0; i < ranges.Length; i++)
            {
                ResourceDescriptorRange range = ranges[i];
                string prefix = range.Type switch
                {
                    ResourceDescriptorRangeType.ShaderResourceView => "t",
                    ResourceDescriptorRangeType.UnorderedAccessView => "u",
                    ResourceDescriptorRangeType.ConstantBufferView => "b",
                    _ => throw new InvalidOperationException($"Unknown descriptor range type {range.Type}.")
                };
                string register = prefix + range.Register.ToString(System.Globalization.CultureInfo.InvariantCulture);
                if (!byRegister.TryGetValue(register, out TranslatedBinding? target))
                {
                    throw new InvalidOperationException(
                        $"The CrossTL {shader.Type} artifact has no binding for register {register} (descriptor range {i}).");
                }

                TranslatedBinding? length = null;
                foreach (TranslatedBinding binding in shader.Bindings)
                {
                    if (binding.Of is string owner && owner == target.Name && binding.Kind.EndsWith("-length", StringComparison.Ordinal))
                    {
                        length = binding;
                    }
                }

                result[i] = new RangeBinding(i, range.Type, register, target, length);
            }

            List<TranslatedBinding> samplers = [];
            foreach (TranslatedBinding binding in shader.Bindings)
            {
                if (binding.Kind == "sampler")
                {
                    samplers.Add(binding);
                }
            }

            if (samplers.Count > 0 && !isStaticSamplerRequired)
            {
                throw new InvalidOperationException($"The CrossTL {shader.Type} artifact declares a sampler but the shader has no static sampler.");
            }

            return new DispatchPlan(constants, result, [.. samplers], constantBufferSize);
        }
    }

    /// <summary>Collects the constants and resources a generated descriptor loads for one dispatch.</summary>
    internal sealed class DispatchBuilder
    {
        public DispatchBuilder(GraphicsDevice device, int rangeCount, int rangeOffset)
        {
            Device = device;
            Views = new PortableResourceView?[rangeCount];
            RangeOffset = rangeOffset;
        }

        public GraphicsDevice Device { get; }

        public byte[] Constants { get; private set; } = [];

        public PortableResourceView?[] Views { get; }

        public int RangeOffset { get; }

        public void SetConstants(ReadOnlySpan<byte> data)
        {
            if ((data.Length & 3) != 0)
            {
                throw new ArgumentException("Constant data must be a multiple of four bytes.", nameof(data));
            }

            // Pad to the 16-byte register granularity of HLSL constant buffers.
            byte[] padded = new byte[Math.Max(16, (data.Length + 15) & ~15)];
            data.CopyTo(padded);
            Constants = padded;
        }

        public void SetResource(IGraphicsResource resource, uint index)
        {
            ArgumentNullException.ThrowIfNull(resource);

            if (resource is not IPortableReadOnlyResource portable)
            {
                throw new ArgumentException($"The resource {resource} cannot be bound to a translated shader.", nameof(resource));
            }

            int slot = checked((int)index + RangeOffset);
            if ((uint)slot >= (uint)Views.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Resource index {index} is outside the shader's descriptor ranges.");
            }

            PortableResourceView view = portable.ValidateAndGetView(Device, out ReferenceTracker.Lease lease);
            lease.Dispose();
            Views[slot] = view;
        }

        public void SetView(int slot, PortableResourceView view)
        {
            Views[slot] = view;
        }
    }

    /// <summary>The <see cref="IConstantBufferLoader"/> used for translated dispatches.</summary>
    internal readonly struct PortableConstantBufferLoader(DispatchBuilder builder) : IConstantBufferLoader
    {
        public void LoadConstantBuffer(ReadOnlySpan<byte> data) => builder.SetConstants(data);
    }

    /// <summary>The <see cref="IGraphicsResourceLoader"/> used for translated dispatches.</summary>
    internal readonly struct PortableGraphicsResourceLoader(DispatchBuilder builder) : IGraphicsResourceLoader
    {
        public void LoadGraphicsResource(IGraphicsResource resource, uint index) => builder.SetResource(resource, index);
    }

    /// <summary>Records a dispatch with its target bindings and writes its evidence.</summary>
    internal static unsafe class DispatchRecorder
    {
        public static void Record(
            NativeCommandList list,
            ComputeSharp.Shaders.Loading.PipelineData pipelineData,
            DispatchBuilder builder,
            int groupsX,
            int groupsY,
            int groupsZ,
            int dispatchX,
            int dispatchY,
            int dispatchZ)
        {
            DispatchPlan plan = pipelineData.Plan;
            byte[] constants = builder.Constants;
            if (BackendSelection.Fault == "perturb-constants" && constants.Length >= 4)
            {
                constants = (byte[])constants.Clone();
                if (plan.ConstantBufferSize > 12)
                {
                    constants[12] ^= 0x01;
                }
                else
                {
                    uint x = BitConverter.ToUInt32(constants, 0);
                    BitConverter.TryWriteBytes(constants.AsSpan(0, 4), x > 1 ? x - 1 : x + 1);
                }
            }

            NativeBinding[] bindings = new NativeBinding[plan.MaximumBindingCount];
            uint[] lengths = new uint[plan.Ranges.Length];
            int count = 0;

            fixed (byte* constantPointer = constants)
            fixed (uint* lengthPointer = lengths)
            {
                if (plan.Constants is TranslatedBinding constantBinding)
                {
                    bindings[count++] = new NativeBinding
                    {
                        Kind = NativeBindingKind.ConstantBytes,
                        Slot = (uint)constantBinding.Binding,
                        Data = constantPointer,
                        Size = (ulong)constants.Length,
                    };
                }

                foreach (RangeBinding range in plan.Ranges)
                {
                    PortableResourceView view = builder.Views[range.RangeIndex]
                        ?? throw new InvalidOperationException($"{pipelineData.TypeName} did not load a resource for register {range.Register}.");
                    bindings[count++] = new NativeBinding
                    {
                        Kind = KindOf(range, view, pipelineData.TypeName),
                        Slot = (uint)range.Target.Binding,
                        Format = view.Format,
                        Access = range.Type == ResourceDescriptorRangeType.UnorderedAccessView ? CsBackend.UsageShaderRead | CsBackend.UsageShaderWrite : CsBackend.UsageShaderRead,
                        Object = view.Handle,
                        Size = (ulong)view.SizeInBytes,
                    };
                    if (range.Length is TranslatedBinding lengthBinding)
                    {
                        lengths[range.RangeIndex] = (uint)view.Length;
                        bindings[count++] = new NativeBinding
                        {
                            Kind = NativeBindingKind.ConstantBytes,
                            Slot = (uint)lengthBinding.Binding,
                            Data = lengthPointer + range.RangeIndex,
                            Size = sizeof(uint),
                        };
                    }
                }

                foreach (TranslatedBinding sampler in plan.Samplers)
                {
                    bindings[count++] = new NativeBinding
                    {
                        Kind = NativeBindingKind.Sampler,
                        Slot = (uint)sampler.Binding,
                        Object = builder.Device.StaticSampler,
                    };
                }

                if (BackendSelection.Fault == "swap-bindings")
                {
                    SwapFirstPair(bindings, count);
                }

                long sequence = EvidenceLog.NextSequence();
                WriteEvidence(sequence, pipelineData, builder, constants, bindings, count, groupsX, groupsY, groupsZ, dispatchX, dispatchY, dispatchZ);
                list.Dispatch(pipelineData.Pipeline, (uint)groupsX, (uint)groupsY, (uint)groupsZ, bindings, count);
                list.DispatchSequence.Add(sequence);
            }
        }

        private static NativeBindingKind KindOf(RangeBinding range, PortableResourceView view, string typeName)
        {
            NativeBindingKind kind = (range.Type, view.Kind) switch
            {
                (ResourceDescriptorRangeType.ConstantBufferView, PortableResourceKind.ConstantBuffer) => NativeBindingKind.UniformBuffer,
                (ResourceDescriptorRangeType.ShaderResourceView, PortableResourceKind.StructuredBuffer) => NativeBindingKind.Buffer,
                (ResourceDescriptorRangeType.UnorderedAccessView, PortableResourceKind.StructuredBuffer) => NativeBindingKind.Buffer,
                (ResourceDescriptorRangeType.ShaderResourceView, PortableResourceKind.Texture) => NativeBindingKind.Texture,
                (ResourceDescriptorRangeType.ShaderResourceView, PortableResourceKind.ReadWriteTexture) => NativeBindingKind.Texture,
                (ResourceDescriptorRangeType.UnorderedAccessView, PortableResourceKind.ReadWriteTexture) => NativeBindingKind.Image,
                _ => throw new InvalidOperationException($"{typeName} binds {view.Description} to register {range.Register}, which expects a different resource kind.")
            };
            return kind;
        }

        private static void SwapFirstPair(NativeBinding[] bindings, int count)
        {
            for (int i = 0; i < count; i++)
            {
                for (int j = i + 1; j < count; j++)
                {
                    if (bindings[i].Kind == bindings[j].Kind &&
                        bindings[i].Kind is NativeBindingKind.Buffer or NativeBindingKind.Texture or NativeBindingKind.Image &&
                        bindings[i].Object != bindings[j].Object &&
                        bindings[i].Format == bindings[j].Format)
                    {
                        (bindings[i].Object, bindings[j].Object) = (bindings[j].Object, bindings[i].Object);
                        (bindings[i].Size, bindings[j].Size) = (bindings[j].Size, bindings[i].Size);
                        return;
                    }
                }
            }
        }

        private static void WriteEvidence(
            long sequence,
            ComputeSharp.Shaders.Loading.PipelineData pipelineData,
            DispatchBuilder builder,
            byte[] constants,
            NativeBinding[] bindings,
            int count,
            int groupsX,
            int groupsY,
            int groupsZ,
            int dispatchX,
            int dispatchY,
            int dispatchZ)
        {
            if (!EvidenceLog.Enabled)
            {
                return;
            }

            string constantHex = EvidenceLog.Hex(constants);
            EvidenceLog.Write("dispatch", json =>
            {
                json.WriteNumber("sequence", sequence);
                json.WriteString("shader", pipelineData.TypeName);
                json.WriteString("hlslSha256", pipelineData.Shader.HlslSha256);
                json.WriteString("artifactSha256", pipelineData.Shader.ArtifactSha256);
                json.WriteStartArray("dispatch");
                json.WriteNumberValue(dispatchX);
                json.WriteNumberValue(dispatchY);
                json.WriteNumberValue(dispatchZ);
                json.WriteEndArray();
                json.WriteStartArray("groups");
                json.WriteNumberValue(groupsX);
                json.WriteNumberValue(groupsY);
                json.WriteNumberValue(groupsZ);
                json.WriteEndArray();
                json.WriteString("constants", constantHex);
                json.WriteStartArray("bindings");
                foreach (RangeBinding range in pipelineData.Plan.Ranges)
                {
                    PortableResourceView? view = builder.Views[range.RangeIndex];
                    json.WriteStartObject();
                    json.WriteString("register", range.Register);
                    json.WriteString("namespace", range.Target.Namespace);
                    json.WriteNumber("slot", range.Target.Binding);
                    if (view is PortableResourceView resource)
                    {
                        json.WriteString("resource", resource.Description);
                        json.WriteNumber("resourceId", resource.ResourceId);
                    }

                    json.WriteEndObject();
                }

                json.WriteEndArray();
                json.WriteNumber("nativeBindings", count);
            });
        }
    }
}
