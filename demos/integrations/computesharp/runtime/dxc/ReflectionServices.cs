using System;
using System.Text.RegularExpressions;
using ComputeSharp.Descriptors;

namespace ComputeSharp.Interop;

/// <summary>
/// Shader reflection for translated backends. The HLSL source is the exact text produced by
/// the source generator. Double-precision use is derived from that source (any HLSL double
/// scalar, vector or matrix type), since the DXIL requirement flags are only available from
/// DXC on Windows. The DXIL instruction statistics are reported as zero.
/// </summary>
public static partial class ReflectionServices
{
    private static readonly Regex DoubleType = DoubleTypeRegex();

    /// <summary>Gets the shader info associated with a given compute shader.</summary>
    public static ShaderInfo GetShaderInfo<T>()
        where T : struct, IComputeShader, IComputeShaderDescriptor<T>
    {
        return GetNonGenericShaderInfo<T>();
    }

    /// <summary>Gets the shader info associated with a given pixel shader.</summary>
    public static ShaderInfo GetShaderInfo<T, TPixel>()
        where T : struct, IComputeShader<TPixel>, IComputeShaderDescriptor<T>
        where TPixel : unmanaged
    {
        return GetNonGenericShaderInfo<T>();
    }

    private static ShaderInfo GetNonGenericShaderInfo<T>()
        where T : struct, IComputeShaderDescriptor<T>
    {
        string source = T.HlslSource;

        return new(
            CompilerVersion: "CrossTL translated backend (DXIL statistics unavailable)",
            HlslSource: source,
            HlslBytecode: T.HlslBytecode,
            ConstantBufferCount: 0,
            BoundResourceCount: 0,
            InstructionCount: 0,
            TemporaryRegisterCount: 0,
            TemporaryArrayCount: 0,
            ConstantDefineCount: 0,
            DeclarationCount: 0,
            TextureNormalInstructions: 0,
            TextureLoadInstructionCount: 0,
            TextureStoreInstructionCount: 0,
            FloatInstructionCount: 0,
            IntInstructionCount: 0,
            UIntInstructionCount: 0,
            StaticFlowControlInstructionCount: 0,
            DynamicFlowControlInstructionCount: 0,
            EmitInstructionCount: 0,
            BarrierInstructionCount: 0,
            InterlockedInstructionCount: 0,
            BitwiseInstructionCount: 0,
            MovcInstructionCount: 0,
            MovInstructionCount: 0,
            InterfaceSlotCount: 0,
            RequiresDoublePrecisionSupport: DoubleType.IsMatch(StripComments(source)));
    }

    private static string StripComments(string source)
    {
        return Regex.Replace(source, @"//[^\n]*|/\*.*?\*/", string.Empty, RegexOptions.Singleline);
    }

    [GeneratedRegex(@"\bdouble(?:[1-4](?:x[1-4])?)?\b")]
    private static partial Regex DoubleTypeRegex();
}
