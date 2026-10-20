using System;
using System.Buffers.Binary;
using ComputeSharp.Resources;
using ComputeSharp.Resources.Interop;

namespace ComputeSharp.Portable;

/// <summary>
/// Implements the Direct3D 12 unordered-access clears used by <see cref="ComputeContextExtensions"/>:
/// clears write zero to every texel, and fills convert a float value to the texture format with the
/// UNORM rules (saturate, scale by 2^n - 1 and round to nearest even).
/// </summary>
internal static class TextureEncoding
{
    public static unsafe long BufferSize<T>(Resources.Buffer<T> buffer)
        where T : unmanaged
    {
        return (long)buffer.Length * sizeof(T);
    }

    public static void Clear(in ComputeContext context, PortableResourceView view, int width, int height, int depth)
    {
        int texelSize = HostTexelSize(view.Format);
        long rowPitch = (long)width * texelSize;
        long slicePitch = rowPitch * height;
        byte[] data = new byte[checked((int)(slicePitch * depth))];

        context.UploadTexture(view, width, height, depth, data, rowPitch, slicePitch);
    }

    public static void Fill(in ComputeContext context, PortableResourceView view, int width, int height, int depth, Float4 value)
    {
        int texelSize = HostTexelSize(view.Format);
        Span<byte> texel = stackalloc byte[16];
        EncodeNormalized(view.Format, value, texel);

        long rowPitch = (long)width * texelSize;
        long slicePitch = rowPitch * height;
        byte[] data = new byte[checked((int)(slicePitch * depth))];
        for (int offset = 0; offset < data.Length; offset += texelSize)
        {
            texel[..texelSize].CopyTo(data.AsSpan(offset));
        }

        context.UploadTexture(view, width, height, depth, data, rowPitch, slicePitch);
    }

    public static int HostTexelSize(uint format)
    {
        return format switch
        {
            2 or 3 or 4 => 16,
            6 or 7 or 8 => 12,
            11 or 16 or 17 or 18 => 8,
            28 or 35 or 41 or 42 or 43 or 87 => 4,
            49 or 56 => 2,
            61 => 1,
            _ => throw new ArgumentException($"Unsupported texture format {format}.", nameof(format))
        };
    }

    private static void EncodeNormalized(uint format, Float4 value, Span<byte> texel)
    {
        switch (format)
        {
            case 61:
                texel[0] = Unorm8(value.X);
                break;
            case 49:
                texel[0] = Unorm8(value.X);
                texel[1] = Unorm8(value.Y);
                break;
            case 28:
                texel[0] = Unorm8(value.X);
                texel[1] = Unorm8(value.Y);
                texel[2] = Unorm8(value.Z);
                texel[3] = Unorm8(value.W);
                break;
            case 87:
                texel[0] = Unorm8(value.Z);
                texel[1] = Unorm8(value.Y);
                texel[2] = Unorm8(value.X);
                texel[3] = Unorm8(value.W);
                break;
            case 56:
                BinaryPrimitives.WriteUInt16LittleEndian(texel, Unorm16(value.X));
                break;
            case 35:
                BinaryPrimitives.WriteUInt16LittleEndian(texel, Unorm16(value.X));
                BinaryPrimitives.WriteUInt16LittleEndian(texel[2..], Unorm16(value.Y));
                break;
            case 11:
                BinaryPrimitives.WriteUInt16LittleEndian(texel, Unorm16(value.X));
                BinaryPrimitives.WriteUInt16LittleEndian(texel[2..], Unorm16(value.Y));
                BinaryPrimitives.WriteUInt16LittleEndian(texel[4..], Unorm16(value.Z));
                BinaryPrimitives.WriteUInt16LittleEndian(texel[6..], Unorm16(value.W));
                break;
            case 41:
                BinaryPrimitives.WriteSingleLittleEndian(texel, value.X);
                break;
            case 16:
                BinaryPrimitives.WriteSingleLittleEndian(texel, value.X);
                BinaryPrimitives.WriteSingleLittleEndian(texel[4..], value.Y);
                break;
            case 2:
                BinaryPrimitives.WriteSingleLittleEndian(texel, value.X);
                BinaryPrimitives.WriteSingleLittleEndian(texel[4..], value.Y);
                BinaryPrimitives.WriteSingleLittleEndian(texel[8..], value.Z);
                BinaryPrimitives.WriteSingleLittleEndian(texel[12..], value.W);
                break;
            default:
                throw new NotSupportedException($"Filling textures with format {format} is not supported.");
        }
    }

    private static byte Unorm8(float value)
    {
        float clamped = float.IsNaN(value) ? 0 : Math.Clamp(value, 0f, 1f);

        return (byte)Math.Round(clamped * 255.0, MidpointRounding.ToEven);
    }

    private static ushort Unorm16(float value)
    {
        float clamped = float.IsNaN(value) ? 0 : Math.Clamp(value, 0f, 1f);

        return (ushort)Math.Round(clamped * 65535.0, MidpointRounding.ToEven);
    }
}
