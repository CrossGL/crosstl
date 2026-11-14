using System;

namespace ComputeSharp.Portable.Native;

/// <summary>
/// The guard regions placed before and after every native buffer. Translated shaders
/// index buffers without the bounds checks of Direct3D 12 descriptors, so a write
/// outside the visible region lands in a guard and is reported after the submission.
/// </summary>
internal static class BufferGuards
{
    /// <summary>
    /// Guard size in bytes. 256 is the largest storage/uniform buffer offset alignment
    /// that OpenGL permits and satisfies Metal buffer offset alignment.
    /// </summary>
    public const ulong Bytes = 256;

    /// <summary>Gets the position-dependent guard pattern.</summary>
    public static byte[] Pattern { get; } = CreatePattern();

    /// <summary>Throws when a guard read back from the device differs from <see cref="Pattern"/>.</summary>
    public static void Check(ReadOnlySpan<byte> guard, string region, ulong visibleSize)
    {
        ReadOnlySpan<byte> expected = Pattern;
        for (int i = 0; i < guard.Length; i++)
        {
            if (guard[i] != expected[i])
            {
                int changed = 0;
                for (int j = i; j < guard.Length; j++)
                {
                    if (guard[j] != expected[j])
                    {
                        changed++;
                    }
                }

                throw new BufferGuardException(region, visibleSize, i, changed);
            }
        }
    }

    private static byte[] CreatePattern()
    {
        byte[] pattern = new byte[Bytes];
        for (int i = 0; i < pattern.Length; i++)
        {
            pattern[i] = (byte)(0xA5 ^ (i * 29));
        }

        return pattern;
    }
}

/// <summary>Raised when device work wrote outside the visible region of a buffer.</summary>
internal sealed class BufferGuardException(string region, ulong visibleSize, int firstByte, int changedBytes)
    : InvalidOperationException(
        $"Buffer guard violation: {changedBytes} byte(s) of the {region} guard of a {visibleSize}-byte buffer changed, first at guard byte {firstByte}.")
{
}
