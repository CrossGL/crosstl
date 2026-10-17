using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using ComputeSharp.Portable.Native;

namespace ComputeSharp
{
    /// <summary>A <see langword="struct"/> that contains info on a given translated device.</summary>
    public readonly struct GraphicsDeviceInfo
    {
        internal GraphicsDeviceInfo(GraphicsDevice device)
        {
            Luid = device.Luid;
            Name = device.Name;
            DedicatedMemorySize = device.DedicatedMemorySize;
            SharedMemorySize = device.SharedMemorySize;
            IsHardwareAccelerated = device.IsHardwareAccelerated;
        }

        /// <inheritdoc cref="GraphicsDevice.Luid"/>
        public Luid Luid { get; }

        /// <inheritdoc cref="GraphicsDevice.Name"/>
        public string Name { get; }

        /// <inheritdoc cref="GraphicsDevice.DedicatedMemorySize"/>
        public nuint DedicatedMemorySize { get; }

        /// <inheritdoc cref="GraphicsDevice.SharedMemorySize"/>
        public nuint SharedMemorySize { get; }

        /// <inheritdoc cref="GraphicsDevice.IsHardwareAccelerated"/>
        public bool IsHardwareAccelerated { get; }
    }

    /// <summary>A locally unique identifier for a graphics device.</summary>
    public readonly struct Luid : IEquatable<Luid>, ISpanFormattable
    {
        private readonly uint lowPart;
        private readonly int highPart;

        private Luid(uint lowPart, int highPart)
        {
            this.lowPart = lowPart;
            this.highPart = highPart;
        }

        internal static Luid FromValue(long value)
        {
            return new((uint)(value & 0xFFFFFFFF), (int)(value >> 32));
        }

        /// <inheritdoc/>
        public bool Equals(Luid other)
        {
            return this.lowPart == other.lowPart && this.highPart == other.highPart;
        }

        /// <inheritdoc/>
        public override bool Equals(object? other)
        {
            return other is Luid luid && Equals(luid);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return HashCode.Combine(this.lowPart, this.highPart);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return ((((long)this.highPart) << 32) | this.lowPart).ToString();
        }

        /// <inheritdoc/>
        public string ToString(string? format, IFormatProvider? formatProvider)
        {
            return ((((long)this.highPart) << 32) | this.lowPart).ToString(format, formatProvider);
        }

        /// <inheritdoc/>
        public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider)
        {
            return ((((long)this.highPart) << 32) | this.lowPart).TryFormat(destination, out charsWritten, format, provider);
        }

        /// <summary>Checks whether two <see cref="Luid"/> values are the same.</summary>
        public static bool operator ==(Luid a, Luid b) => a.Equals(b);

        /// <summary>Checks whether two <see cref="Luid"/> values are different.</summary>
        public static bool operator !=(Luid a, Luid b) => !a.Equals(b);
    }

    /// <summary>Indicates the reason for a device being lost (values match the Direct3D 12 HRESULT codes).</summary>
    public enum DeviceLostReason
    {
        /// <summary>The device stopped responding.</summary>
        DeviceHung = unchecked((int)0x887A0006),

        /// <summary>The device was removed.</summary>
        DeviceRemoved = unchecked((int)0x887A0005),

        /// <summary>The device was reset.</summary>
        DeviceReset = unchecked((int)0x887A0007),

        /// <summary>The driver encountered an internal error.</summary>
        DriverInternalError = unchecked((int)0x887A0020),

        /// <summary>The application issued an invalid call.</summary>
        InvalidCall = unchecked((int)0x887A0001),

        /// <summary>Access to the device was denied.</summary>
        AccessDenied = unchecked((int)0x887A002B),

        /// <summary>An unspecified error.</summary>
        UnspecifiedError = unchecked((int)0x80004005)
    }
}

namespace ComputeSharp.Interop
{
    /// <summary>The type of a resource descriptor range (values match D3D12_DESCRIPTOR_RANGE_TYPE).</summary>
    public enum ResourceDescriptorRangeType
    {
        /// <summary>A shader resource view (SRV), bound to a <c>t</c> register.</summary>
        ShaderResourceView = 0,

        /// <summary>An unordered access view (UAV), bound to a <c>u</c> register.</summary>
        UnorderedAccessView = 1,

        /// <summary>A constant buffer view (CBV), bound to a <c>b</c> register.</summary>
        ConstantBufferView = 2
    }
}

namespace ComputeSharp.Graphics.Helpers
{
    /// <summary>Creates and enumerates the single translated device of the selected backend.</summary>
    internal static class DeviceHelper
    {
        private static readonly Lock Gate = new();
        private static GraphicsDevice? defaultDevice;

        public static GraphicsDevice GetDefaultDeviceFromCacheOrCreateInstance()
        {
            lock (Gate)
            {
                return defaultDevice ??= new GraphicsDevice(NativeDevice.Create());
            }
        }

        public static void NotifyDisposedDevice(GraphicsDevice device)
        {
            lock (Gate)
            {
                if (ReferenceEquals(defaultDevice, device))
                {
                    defaultDevice = null;
                }
            }
        }

        /// <summary>Derives a stable, nonzero identifier from the device identity.</summary>
        public static long StableLuid(string name, string driver)
        {
            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes($"{BackendSelection.Backend}\n{name}\n{driver}"));
            long value = BitConverter.ToInt64(hash, 0) & long.MaxValue;
            return value == 0 ? 1 : value;
        }

        /// <summary>An enumerable over the available devices, optionally filtered.</summary>
        public sealed class DeviceQuery(Predicate<GraphicsDeviceInfo>? predicate) : IEnumerable<GraphicsDevice>
        {
            public IEnumerator<GraphicsDevice> GetEnumerator()
            {
                GraphicsDevice device = GetDefaultDeviceFromCacheOrCreateInstance();
                if (predicate is null || predicate(new GraphicsDeviceInfo(device)))
                {
                    yield return device;
                }
            }

            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }
    }

    /// <summary>Maps resource element types to DXGI format values, shared with the native backends.</summary>
    internal static class DXGIFormatHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint GetForType<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(int)) return 43;
            if (typeof(T) == typeof(Int2)) return 18;
            if (typeof(T) == typeof(Int3)) return 8;
            if (typeof(T) == typeof(Int4)) return 4;
            if (typeof(T) == typeof(uint)) return 42;
            if (typeof(T) == typeof(UInt2)) return 17;
            if (typeof(T) == typeof(UInt3)) return 7;
            if (typeof(T) == typeof(UInt4)) return 3;
            if (typeof(T) == typeof(float)) return 41;
            if (typeof(T) == typeof(Float2)) return 16;
            if (typeof(T) == typeof(Float3)) return 6;
            if (typeof(T) == typeof(Float4)) return 2;
            if (typeof(T) == typeof(Vector2)) return 16;
            if (typeof(T) == typeof(Vector3)) return 6;
            if (typeof(T) == typeof(Vector4)) return 2;
            if (typeof(T) == typeof(Bgra32)) return 87;
            if (typeof(T) == typeof(Rgba32)) return 28;
            if (typeof(T) == typeof(Rgba64)) return 11;
            if (typeof(T) == typeof(R8)) return 61;
            if (typeof(T) == typeof(R16)) return 56;
            if (typeof(T) == typeof(Rg16)) return 49;
            if (typeof(T) == typeof(Rg32)) return 35;

            throw new ArgumentException($"The type {typeof(T)} is not a supported texture element type.", nameof(T));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNormalizedType<T>()
            where T : unmanaged
        {
            if (typeof(T) == typeof(Bgra32) ||
                typeof(T) == typeof(Rgba32) ||
                typeof(T) == typeof(Rgba64) ||
                typeof(T) == typeof(R8) ||
                typeof(T) == typeof(R16) ||
                typeof(T) == typeof(Rg16) ||
                typeof(T) == typeof(Rg32))
            {
                return true;
            }

            _ = GetForType<T>();

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float4 ExtendToNormalizedValue<T>(T value)
            where T : unmanaged
        {
            if (typeof(T) == typeof(Float4))
            {
                return Unsafe.As<T, Float4>(ref value);
            }

            Float4 result = default;

            Unsafe.As<Float4, T>(ref result) = value;

            return result;
        }
    }
}
