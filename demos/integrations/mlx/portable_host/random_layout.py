"""Map logical random output bytes to the unchanged kernel's per-key storage."""

import struct
from dataclasses import dataclass

from crosstl.project.runtime_values import exact_types_within

INT_TYPES = frozenset({int})

MAX_KEY_ELEMENTS = 65535
MAX_WORKGROUP_COUNT = 65535
# The portable per-group invocation limit (D3D12 and OpenGL 4.3 minimums).
MAX_WORKGROUP_HEIGHT = 1024
GUARD_COUNT = 17
MAX_NATIVE_BYTES = 2**31 - 1 - GUARD_COUNT


@dataclass(frozen=True)
class RandomOutputLayout:
    key_count: int
    bytes_per_key: int

    def __post_init__(self):
        if (
            type(self.key_count) is not int
            or type(self.bytes_per_key) is not int
            or self.key_count < 1
            or self.bytes_per_key < 0
        ):
            raise ValueError(
                "Random output requires positive keys and nonnegative bytes"
            )

    @property
    def native_bytes_per_key(self):
        # The source always stores its first complete word, even for short outputs.
        return max(4, self.bytes_per_key) if self.bytes_per_key else 0

    @property
    def native_byte_count(self):
        return self.key_count * self.native_bytes_per_key

    @property
    def logical_byte_count(self):
        return self.key_count * self.bytes_per_key

    @property
    def thread_grid(self):
        """The source grid: one thread per key and per pair of output words."""
        words = (self.bytes_per_key + 3) // 4
        return [self.key_count, (words + 1) // 2, 1]

    @property
    def workgroup_size(self):
        """Full Y groups that keep every portable workgroup count in bounds.

        The kernel has no group-shared state, so any exact factorization of the
        word-pair axis yields the same invocations. The smallest factor keeps
        launches that already fit at one thread per group.
        """
        keys, columns, _ = self.thread_grid
        if not 1 <= keys <= MAX_WORKGROUP_COUNT or columns < 1:
            raise ValueError("Random launch exceeds the portable workgroup limits")
        first = (columns + MAX_WORKGROUP_COUNT - 1) // MAX_WORKGROUP_COUNT
        for height in range(first, MAX_WORKGROUP_HEIGHT + 1):
            if columns % height == 0:
                return [1, height, 1]
        raise ValueError("Random launch exceeds the portable workgroup limits")

    @property
    def workgroup_count(self):
        size = self.workgroup_size
        return [extent // width for extent, width in zip(self.thread_grid, size)]

    @property
    def execution(self):
        return {
            "workgroupCount": self.workgroup_count,
            "workgroupSize": self.workgroup_size,
        }

    def unpack(self, values):
        """Copy logical bytes from signed native carriers without changing their bits."""
        if (
            len(values) != self.native_byte_count
            or not exact_types_within(values, INT_TYPES)
            or (values and not (-128 <= min(values) and max(values) <= 127))
        ):
            raise ValueError(
                "Random readback must contain the exact signed byte storage"
            )
        if not self.bytes_per_key:
            return b""
        # Signed bytes in [-128, 127] pack to the same bits as ``value & 255``.
        raw = struct.pack(f"<{len(values)}b", *values)
        return b"".join(
            raw[start : start + self.bytes_per_key]
            for start in (
                key * self.native_bytes_per_key for key in range(self.key_count)
            )
        )
