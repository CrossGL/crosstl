"""Map logical random output bytes to the unchanged kernel's per-key storage."""

from dataclasses import dataclass

MAX_KEY_ELEMENTS = 65535
MAX_WORKGROUP_COUNT = 65535
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
    def workgroup_count(self):
        words = (self.bytes_per_key + 3) // 4
        count = [self.key_count, (words + 1) // 2, 1]
        if any(not 1 <= value <= MAX_WORKGROUP_COUNT for value in count):
            raise ValueError("Random launch exceeds the portable workgroup limits")
        return count

    def unpack(self, values):
        """Copy logical bytes from signed native carriers without changing their bits."""
        if len(values) != self.native_byte_count or any(
            type(value) is not int or not -128 <= value <= 127 for value in values
        ):
            raise ValueError(
                "Random readback must contain the exact signed byte storage"
            )
        if not self.bytes_per_key:
            return b""
        raw = bytes(value & 255 for value in values)
        return b"".join(
            raw[start : start + self.bytes_per_key]
            for start in (
                key * self.native_bytes_per_key for key in range(self.key_count)
            )
        )
