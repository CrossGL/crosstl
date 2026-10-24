"""Translate whole-array reductions of the 8-byte, 16-bit and 8-bit types.

The pinned ``reduce.metal`` instantiates ``all_reduce`` for every dtype. The
base families install float32, int32, uint32 and Boolean entries and 16-bit
float extrema. Sums and products of int64, uint64, float16, bfloat16 and
complex64, and extrema of int64, uint64 and complex64, keep their element
type as accumulator (``remap_reduce_types`` in the pinned ``reduce.cpp``), so
each addition or product rounds in that type. Extrema of the 8- and 16-bit
integers also keep their type; their sums and products accumulate in 32 bits
(``ACCUMULATORS``), and Boolean sums read their bytes as int8. All and any of
a non-Boolean array test 16-bit floats and float32 by value, so -0.0 is false,
and the other types by their bits as int16, int32 or int64 (``LOGICAL``); they
write Booleans. They are
translated on first use for their exact launch, with the checked in_size and
row_size metadata of ``reduction_cache``: one package per entry, count and
launch.
"""

import hashlib
from pathlib import Path

from crosstl.translator.dispatch_region_identity import translation_implementation_hash
from demos.integrations.mlx.portable_host import reduction_layout
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.portable_host.reduction_cache import (
    SOURCE_SHA256,
    ReductionPackageCache,
)

CACHE = "wide-reductions"
PREFIX = "all_reduce_"
# instantiate_sum_prod and instantiate_min_max in the pinned reduce.metal, for
# the element types no installed whole-array family covers.
NARROW = ("int8", "int16", "uint8", "uint16")
# instantiate_and_or's non-Boolean instantiations: the types remap_reduce_types
# tests all and any in. 1-byte types read as Booleans there.
LOGICAL = ("int16", "int32", "int64", "float16", "bfloat16", "float32")
OPERATIONS = {
    "sum": ("int64", "uint64", "float16", "bfloat16", "complex64", *NARROW),
    "prod": ("int64", "uint64", "float16", "bfloat16", "complex64", *NARROW),
    "min": ("int64", "uint64", "complex64", *NARROW),
    "max": ("int64", "uint64", "complex64", *NARROW),
    "and": LOGICAL,
    "or": LOGICAL,
}
ENTRIES = {
    f"{PREFIX}{operation}{dtype}": dtype
    for operation, dtypes in OPERATIONS.items()
    for dtype in dtypes
}
# The output type of instantiate_sum_prod's narrow integer kernels: uint8 as
# int32_t, uint16 as uint32_t, int8 and int16 as int32_t; and/or write bool.
# Every other entry writes its element type.
ACCUMULATORS = {
    **{
        (operation, dtype): "uint32" if dtype == "uint16" else "int32"
        for operation in ("sum", "prod")
        for dtype in NARROW
    },
    **{(operation, dtype): "bool_" for operation in ("and", "or") for dtype in LOGICAL},
}


def accumulator(operation, dtype):
    """The type a reduction kernel of ``operation`` over ``dtype`` writes."""
    return ACCUMULATORS.get((operation, dtype), dtype)


def signature(entry):
    """The ``(operation, element type)`` of a wide whole-array entry."""
    try:
        dtype = ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown wide whole-array reduction entry") from error
    return entry.removeprefix(PREFIX).removesuffix(dtype), dtype


def dispatch_contract(entry, count, execution):
    """The checked launch of one pass: upstream's plan for ``count`` elements."""
    _, dtype = signature(entry)
    plan = reduction_layout.stage(count, dtype)
    launch = {key: plan[key] for key in ("workgroupCount", "workgroupSize")}
    if not isinstance(execution, dict) or execution != launch:
        raise ValueError("Wide reduction launch differs from the upstream plan")
    return {
        "entry": entry,
        "count": count,
        "rowSize": plan["rowSize"],
        "execution": launch,
    }


class WideReductionCache(ReductionPackageCache):
    """``ReductionPackageCache``'s checked recipe for the wide entries.

    The identity also covers this module, so a changed entry table or plan
    never reuses an older package.
    """

    def _identity(self, entry, count, execution, *, layout=None):
        if layout is not None:
            raise ValueError("Wide whole-array reductions take no row layout")
        contract = dispatch_contract(entry, count, execution)
        self._require_source()
        recipe = hashlib.sha256()
        for module in (__file__, reduction_layout.__file__):
            recipe.update(Path(module).read_bytes() + b"\0")
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "contract": contract,
            "sourceHash": SOURCE_SHA256,
            "implementationHash": translation_implementation_hash(),
            "recipeHash": recipe.hexdigest(),
        }


def cache(root, directory, target):
    """One target's wide whole-array packages under ``directory``."""
    return WideReductionCache(root, directory, target)
