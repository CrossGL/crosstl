"""spirv_std's RuntimeArray methods and debug_printf macros in the Rust feature list, and their probes."""

import pytest

from tools.coverage import probes, rustgpu, validators
from tools.coverage.features import spirv_std
from tools.coverage.model import STRING, VOID, Type
from tools.coverage.spellings import SUBSCRIPT

# An excerpt of spirv-std 0.11's runtime_array.rs, whose asm! strings decorate the index non-uniform.
RUNTIME_ARRAY = r"""
/// Dynamically-sized arrays in Rust carry around their length as the second half of a tuple.
#[spirv(runtime_array)]
pub struct RuntimeArray<T> {
    _anti_zst_padding: core::mem::MaybeUninit<u32>,
    _phantom: PhantomData<T>,
}

impl<T> RuntimeArray<T> {
    /// Index the array. Unfortunately, because the length of the runtime array cannot be known,
    /// this function will happily index outside of the bounds of the array, and so is unsafe.
    #[spirv_std_macros::gpu_only]
    pub unsafe fn index(&self, index: usize) -> &T {
        unsafe {
            let mut result = core::mem::MaybeUninit::uninit();
            asm! {
                "OpDecorate %index NonUniform",
                "%result = OpAccessChain typeof*{result} {this} %index",
                "OpStore {result} %result",
            }
            result.assume_init()
        }
    }

    #[spirv_std_macros::gpu_only]
    pub unsafe fn index_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            let mut result = core::mem::MaybeUninit::uninit();
            asm! {
                "%result = OpAccessChain typeof*{result} {this} %index",
            }
            result.assume_init()
        }
    }
}
"""

# An excerpt of spirv-std 0.11's macros/src/lib.rs: two print macros among the crate's other procedural macros.
MACROS = r"""
#[proc_macro]
#[allow(nonstandard_style)]
pub fn Image(item: TokenStream) -> TokenStream {
    let output = syn::parse_macro_input!(item as image::ImageType).into_token_stream();
    output.into()
}

/// Print a formatted string using the debug printf extension.
#[proc_macro]
pub fn debug_printf(input: TokenStream) -> TokenStream {
    debug_printf_inner(syn::parse_macro_input!(input as DebugPrintfInput))
}

/// Similar to `debug_printf` but appends a newline to the format string.
#[proc_macro]
pub fn debug_printfln(input: TokenStream) -> TokenStream {
    let mut input = syn::parse_macro_input!(input as DebugPrintfInput);
    input.format_string.push('\n');
    debug_printf_inner(input)
}
"""


@pytest.fixture(scope="module")
def arrays():
    return {f.name: f for f in spirv_std.runtime_arrays(RUNTIME_ARRAY)}


@pytest.fixture(scope="module")
def prints():
    return {f.name: f for f in spirv_std.debug_printf(MACROS)}


def test_runtime_array_methods_return_references_to_words(arrays):
    assert sorted(arrays) == ["RuntimeArray::index", "RuntimeArray::index_mut"]
    index = arrays["RuntimeArray::index"]
    assert index.signature.result.opaque == "&u32"
    assert index.signature.params[0].type.opaque == spirv_std.RUNTIME_ARRAY
    assert index.signature.params[1].type == Type("usize")
    mutable = arrays["RuntimeArray::index_mut"]
    assert mutable.signature.params[0].type.opaque == spirv_std.MUTABLE_RUNTIME_ARRAY
    for feature in arrays.values():
        assert feature.crossgl == SUBSCRIPT
        assert feature.counterpart.params[1].type == Type("uint")


def test_runtime_array_probes_index_a_storage_buffer_of_words(arrays):
    native = probes.build(arrays["RuntimeArray::index_mut"], "rust").source
    assert "use spirv_std::RuntimeArray;" in native
    assert (
        "#[spirv(storage_buffer, descriptor_set = 0, binding = 8)] u0: &mut RuntimeArray<u32>,"
        in native
    )
    assert "let r: u32 = *(unsafe { u0.index_mut(a1) });" in native
    crossgl = probes.build(arrays["RuntimeArray::index"], "crossgl").source
    assert "buffer uint* u0;" in crossgl
    assert "uint r = u0[a1];" in crossgl


def test_print_macros_take_a_format_string_and_print_as_crossgl_does(prints):
    assert sorted(prints) == ["debug_printf!", "debug_printfln!"]
    for feature in prints.values():
        assert feature.signature.result == VOID
        assert feature.signature.params[0].type == STRING
    assert prints["debug_printf!"].crossgl == "printf"
    # debug_printfln! appends a newline to the format string.
    assert prints["debug_printfln!"].crossgl == 'printf("\\n")'


def test_print_probes_pass_a_literal_and_enable_inline_assembly(prints):
    native = probes.build(prints["debug_printfln!"], "rust").source
    assert native.startswith("#![no_std]\n#![feature(asm_experimental_arch)]\n")
    assert 'unsafe { spirv_std::debug_printfln!("") };' in native
    crossgl = probes.build(prints["debug_printfln!"], "crossgl").source
    assert 'printf("\\n");' in crossgl
    assert 'printf("");' in probes.build(prints["debug_printf!"], "crossgl").source


def test_rust_gpu_enables_non_semantic_instructions():
    assert "SPV_KHR_non_semantic_info" in rustgpu.EXTENSIONS


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
def test_runtime_array_and_print_probes_compile_with_rust_gpu(arrays, prints):
    for feature in [*arrays.values(), *prints.values()]:
        probe = probes.build(feature, "rust")
        verdict = validators.validate("rust", probe.source, entry=probe.entry)
        assert verdict.accepted, (feature.name, verdict.log)
