"""Member functions keep their own struct-typed locals, constants and pointer views.

A member function body is lowered to a free function, so everything it names must
stay concrete there: a local object whose struct declares member functions, a
class template argument spelled with the owner's constant static data member,
a local pointer passed to a member function template, an explicit member
template argument whose static constant initializer applies `sizeof` to a struct,
every name of a declarator list such as `Tile<float, 2> P, Q, R;`, and sibling
reference aliases of accessor results used inside call arguments.
"""

import pytest

import crosstl
from tests.test_translator.test_metal_builtin_ownership import _compile

TILE = """#include <metal_stdlib>
using namespace metal;

template <typename T, int N>
struct Tile {
  T v[N];
  void fill(T x) thread {
    for (int i = 0; i < N; i++) v[i] = x;
  }
  template <typename U, int S>
  void load(const device U* src) thread {
    for (int i = 0; i < N; i++) v[i] = T(src[i * S]);
  }
  T sum() const thread {
    T s = T(0);
    for (int i = 0; i < N; i++) s += v[i];
    return s;
  }
};
"""

# The only receivers in `run` are its own local `Tile` objects.
LOCAL_RECEIVERS = TILE + """
struct Block {
  void run(device float* out) thread {
    Tile<float, 2> t;
    t.fill(1.5f);
    out[0] = t.sum();
  }
};

kernel void probe(device float* out [[buffer(0)]]) {
  Block b;
  b.run(out);
}
"""

# `TM` names the owner's constant, a hiding local constant, and a member
# template parameter that hides the owner's constant.
OWNER_CONSTANTS = TILE + """
template <int M>
struct Block {
  static constant constexpr const int TM = M / 2;
  float local_constant() thread {
    constexpr int TM = 3;
    Tile<float, TM> t;
    t.fill(1.0f);
    return t.sum();
  }
  template <int TM>
  float member_parameter() thread {
    Tile<float, TM> t;
    t.v[0] = float(TM);
    return t.v[0];
  }
  float owner_constant() thread {
    Tile<float, TM> t;
    t.fill(1.0f);
    return t.sum();
  }
};

kernel void probe(device float* out [[buffer(0)]]) {
  Block<8> b;
  out[0] = b.local_constant();
  out[1] = b.template member_parameter<5>();
  out[2] = b.owner_constant();
}
"""

# `lanes` is a local pointer; the member template deduces `U` from it.
LOCAL_POINTER = TILE + """
struct Block {
  float acc;
  void run(const device float* values) thread {
    const device float* lanes = values + 2;
    Tile<float, 2> re;
    re.template load<float, 2>(lanes + 0);
    Tile<float, 2> im;
    im.template load<float, 2>(lanes + 1);
    acc = re.sum() - im.sum();
  }
};

kernel void probe(const device float* values [[buffer(0)]],
                  device float* out [[buffer(1)]]) {
  Block b;
  b.acc = 0.0f;
  b.run(values);
  out[0] = b.acc;
}
"""

# `kStrF` folds only once `sizeof(Quad)` folds to 16 ([expr.sizeof]/2), giving
# the explicit member template argument 2 * (4 + 16 / 16 - 2) = 6.
SIZEOF_STRUCT_CONSTANT = TILE + """
struct Quad {
  float a;
  float b;
  float c;
  float d;
};

template <int ld>
struct Block {
  static constant constexpr const short kStr = ld + (16 / sizeof(Quad)) - 2;
  static constant constexpr const short kStrF = kStr * 2;
  float acc;
  void run(const device float* src) thread {
    Tile<float, 2> t;
    t.template load<float, kStrF>(src);
    acc = t.sum();
  }
};

kernel void probe(device float* out [[buffer(0)]],
                  const device float* values [[buffer(1)]]) {
  Block<4> b;
  b.acc = 0.0f;
  b.run(values);
  out[0] = b.acc;
}
"""

# `Q` and `R` are declared by the same declaration as `P`; the comment above it
# must not hide the statement boundary.
DECLARATOR_LIST = TILE + """
template <int M>
struct Block {
  float acc;
  void run() thread {
    // P = 1 ; Q = 2 ; R = (3)
    Tile<float, M> P, Q, R;
    P.fill(1.0f);
    Q.fill(2.0f);
    R.fill(3.0f);
    acc = P.sum() + Q.sum() + R.sum();
  }
};

kernel void probe(device float* out [[buffer(0)]]) {
  Block<2> b;
  b.acc = 0.0f;
  b.run();
  out[0] = b.acc;
}
"""

# Two `thread const auto&` aliases of reference accessors in a const member
# function. Replacing `im` puts the indices `i` and `j` inside the call
# arguments that use `r`; as operands of arithmetic they cannot be bound by
# reference, so `r` keeps its replacement.
REFERENCE_ALIASES = """#include <metal_stdlib>
using namespace metal;

template <typename T, int R, int C>
struct Frags {
  typedef vec<T, 2> frag_type;
  frag_type val_frags[R * C];
  thread frag_type& frag_at(const short i, const short j) thread {
    return val_frags[i * C + j];
  }
  const thread frag_type& frag_at(const short i, const short j) const thread {
    return val_frags[i * C + j];
  }
};

struct Scale {
  float s;
  float apply(float2 x) const thread { return x.x * s + x.y; }
};

template <int M>
struct Block {
  Frags<float, M, M> re;
  Frags<float, M, M> im;
  void store(device float* out, thread const Scale& op) const thread {
    for (short i = 0; i < M; i++) {
      for (short j = 0; j < M; j++) {
        thread const auto& r = re.frag_at(i, j);
        thread const auto& m = im.frag_at(i, j);
        for (short k = 0; k < 2; k++) {
          out[(i * M + j) * 2 + k] = op.apply(float2(r[k], m[k]));
        }
      }
    }
  }
};

kernel void probe(device float* out [[buffer(0)]]) {
  Block<2> b;
  for (short i = 0; i < 4; i++) {
    b.re.val_frags[i] = float2(float(i), float(i) + 0.5f);
    b.im.val_frags[i] = float2(1.0f, 2.0f);
  }
  Scale op;
  op.s = 10.0f;
  b.store(out, op);
}
"""

SOURCES = {
    "declarator-list": DECLARATOR_LIST,
    "reference-aliases": REFERENCE_ALIASES,
    "local-receivers": LOCAL_RECEIVERS,
    "owner-constants": OWNER_CONSTANTS,
    "local-pointer": LOCAL_POINTER,
    "sizeof-struct-constant": SIZEOF_STRUCT_CONSTANT,
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _flat(text):
    return " ".join(text.split())


def test_member_function_lowers_calls_on_its_local_struct_objects(tmp_path):
    crossgl = _flat(_translate(tmp_path, LOCAL_RECEIVERS, "cgl"))
    assert "Tile_float_2__fill(t, " in crossgl
    assert "Tile_float_2__sum(t)" in crossgl
    assert "t.fill(" not in crossgl and "t.sum(" not in crossgl


def test_local_template_arguments_name_the_owner_constant_unless_hidden(tmp_path):
    crossgl = _flat(_translate(tmp_path, OWNER_CONSTANTS, "cgl"))
    # local_constant sees its own TM = 3, member_parameter<5> its parameter,
    # and owner_constant the owner's TM = 8 / 2.
    for extent in (3, 4, 5):
        assert f"Tile_float_{extent} t;" in crossgl
    assert "Tile_float_TM" not in crossgl
    assert "Tile_float_4__fill(t, " in crossgl
    assert "Tile_float_3__fill(t, " in crossgl


def test_member_template_deduces_from_a_local_pointer(tmp_path):
    crossgl = _flat(_translate(tmp_path, LOCAL_POINTER, "cgl"))
    assert "re.load(" not in crossgl and "im.load(" not in crossgl
    assert "Tile_float_2__load__float_2(re, " in crossgl
    assert "Tile_float_2__load__float_2(im, " in crossgl


def test_static_constant_template_argument_folds_sizeof_of_a_struct(tmp_path):
    crossgl = _flat(_translate(tmp_path, SIZEOF_STRUCT_CONSTANT, "cgl"))
    assert "Tile_float_2__load__float_6(t, src);" in crossgl
    assert "t.load(" not in crossgl


def test_every_declarator_of_a_struct_declaration_is_a_receiver(tmp_path):
    crossgl = _flat(_translate(tmp_path, DECLARATOR_LIST, "cgl"))
    for name in ("P", "Q", "R"):
        assert f"Tile_float_2__fill({name}, " in crossgl
        assert f"Tile_float_2__sum({name})" in crossgl
    assert ".fill(" not in crossgl and ".sum(" not in crossgl


def test_sibling_reference_aliases_survive_index_uses_in_call_arguments(tmp_path):
    crossgl = _flat(_translate(tmp_path, REFERENCE_ALIASES, "cgl"))
    assert "frag_at(" not in crossgl
    assert "auto" not in crossgl
    assert "self.re.val_frags[" in crossgl and "self.im.val_frags[" in crossgl


@pytest.mark.parametrize("name", sorted(SOURCES))
@pytest.mark.parametrize("backend", ("metal", "opengl", "directx"))
def test_member_function_locals_compile_natively(tmp_path, name, backend):
    generated = _translate(tmp_path, SOURCES[name], backend)
    assert ".fill(" not in generated and ".load(" not in generated
    _compile(generated, backend, tmp_path)
