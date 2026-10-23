import subprocess
import sys
from pathlib import Path


def test_unterminated_rust_string_does_not_backtrack():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from crosstl.backend.Rust.RustCrossGLCodeGen import RUST_STRING_RE; "
            "literal = chr(34) + (chr(92) + chr(10)) * 64; "
            "assert RUST_STRING_RE.match(literal) is None",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=30,
    )
