"""DXC API bridge for compiler inputs beyond the Windows path limit."""

from __future__ import annotations

import argparse
import ctypes
import ntpath
import os
import shutil
import sys
import uuid
from pathlib import Path

_HRESULT = ctypes.c_int32
_ULONG = ctypes.c_uint32
_VOID = ctypes.c_void_p
_POINTER = ctypes.POINTER(_VOID)
_CALL = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


class _Guid(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _Buffer(ctypes.Structure):
    _fields_ = [("data", _VOID), ("size", ctypes.c_size_t), ("encoding", _ULONG)]


_IUNKNOWN = _Guid.parse("00000000-0000-0000-c000-000000000046")
_INCLUDE_HANDLER = _Guid.parse("7f61fc7d-950d-467f-b3e3-3c02fb49187c")
_COMPILER_CLASS = _Guid.parse("73e22d93-e6ce-47f3-b5bf-f0664f39c1b0")
_COMPILER_INTERFACE = _Guid.parse("228b4687-5a6a-4730-900c-9702b2203f54")
_UTILS_CLASS = _Guid.parse("6245d6af-66e0-48fd-80b4-4d271796748c")
_UTILS_INTERFACE = _Guid.parse("4605c4cb-2019-492a-ada4-65f20bb7d67f")
_RESULT_INTERFACE = _Guid.parse("58346cda-dde7-4497-9461-6f87af5e0659")


def _method(pointer, index, result_type, *argument_types):
    table = ctypes.cast(pointer, ctypes.POINTER(_POINTER)).contents
    return _CALL(result_type, _VOID, *argument_types)(table[index])


def _check(status, operation):
    if status < 0:
        raise RuntimeError(f"{operation} failed (HRESULT 0x{status & 0xffffffff:08x}).")


def _release(pointer):
    if pointer:
        _method(pointer, 2, _ULONG)(pointer)


def _blob_bytes(pointer):
    if not pointer:
        return b""
    size = _method(pointer, 4, ctypes.c_size_t)(pointer)
    address = _method(pointer, 3, _VOID)(pointer)
    return ctypes.string_at(address, size) if size else b""


def _include_path(value):
    if sys.platform != "win32":
        return os.path.normpath(value)
    # DXC can append ../ to an extended path. Normalize before asking Windows
    # to open it, since the extended namespace deliberately skips that step.
    value = value.replace("/", "\\")
    if value[:8].upper() == "\\\\?\\UNC\\":
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\") and value[5:7] == ":\\":
        value = value[4:]
    value = ntpath.abspath(ntpath.normpath(value))
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


class _IncludeHandler(ctypes.Structure):
    _fields_ = [("vtable", _POINTER)]

    def __init__(self, default_handler):
        super().__init__()
        self.references = 1
        self.errors = []
        load_source = _method(default_handler, 3, _HRESULT, ctypes.c_wchar_p, _POINTER)

        def query_interface(this, interface, result):
            result[0] = None
            if bytes(interface.contents) not in (
                bytes(_IUNKNOWN),
                bytes(_INCLUDE_HANDLER),
            ):
                return -2147467262  # E_NOINTERFACE
            result[0] = this
            add_ref(this)
            return 0

        def add_ref(this):
            self.references += 1
            return self.references

        def release(this):
            self.references -= 1
            return self.references

        def load(this, filename, result):
            result[0] = None
            try:
                return load_source(default_handler, _include_path(filename), result)
            except Exception as error:
                self.errors.append(str(error))
                return -2147467259  # E_FAIL; exceptions must not cross the COM ABI.

        self.callbacks = (
            _CALL(_HRESULT, _VOID, ctypes.POINTER(_Guid), _POINTER)(query_interface),
            _CALL(_ULONG, _VOID)(add_ref),
            _CALL(_ULONG, _VOID)(release),
            _CALL(_HRESULT, _VOID, ctypes.c_wchar_p, _POINTER)(load),
        )
        self.table = (_VOID * 4)(
            *(ctypes.cast(callback, _VOID).value for callback in self.callbacks)
        )
        self.vtable = self.table


def compile_source(library_path, source_path, arguments):
    """Return the compiler status, diagnostic bytes and primary DXIL object."""

    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    library = loader(str(library_path))
    create = library.DxcCreateInstance
    create.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(_Guid), _POINTER]
    create.restype = _HRESULT
    owned = []

    def instance(class_id, interface):
        pointer = _VOID()
        _check(
            create(
                ctypes.byref(class_id), ctypes.byref(interface), ctypes.byref(pointer)
            ),
            "DxcCreateInstance",
        )
        owned.append(pointer)
        if not pointer:
            raise RuntimeError("DXC did not return the requested interface.")
        return pointer

    def output(pointer, index, label):
        blob = _VOID()
        _check(
            _method(pointer, index, _HRESULT, _POINTER)(pointer, ctypes.byref(blob)),
            label,
        )
        owned.append(blob)
        return blob

    try:
        utilities = instance(_UTILS_CLASS, _UTILS_INTERFACE)
        compiler = instance(_COMPILER_CLASS, _COMPILER_INTERFACE)
        default = output(utilities, 9, "CreateDefaultIncludeHandler")
        if not default:
            raise RuntimeError("DXC did not return its default include handler.")
        includes = _IncludeHandler(default)
        source_path = Path(source_path)
        source = ctypes.create_string_buffer(source_path.read_bytes())
        buffer = _Buffer(ctypes.cast(source, _VOID), len(source) - 1, 65001)
        values = [str(source_path), *arguments]
        argv = (ctypes.c_wchar_p * len(values))(*values)
        result = _VOID()
        status = _method(
            compiler,
            3,
            _HRESULT,
            ctypes.POINTER(_Buffer),
            ctypes.POINTER(ctypes.c_wchar_p),
            _ULONG,
            _VOID,
            ctypes.POINTER(_Guid),
            _POINTER,
        )(
            compiler,
            ctypes.byref(buffer),
            argv,
            len(values),
            ctypes.byref(includes),
            ctypes.byref(_RESULT_INTERFACE),
            ctypes.byref(result),
        )
        owned.append(result)
        _check(status, "IDxcCompiler3.Compile")
        if not result:
            raise RuntimeError("DXC did not return a compilation result.")
        diagnostics = _blob_bytes(output(result, 5, "GetErrorBuffer")).rstrip(b"\0")
        if includes.errors:
            raise RuntimeError(
                "DXC include handling failed: " + "; ".join(includes.errors)
            )
        status = _HRESULT()
        _check(
            _method(result, 3, _HRESULT, ctypes.POINTER(_HRESULT))(
                result, ctypes.byref(status)
            ),
            "GetStatus",
        )
        if status.value < 0:
            return 1, diagnostics, b""
        module = _blob_bytes(output(result, 4, "GetResult"))
        if not module.startswith(b"DXBC"):
            raise RuntimeError("DXC did not return a DXIL container.")
        return 0, diagnostics, module
    finally:
        for pointer in reversed(owned):
            _release(pointer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("The DXC long-path bridge requires Windows.")
    directory_handle = None
    try:
        executable = shutil.which(args.compiler)
        if executable is None:
            raise RuntimeError("DXC executable was not found.")
        directory = Path(executable).resolve().parent
        directory_handle = os.add_dll_directory(str(directory))
        arguments = (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        status, diagnostics, module = compile_source(
            directory / "dxcompiler.dll", args.source, arguments
        )
        sys.stderr.buffer.write(diagnostics)
        if status == 0:
            args.output.write_bytes(module)
        return status
    except (OSError, RuntimeError, ValueError) as error:
        print(f"DXC: {error}", file=sys.stderr)
        return 1
    finally:
        if directory_handle is not None:
            directory_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
