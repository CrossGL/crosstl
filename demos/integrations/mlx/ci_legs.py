"""Find the heavy MLX jobs and matrix groups that a change can affect.

Every heavy job of the MLX workflow names the files it runs under
``demos/integrations/mlx``. The planner follows those names to every MLX file a
job can read: Python imports, the package ``__init__`` and ``conftest`` modules
that the import system and pytest load, and any MLX path that a module or a
data file spells in a string. A changed MLX file selects the legs whose closure
contains it. A changed file that no leg reaches selects every leg, so a
dependency the scan cannot see costs runner time instead of skipping
validation.

The current-tree parity jobs run one matrix leg per kernel group. Their steps
select a group with ``matrix.group == '...'`` or ``matrix.family == '...'``, so
each group is a leg of its own that reads its own steps and the steps every
group shares.
"""

import ast
import json
import posixpath
import re
from fnmatch import fnmatchcase

from demos.integrations.mlx.ci_conditions import JOBS_OUTPUT

MLX = "demos/integrations/mlx"
MODULE = "demos.integrations.mlx"
# A heavy job runs when the plan's ``jobs`` output names it (``job_condition``).
# Before that output existed, the condition was ``HEAVY``, which a workflow
# change is compared with.
HEAVY = "needs.ci-plan.outputs.heavy == 'true'"
EVERY_LEG = None
# Python modules join a directory with further names before they open a file,
# so a one-part directory such as ``contracts`` names nothing by itself. A
# deeper one, such as ``tests/kernels``, is usually a whole directory handed to
# a tool, so it names every file below it.
DEEP = "deep"

_MATRIX_TEST = re.compile(r"matrix\.([A-Za-z_][\w-]*)\s*==\s*'([^']*)'")
_MATRIX_ANY = re.compile(r"matrix\.")
_TEXT_TOKEN = re.compile(r"[^\s\"'`=,;:()\[\]{}<>|&$\\]+")
_FORMAT_FIELD = re.compile(r"\{[^{}]*\}|%[-#0 +]*\d*(?:\.\d+)?[a-zA-Z]")
_WORD = re.compile(r"[A-Za-z0-9]")
_PATH_WORD = re.compile(r"[\w./-]+")
# A fixed piece of a formatted path carries a separator; one without, such as
# a word in a message, says nothing about the path. A bare extension such as
# ``.json`` completes a name spelled elsewhere, so it says nothing either.
_SEPARATOR = re.compile(r"[/._-]")
_SPACE = re.compile(r"\s")
_EXTENSION = re.compile(r"\.[A-Za-z0-9]+")
_GLOB_CHARACTER = re.compile(r"([*?\[])")
_GLOB = re.compile(r"[*?\[]")
# Calls that list a directory, which can read files that no string names.
_ENUMERATORS = frozenset({"glob", "iterdir", "listdir", "rglob", "scandir", "walk"})
# Prose documents name files without reading them.
_PROSE = (".md", ".rst", ".txt")
_PREFIX_PARTS = frozenset({"", ".", "demos", "integrations", "mlx"})


class PathIndex:
    """Resolve the strings that can name MLX files to those files.

    Paths are relative to the MLX demo. A string names a file when it spells
    the file's path, a trailing part of it, its name without one or more
    extensions, or a directory that contains it, optionally after the
    repository prefix or as a dotted module name.
    """

    def __init__(self, paths):
        self.paths = tuple(sorted(set(paths)))
        self.exact = frozenset(self.paths)
        self._joined = "\n".join(self.paths)
        self.extensions = frozenset(
            path.rsplit(".", 1)[1]
            for path in self.paths
            if "." in path.rsplit("/", 1)[-1]
        )
        files, folders = {}, {}
        for path in self.paths:
            parts = path.split("/")
            name = parts[-1]
            stems = [name[: dot.start()] for dot in re.finditer(r"\.", name)]
            for start in range(len(parts)):
                folder = parts[start:-1]
                for stem in (name, *filter(None, stems)):
                    files.setdefault("/".join((*folder, stem)), set()).add(path)
                for end in range(start + 1, len(parts)):
                    folders.setdefault("/".join(parts[start:end]), set()).add(path)
        self._files = {key: frozenset(value) for key, value in files.items()}
        self._folders = {key: frozenset(value) for key, value in folders.items()}

    def under(self, folder):
        """Return every file below an MLX directory, or every file for the root."""
        folder = folder.strip("/")
        if not folder:
            return set(self.paths)
        return {path for path in self.paths if path.startswith(folder + "/")}

    def folder(self, name):
        """Return the files below every MLX directory that ``name`` spells.

        ``name`` may be a trailing part of a directory path, so ``kernels``
        names the files below ``tests/kernels``.
        """
        return set(self._folders.get(name.strip("/"), ()))

    def module(self, dotted):
        """Return the files that importing a dotted module name executes."""
        if dotted == MODULE:
            return {"__init__.py"} & self.exact
        if not dotted.startswith(MODULE + "."):
            return set()
        relative = dotted[len(MODULE) + 1 :].replace(".", "/")
        return {f"{relative}.py", f"{relative}/__init__.py"} & self.exact

    def complete(self, text, folders=False):
        """Return the files that a string, or any path-like word in it, names.

        A string names a directory, and so every file below it, only when
        ``folders`` is set: a module that only joins a directory with other
        names reads the files those names spell, while a module that lists
        directories or a command that receives one can read all of them.
        """
        found = self.lookup(text, folders)
        for token in _TEXT_TOKEN.findall(text):
            if token != text and ("/" in token or "." in token):
                found |= self.lookup(token, folders)
        return found

    def template(self, pieces):
        """Return the files a formatted string can complete to.

        Only a template that spells a path, with a ``/`` or a file extension
        that a demo file has, names files; identifiers and messages do not.
        A path that contains the formatted string contains its fixed pieces in
        order, so ``f"contracts/{family}.current-tree.translation.json"`` names
        the family contracts and no other file below ``contracts``. A piece
        that spells the demo's own prefix starts the path after it. A template
        without a piece that says something about the path, such as
        ``f"{name}.json"``, names nothing.
        """
        pieces = [piece for piece in pieces if piece]
        if not any("/" in piece or self._extension(piece) for piece in pieces):
            return set()
        fixed = []
        for piece in pieces:
            relative = _relative(piece, strip=False)
            if relative is None:
                return set(self.paths)
            if relative != piece:
                fixed = []
            fixed.append(relative)
        if not any(map(_informative, fixed)):
            return set()
        pattern = "*".join(_GLOB_CHARACTER.sub(r"[\1]", piece) for piece in fixed)
        return {path for path in self.paths if fnmatchcase(path, f"*{pattern}*")}

    def _extension(self, piece):
        return any(
            match.group(1) in self.extensions
            for match in re.finditer(r"\.([A-Za-z0-9]+)(?![A-Za-z0-9])", piece)
        )

    def lookup(self, token, folders=False):
        """Return the files that one path-like word names."""
        relative = _relative(token)
        if relative is None:
            return set(self.paths) if folders else set()
        if relative in _PREFIX_PARTS:
            return set()
        if _GLOB.search(relative):
            if folders is not True:
                return set()
            return {
                path
                for path in self.paths
                if fnmatchcase(path, relative)
                or fnmatchcase(path, f"*/{relative}")
                or fnmatchcase(path.rsplit("/", 1)[-1], relative)
            }
        found = set(self._files.get(relative, ()))
        if folders is True or (folders == DEEP and "/" in relative):
            found |= self._folders.get(relative, frozenset())
        return found


def _informative(piece):
    return (
        len(_WORD.findall(piece)) >= 3
        and _SEPARATOR.search(piece) is not None
        and _SPACE.search(piece) is None
        and _EXTENSION.fullmatch(piece) is None
    )


def _relative(token, strip=True):
    """Spell a path or module string relative to the MLX demo.

    Return ``None`` for a string that names the whole demo directory.
    """
    token = token.replace("\\", "/")
    for prefix, separator in ((MLX, "/"), (MODULE, ".")):
        index = token.find(prefix)
        if index < 0:
            continue
        rest = token[index + len(prefix) :]
        if not rest.strip("/."):
            return None
        if rest.startswith(separator):
            token = rest[1:]
            if separator == ".":
                token = token.replace(".", "/")
            break
    if not strip:
        return token
    while token.startswith("./"):
        token = token[2:]
    return token.strip("/")


def python_references(path, source, index):
    """Return the MLX files a Python module imports, lists or names."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return data_references(source, index)
    package = _package(path)
    assigned, listed, templates, held = {}, [], [], set()
    strings, modules, joins = [], [], []
    for node in ast.walk(tree):
        parts = _join_parts(node)
        if parts is not None:
            joins.append(parts)
        if id(node) in held:
            continue
        if isinstance(node, _SCOPES):
            docstring = _docstring(node)
            if docstring is not None:
                held.add(id(docstring))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                strings.append(node.value)
        elif isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod))
        ):
            templates.append(_template_pieces(node, held))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "format":
                templates.append(_template_pieces(node.func.value, held))
            elif node.func.attr in _ENUMERATORS:
                listed.append(_listed_directory(node))
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_module(package, node.module, node.level)
            modules.append(base)
            modules.extend(f"{base}.{alias.name}" for alias in node.names)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, node.value)
    found = set()
    for module in modules:
        found |= _imported(index, module)
    for base in filter(None, listed):
        folder = _spell(base, assigned, f"{MLX}/{path}")
        if folder is not None:
            relative = _relative(folder)
            if relative is None:
                found |= set(index.paths)
            elif folder.startswith(MLX + "/"):
                found |= index.under(relative)
    for parts in joins:
        found |= _joined_folder(parts, assigned, f"{MLX}/{path}", index)
    for pieces in templates:
        found |= index.template(pieces)
    for text in set(strings):
        pieces = _FORMAT_FIELD.split(text)
        if len(pieces) > 1:
            found |= index.template(pieces)
        else:
            found |= index.complete(text, folders=DEEP)
    return found


_SCOPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring(node):
    """Return a scope's docstring node, which describes files without reading them."""
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value
    return None


def _template_pieces(node, held):
    """Return the literal pieces of a formatted or concatenated string.

    The nodes it reads are added to ``held`` so they are not read again as
    whole strings.
    """
    held.add(id(node))
    if isinstance(node, ast.Constant):
        return _FORMAT_FIELD.split(node.value) if isinstance(node.value, str) else []
    if isinstance(node, ast.JoinedStr):
        return [
            piece for value in node.values for piece in _template_pieces(value, held)
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _template_pieces(node.left, held) + _template_pieces(node.right, held)
    return []


def _listed_directory(call):
    """Return the expression for the directory a listing call reads."""
    receiver = call.func.value
    if isinstance(receiver, ast.Name) and receiver.id in {"glob", "os"}:
        return call.args[0] if call.args else None
    return receiver


def _spell(node, assigned, module, depth=0):
    """Spell a path expression as a repository path, or ``None`` if unknown.

    The expression may join constants, module names bound to path expressions,
    ``__file__`` and its parents, ``Path(...)`` and ``os.path.join``.
    """
    if depth > 8:
        return None
    spell = lambda item: _spell(item, assigned, module, depth + 1)  # noqa: E731
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return module
        return spell(assigned[node.id]) if node.id in assigned else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = spell(node.left), spell(node.right)
        return None if left is None or right is None else _join(left, right)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        inner = spell(node.value)
        return None if inner is None else posixpath.dirname(inner)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "parents" and isinstance(node.slice, ast.Constant):
            inner = spell(node.value.value)
            if inner is None or not isinstance(node.slice.value, int):
                return None
            for _ in range(node.slice.value + 1):
                inner = posixpath.dirname(inner)
            return inner
    if isinstance(node, ast.Call):
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else getattr(function, "id", "")
        )
        if name in {"resolve", "absolute", "expanduser"} and isinstance(
            function, ast.Attribute
        ):
            return spell(function.value)
        if name in {"Path", "PurePath", "PurePosixPath", "join", "joinpath"}:
            parts = list(node.args)
            if name == "joinpath" and isinstance(function, ast.Attribute):
                parts.insert(0, function.value)
            spelled = [spell(part) for part in parts]
            if not spelled or None in spelled:
                return None
            joined = spelled[0]
            for part in spelled[1:]:
                joined = _join(joined, part)
            return joined
    return None


_JOIN_CALLS = frozenset({"Path", "PurePath", "PurePosixPath", "join", "joinpath"})


def _join_parts(node):
    """Return the parts a path join expression joins, in order, or ``None``.

    A join is ``left / right`` or a ``Path``, ``os.path.join`` or ``joinpath``
    call with more than one part.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return (_join_parts(node.left) or [node.left]) + [node.right]
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    name = (
        function.attr
        if isinstance(function, ast.Attribute)
        else getattr(function, "id", "")
    )
    if name not in _JOIN_CALLS:
        return None
    parts = list(node.args)
    if name == "joinpath" and isinstance(function, ast.Attribute):
        parts.insert(0, function.value)
    return parts if len(parts) > 1 else None


def _joined_folder(parts, assigned, module, index):
    """Return the files below an MLX folder that a join can complete to.

    A join that reaches an MLX folder and continues with a part no string
    spells, such as a variable or a template that names no file, can reach any
    file below that folder: ``CONTRACTS / f"{name}.json"`` reads a contract
    whose name only the caller knows. The folder is either the whole spelled
    prefix, or, after a part such as a ``root`` parameter that cannot be
    spelled, the constant parts that follow it when they name an MLX folder,
    as ``"contracts"`` does in ``root / "contracts" / name``.
    """
    found, prefix, tail = set(), "", []
    for part in parts:
        spelled = _spell(part, assigned, module)
        if spelled is not None:
            prefix = None if prefix is None else _join(prefix, spelled)
            tail.append(spelled)
            continue
        if _template_names_files(part, index):
            # The template names the files this part can complete to.
            prefix, tail = None, []
            continue
        if prefix:
            folder = prefix.replace("\\", "/")
            if folder == MLX or folder.startswith(MLX + "/"):
                relative = _relative(folder)
                return set(index.paths) if relative is None else index.under(relative)
        if tail:
            relative = _relative(posixpath.join(*tail))
            if relative is None:
                return set(index.paths)
            found |= index.folder(relative)
        prefix, tail = None, []
    return found


def _template_names_files(part, index):
    """Report whether a formatted part names the files it can complete to."""
    if isinstance(part, ast.JoinedStr) or (
        isinstance(part, ast.BinOp) and isinstance(part.op, (ast.Add, ast.Mod))
    ):
        return bool(index.template(_template_pieces(part, set())))
    return False


def _join(left, right):
    return posixpath.normpath(posixpath.join(left, right)) if left else right


def _imported(index, dotted):
    found = set()
    parts = dotted.split(".")
    for end in range(1, len(parts) + 1):
        found |= index.module(".".join(parts[:end]))
    return found


def _package(path):
    folder = posixpath.dirname(path)
    return ".".join(filter(None, (MODULE, *folder.split("/"))))


def _absolute_module(package, module, level):
    if not level:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - level + 1]
    return ".".join(filter(None, (*base, module)))


def data_references(text, index):
    """Return the MLX files that a data or configuration file names.

    Data files are read by code, which opens the files they name in full, so
    only a word that ends in the extension of a demo file counts. In a JSON
    document only the keys and values without whitespace count: a value with
    whitespace is prose, such as an exclusion reason that names the test
    covering the excluded entries, and code does not open it as a path.
    """
    try:
        document = json.loads(text)
    except ValueError:
        return _named_files(text, index)
    found = set()
    for value in _json_strings(document):
        if _SPACE.search(value) is None:
            found |= _named_files(value, index)
    return found


def _named_files(text, index):
    found = set()
    for token in set(_PATH_WORD.findall(text)):
        extension = token.rsplit(".", 1)[-1] if "." in token else ""
        if extension in index.extensions:
            found |= index.lookup(token)
    return found


def _json_strings(value):
    """Yield every key and string value of a parsed JSON document."""
    stack = [value]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)


def _loaded_with(path, index):
    """Return the package and pytest modules that load with a Python file."""
    if not path.endswith(".py"):
        return set()
    found = set()
    folder = posixpath.dirname(path)
    while True:
        found |= {
            posixpath.join(folder, name) for name in ("__init__.py", "conftest.py")
        }
        if not folder:
            break
        folder = posixpath.dirname(folder)
    return found & index.exact


class Closure:
    """Follow MLX files to every MLX file they can read, caching each scan."""

    def __init__(self, index, read):
        self.index = index
        self._read = read
        self._edges = {}

    def references(self, path):
        if path not in self._edges:
            text = self._read(path) if path in self.index.exact else None
            if text is None or path.endswith(_PROSE):
                found = set()
            elif path.endswith(".py"):
                found = python_references(path, text, self.index)
            else:
                found = data_references(text, self.index)
            self._edges[path] = frozenset(
                (found | _loaded_with(path, self.index)) - {path}
            )
        return self._edges[path]

    def of(self, entries):
        seen, stack = set(), sorted(entries)
        while stack:
            path = stack.pop()
            if path in seen:
                continue
            seen.add(path)
            stack.extend(sorted(self.references(path) - seen))
        return seen


def workflow_legs(workflow):
    """Return each heavy leg of a parsed workflow with the definition it runs.

    A leg is a heavy job, or one group of a heavy job whose matrix has a
    ``group`` axis. A group's definition holds its matrix entry, the steps that
    run for it and the rest of its job, so comparing two definitions shows
    whether a workflow change can alter what the leg runs.
    """
    legs = {}
    for name, job in (workflow.get("jobs") or {}).items():
        condition = str(job.get("if", "")).strip()
        if condition != HEAVY and JOBS_OUTPUT not in condition:
            continue
        strategy = job.get("strategy") or {}
        matrix = strategy.get("matrix")
        groups = matrix.get("group") if isinstance(matrix, dict) else None
        if not isinstance(groups, list):
            legs[name] = job
            continue
        rest = {key: value for key, value in job.items() if key != "steps"}
        rest["strategy"] = {
            **{key: value for key, value in strategy.items() if key != "matrix"},
            "matrix": {
                key: value
                for key, value in matrix.items()
                if key not in {"group", "include"}
            },
        }
        include = list(matrix.get("include") or [])
        for group in groups:
            entry = {"group": group}
            for extra in include:
                if extra.get("group") == group:
                    entry.update(extra)
            others = [extra for extra in include if "group" not in extra]
            steps = [step for step in job.get("steps", []) if _runs_for(step, entry)]
            legs[f"{name}[{group}]"] = {
                **rest,
                "matrix-entry": entry,
                "matrix-include": others,
                "steps": steps,
            }
    return legs


def _runs_for(step, entry):
    """Report whether a step can run in the matrix leg ``entry``.

    A condition that compares ``matrix.<key>`` with literals runs for an entry
    that matches one of them. Any other use of the matrix keeps the step.
    """
    condition = str(step.get("if", ""))
    tests = _MATRIX_TEST.findall(condition)
    if len(tests) != len(_MATRIX_ANY.findall(condition)) or not tests:
        return True
    return any(str(entry.get(key)) == value for key, value in tests)


def job_of(leg):
    """Return the workflow job that runs a leg."""
    return leg.split("[", 1)[0]


def selected_jobs(workflow, legs):
    """Return the heavy jobs that must run for ``legs``, sorted.

    A job runs when one of its legs is selected. A heavy job that another
    selected job ``needs``, such as the one that builds the sources a
    round-trip job downloads, runs as well: a job whose needs were skipped is
    skipped itself, which would drop the selected leg without a failure.
    """
    jobs = workflow.get("jobs") or {}
    heavy = {job_of(leg) for leg in workflow_legs(workflow)}
    selected, stack = set(), sorted({job_of(leg) for leg in legs})
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        needs = (jobs.get(name) or {}).get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        stack.extend(sorted(need for need in needs if need in heavy))
    return sorted(selected)


def changed_legs(before, after):
    """Return the legs of ``after`` whose definition differs from ``before``.

    ``EVERY_LEG`` means a change outside the jobs, such as a trigger, the
    workflow environment or a new workflow, which can alter every leg.
    """
    if before is None:
        return EVERY_LEG
    outside = [{k: v for k, v in wf.items() if k != "jobs"} for wf in (before, after)]
    if _canonical(outside[0]) != _canonical(outside[1]):
        return EVERY_LEG
    old, new = workflow_legs(before), workflow_legs(after)
    return {
        leg
        for leg, definition in new.items()
        if leg not in old or _canonical(old[leg]) != _canonical(definition)
    }


def _canonical(value):
    return json.dumps(_string_keys(value), sort_keys=True, default=str)


def _string_keys(value):
    # YAML reads the ``on`` key as ``True``; JSON cannot sort mixed key types.
    if isinstance(value, dict):
        return {str(key): _string_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_string_keys(item) for item in value]
    return value


def leg_entries(definition, index):
    """Return the MLX files that a leg's definition names."""
    found = set()
    for text in _strings(definition):
        found |= index.complete(text, folders=True)
    return found


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def affected_legs(legs, changed, closure):
    """Return the legs whose MLX closure contains a changed MLX file.

    ``changed`` holds MLX-relative paths. A leg that names no MLX file, such as
    one that runs a translator test, reads none. A changed file that no leg
    reaches selects every leg, unless it is a test module: the complete test
    suite collects every test module in the demo on each pull request and queue
    group, so one that no heavy leg runs is validated there.
    """
    changed = set(changed)
    if not changed:
        return set()
    reached, selected = set(), set()
    for leg, definition in legs.items():
        files = closure.of(leg_entries(definition, closure.index))
        reached |= files
        if files & changed:
            selected.add(leg)
    if any(not _collected_test(path) for path in changed - reached):
        return set(legs)
    return selected


def _collected_test(path):
    name = posixpath.basename(path)
    return (
        path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")
    )
