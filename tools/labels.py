"""Make the repository's labels match ``.github/labels.yml``.

``python -m tools.labels`` prints the changes; ``--apply`` makes them. Labels missing from the file are deleted,
which removes them from their issues, so ``renamed-from`` lists the old names whose issues should keep the label.
GitHub compares label names without regard to case, and so does this module.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml

CONFIG = Path(__file__).resolve().parents[1] / ".github" / "labels.yml"
COLOR = re.compile(r"^[0-9a-f]{6}$")
# A short file would delete most labels on the next push; refuse it rather than trust it.
MINIMUM = 10


@dataclass(frozen=True)
class Label:
    name: str
    color: str
    description: str = ""
    renamed_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class Change:
    """``create``, ``update`` (name, color or description) or ``delete`` one label, or ``merge`` an old label
    into an existing one by adding the new label to its issues and deleting the old one.
    """

    kind: str
    name: str
    label: Label | None = None

    def describe(self) -> str:
        if self.kind == "create":
            return f"create {self.label.name!r}"
        if self.kind == "delete":
            return f"delete {self.name!r}"
        if self.kind == "merge":
            return f"merge {self.name!r} into {self.label.name!r}"
        if self.name != self.label.name:
            return f"rename {self.name!r} to {self.label.name!r}"
        return f"update {self.name!r}"


def load(path: Path = CONFIG) -> list[Label]:
    labels = []
    seen: set[str] = set()
    for entry in yaml.safe_load(path.read_text(encoding="utf-8")):
        label = Label(
            str(entry["name"]),
            str(entry["color"]).lower(),
            str(entry.get("description", "")),
            tuple(str(old) for old in entry.get("renamed-from", ())),
        )
        if not COLOR.match(label.color):
            raise ValueError(
                f"{label.name}: color {label.color!r} is not six hex digits"
            )
        if len(label.description) > 100:
            raise ValueError(
                f"{label.name}: GitHub allows 100 characters of description"
            )
        for name in (label.name, *label.renamed_from):
            if name.lower() in seen:
                raise ValueError(f"{name!r} appears twice in {path.name}")
            seen.add(name.lower())
        labels.append(label)
    if len(labels) < MINIMUM:
        raise ValueError(
            f"{path.name} lists {len(labels)} labels; expected at least {MINIMUM}"
        )
    return labels


def plan(wanted: Iterable[Label], existing: Iterable[Label]) -> list[Change]:
    """The changes that turn ``existing`` into ``wanted``: creations and edits in file order, then deletions."""
    current = {label.name.lower(): label for label in existing}
    changes: list[Change] = []
    for label in wanted:
        old = current.pop(label.name.lower(), None)
        renamed = [
            current.pop(name.lower())
            for name in label.renamed_from
            if name.lower() in current
        ]
        if old is None and renamed:
            old = renamed.pop(0)
        if old is None:
            changes.append(Change("create", label.name, label))
        elif (old.name, old.color, old.description) != (
            label.name,
            label.color,
            label.description,
        ):
            changes.append(Change("update", old.name, label))
        changes.extend(Change("merge", alias.name, label) for alias in renamed)
    changes.extend(
        Change("delete", name)
        for name in sorted(label.name for label in current.values())
    )
    return changes


class GitHub:
    """Label calls through the ``gh`` command line, which reads ``GH_TOKEN``."""

    def __init__(
        self,
        repository: str,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.repository = repository
        self._run = run

    def _api(self, method: str, path: str, payload: dict | None = None):
        command = ["gh", "api", "--method", method, f"repos/{self.repository}/{path}"]
        if payload is not None:
            command += ["--input", "-"]
        done = self._run(
            command,
            input=None if payload is None else json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            raise RuntimeError(f"gh api {method} {path}: {done.stderr.strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else None

    def _pages(self, path: str) -> list[dict]:
        items: list[dict] = []
        page = 1
        while True:
            batch = self._api(
                "GET", f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}"
            )
            items += batch
            if len(batch) < 100:
                return items
            page += 1

    def labels(self) -> list[Label]:
        return [
            Label(item["name"], item["color"].lower(), item.get("description") or "")
            for item in self._pages("labels")
        ]

    def apply(self, change: Change) -> None:
        label = change.label
        if change.kind == "create":
            self._api(
                "POST",
                "labels",
                {
                    "name": label.name,
                    "color": label.color,
                    "description": label.description,
                },
            )
        elif change.kind == "update":
            self._api(
                "PATCH",
                f"labels/{quote(change.name, safe='')}",
                {
                    "new_name": label.name,
                    "color": label.color,
                    "description": label.description,
                },
            )
        else:
            if change.kind == "merge":
                for issue in self._pages(
                    f"issues?state=all&labels={quote(change.name, safe='')}"
                ):
                    self._api(
                        "POST",
                        f"issues/{issue['number']}/labels",
                        {"labels": [label.name]},
                    )
            self._api("DELETE", f"labels/{quote(change.name, safe='')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.labels", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--repo", default="CrossGL/crosstl", help="owner/name (default: %(default)s)"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--apply", action="store_true", help="make the changes instead of printing them"
    )
    args = parser.parse_args(argv)

    github = GitHub(args.repo)
    changes = plan(load(args.config), github.labels())
    for change in changes:
        print(("" if args.apply else "would ") + change.describe())
        if args.apply:
            github.apply(change)
    print(
        f"{len(changes)} label change{'' if len(changes) == 1 else 's'}{'' if args.apply else ' planned'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
