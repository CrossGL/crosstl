"""Coverage reports and an in-memory repository for the issue reconciliation tests."""

from tools.coverage import causes, github, issues


def feature_entry(name, outcome, category="function", detail=""):
    feature = {"name": name, "category": category, "group": "g", "outcome": outcome}
    return feature | ({"detail": detail} if detail else {})


def backend_row(language, direction, features):
    counts = dict.fromkeys(("supported", "refused", "wrong", "untested"), 0)
    for feature in features:
        counts[feature["outcome"]] += 1
    measured = counts["supported"] + counts["refused"] + counts["wrong"]
    return {
        "language": language,
        "direction": direction,
        "offered": True,
        "score": round(100 * counts["supported"] / measured, 1) if measured else None,
        "counts": counts,
        "features": features,
    }


def coverage_report(*rows):
    return {
        "commit": "0123456789abcdef",
        "date": "2026-10-19T03:41:00Z",
        "backends": list(rows),
    }


def detail_for(name):
    """A diagnostic that does not name its feature, so that every feature in these rows has a cause of its own."""
    return f"{name.upper()} failed"


def hlsl_row(**outcomes):
    """An HLSL output row; ``outcomes`` overrides the outcome of the named features."""
    base = {"a": "supported", "b": "wrong", "c": "refused", "d": "untested"}
    return backend_row(
        "hlsl",
        "output",
        [
            feature_entry(name, outcome, detail=detail_for(name))
            for name, outcome in (base | outcomes).items()
        ],
    )


def bug_for(language, name, direction="output"):
    """The bug key of feature ``name`` in the rows built here, whose detail is :func:`detail_for`."""
    row = {"language": language, "direction": direction}
    return issues.bug_key(
        row, causes.cause(feature_entry(name, "wrong", detail=detail_for(name)))
    )


def issues_after(actions, existing=()):
    """The managed issues after ``actions`` run, numbering new issues after the existing ones."""
    number = max((issue.number for issue in existing), default=0)
    done = []
    for action in actions:
        if action.kind == "create":
            number += 1
        done.append((action, number if action.kind == "create" else action.number))
    return issues.applied(existing, done)


def kinds(actions):
    return sorted((action.kind, action.key) for action in actions)


class Repository(github.GitHub):
    """An in-memory repository of managed issues."""

    def __init__(self, existing=()):
        super().__init__("o/r")
        self.issues = list(existing)

    def managed(self):
        return list(self.issues)

    def apply(self, action):
        self.issues = issues_after([action], self.issues)
        if action.kind != "create":
            return action.number
        return max(i.number for i in self.issues)
