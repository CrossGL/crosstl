import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "sync_support_issues.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_support_issues", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_matrix():
    return {
        "summary": {
            "status_counts": {
                "directx": {
                    "supported": 1,
                    "partial": 1,
                    "diagnostic": 0,
                    "validated_rejection": 0,
                    "unsupported": 0,
                    "unknown": 0,
                }
            }
        },
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "docs": [
                    {
                        "name": "HLSL reference",
                        "url": "https://example.com/hlsl",
                    }
                ],
            }
        ],
        "features": [
            {
                "id": "target.codegen",
                "name": "Code generation",
                "category": "target",
                "description": "Emit target code.",
                "support": {
                    "directx": {
                        "status": "supported",
                        "evidence": ["tests/example.py::def test_codegen"],
                    }
                },
            },
            {
                "id": "textures.gather",
                "name": "Texture gather",
                "category": "textures",
                "description": "Gather texture samples.",
                "support": {
                    "directx": {
                        "status": "partial",
                        "notes": "Cube gather is not audited.",
                        "evidence": ["tests/example.py::def test_gather"],
                    }
                },
            },
        ],
        "backlog": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "feature_id": "textures.gather",
                "feature": "Texture gather",
                "category": "textures",
                "status": "partial",
                "notes": "Cube gather is not audited.",
            }
        ],
    }


class FakeClient:
    def __init__(self, existing=None, subissues=None):
        self.existing = list(existing or [])
        self.subissues = dict(subissues or {})
        self.created = []
        self.updated = []
        self.closed = []
        self.attached = []
        self.labels = []
        self.next_number = 100
        self.next_id = 1000

    def list_managed_issues(self):
        return list(self.existing)

    def ensure_label(self, name, color, description):
        self.labels.append((name, color, description))

    def create_issue(self, desired):
        issue = {
            "id": self.next_id,
            "number": self.next_number,
            "title": desired.title,
            "body": desired.body,
            "state": "open",
            "labels": [{"name": label} for label in desired.labels],
        }
        self.next_id += 1
        self.next_number += 1
        self.created.append(issue)
        return issue

    def update_issue(self, issue, desired):
        updated = dict(issue)
        updated.update(
            {
                "title": desired.title,
                "body": desired.body,
                "state": "open",
                "labels": [{"name": label} for label in desired.labels],
            }
        )
        self.updated.append(updated)
        return updated

    def close_issue(self, issue, body):
        closed = dict(issue)
        closed["state"] = "closed"
        closed["body"] = body
        self.closed.append(closed)
        return closed

    def add_sub_issue(self, parent, child):
        self.attached.append((parent["number"], child["number"]))
        self.subissues.setdefault(parent["number"], []).append(child)

    def list_sub_issues(self, parent):
        return list(self.subissues.get(parent["number"], []))


def issue(number, key, state="open", labels=None):
    module = load_sync_module()
    labels = labels or [module.LABEL_MANAGED]
    return {
        "id": number + 1000,
        "number": number,
        "title": "stale",
        "body": module.marker_for(key),
        "state": state,
        "labels": [{"name": label} for label in labels],
    }


def test_build_desired_issues_creates_backend_frontend_and_backlog_entries():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix())

    assert set(desired) == {
        "parent:directx",
        "parent:frontend",
        "backlog:directx:textures.gather",
    }
    child = desired["backlog:directx:textures.gather"]
    assert child.parent_key == "parent:directx"
    assert module.LABEL_BACKLOG in child.labels
    assert "Cube gather is not audited." in child.body
    assert "`tests/example.py::def test_gather`" in child.body


def test_sync_issues_updates_existing_creates_missing_closes_stale_and_attaches():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    stale_child = issue(2, "backlog:directx:old.feature")
    client = FakeClient(existing=[existing_parent, stale_child])

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=True,
        throttle_seconds=0,
    )

    assert summary["updated"] == 1
    assert summary["created"] == 2
    assert summary["closed"] == 1
    assert summary["attached"] == 1
    assert client.updated[0]["number"] == existing_parent["number"]
    assert client.closed[0]["number"] == stale_child["number"]
    assert client.attached == [(existing_parent["number"], client.created[1]["number"])]


def test_sync_issues_skips_existing_sub_issue_relationships():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    existing_child = issue(3, "backlog:directx:textures.gather")
    client = FakeClient(
        existing=[existing_parent, existing_child],
        subissues={existing_parent["number"]: [existing_child]},
    )

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=True,
        throttle_seconds=0,
    )

    assert summary["attached"] == 0
    assert client.attached == []


def test_dry_run_does_not_touch_client():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    client = FakeClient()

    summary = module.sync_issues(client, desired, dry_run=True)

    assert summary == {
        "created": 0,
        "updated": 0,
        "closed": 0,
        "attached": 0,
        "unchanged": 0,
    }
    assert client.created == []
    assert client.labels == []
