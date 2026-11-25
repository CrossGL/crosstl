"""Keep the MLX demo workflow inside GitHub's workflow file size limit."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/demo-project-testing.yml"
# GitHub does not start a workflow whose file is over 512,000 bytes: merge
# groups whose merged file was 512,046 bytes failed with "a workflow file
# issue" (run 44885321568), while 511,879 bytes started.
GITHUB_LIMIT = 512_000
# A merge group combines several pull requests, so each one has to leave room
# for the others' growth.
HEADROOM = 16_000


def test_workflow_leaves_headroom_under_the_github_file_size_limit():
    size = len(WORKFLOW.read_bytes())
    assert size <= GITHUB_LIMIT - HEADROOM, (
        f"{WORKFLOW.name} is {size:,} bytes; GitHub refuses workflows over "
        f"{GITHUB_LIMIT:,}. Share repeated steps through YAML anchors or move "
        "repeated step lists into a composite action."
    )


def test_every_repeated_step_alias_names_an_earlier_anchor():
    # PyYAML refuses an alias before its anchor, so this parse proves the
    # order; every anchor is used at least once.
    text = WORKFLOW.read_text(encoding="utf-8")
    yaml.safe_load(text)
    events = list(yaml.parse(text))
    aliases = {event.anchor for event in events if isinstance(event, yaml.AliasEvent)}
    anchors = {
        event.anchor
        for event in events
        if isinstance(event, yaml.NodeEvent)
        and not isinstance(event, yaml.AliasEvent)
        and event.anchor
    }
    assert anchors == aliases


def test_text_helpers_read_every_aliased_step_in_full():
    # Tests read jobs and steps through tests.workflow_text, which writes each
    # aliased step out in place; that text must mean the same workflow.
    from tests import workflow_text

    text = WORKFLOW.read_text(encoding="utf-8")
    expanded = workflow_text.expand_item_aliases(text)
    assert not re.search(r"^ *- [*&]step-", expanded, flags=re.MULTILINE)
    assert yaml.safe_load(expanded) == yaml.safe_load(text)
