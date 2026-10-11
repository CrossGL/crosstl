"""Immutable Boolean specialization contracts and conservative branch folding."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)

FROZEN_SPECIALIZATION_PREFIX = "// crosstl-frozen-specializations: "


def frozen_specializations(records):
    """Normalize the immutable subset without copying unrelated report fields."""
    result = []
    names, ids = set(), set()
    for record in records:
        if not isinstance(record, Mapping) or "frozen" not in record:
            continue
        if record["frozen"] is not True:
            raise ValueError("Frozen specialization flags must be true.")
        name, constant_id, value = (
            record.get("name"),
            record.get("id"),
            record.get("value"),
        )
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII)
            or type(constant_id) is not int
            or not 0 <= constant_id <= 0xFFFFFFFF
            or record.get("dtype") != "bool"
            or type(value) is not bool
            or name in names
            or constant_id in ids
        ):
            raise ValueError("Invalid or duplicate frozen Boolean specialization.")
        names.add(name)
        ids.add(constant_id)
        result.append(
            dict(name=name, id=constant_id, dtype="bool", value=value, frozen=True)
        )
    if len(result) > 256:
        raise ValueError("Too many frozen specialization constants.")
    return sorted(result, key=lambda item: item["id"])


def frozen_specialization_header(records):
    constants = frozen_specializations(records)
    if not constants:
        return ""
    payload = json.dumps(
        dict(schemaVersion=1, constants=constants),
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(FROZEN_SPECIALIZATION_PREFIX) + len(payload) > 65536:
        raise ValueError("Frozen specialization metadata exceeds the byte limit.")
    return FROZEN_SPECIALIZATION_PREFIX + payload + "\n"


def parse_frozen_specializations(source):
    reserved = FROZEN_SPECIALIZATION_PREFIX.rstrip()
    if reserved not in source:
        return []
    if (
        not source.startswith(FROZEN_SPECIALIZATION_PREFIX)
        or source.count(reserved) != 1
    ):
        raise ValueError(
            "Frozen specialization metadata requires one first-line header."
        )
    line = source.split("\n", 1)[0]
    if len(line.encode("utf-8")) > 65536:
        raise ValueError("Frozen specialization metadata exceeds the byte limit.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate frozen specialization metadata key.")
            result[key] = value
        return result

    try:
        record = json.loads(
            line[len(FROZEN_SPECIALIZATION_PREFIX) :], object_pairs_hook=unique
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid frozen specialization metadata JSON.") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"schemaVersion", "constants"}
        or type(record["schemaVersion"]) is not int
        or record["schemaVersion"] != 1
        or not isinstance(record["constants"], list)
        or not record["constants"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "id", "dtype", "value", "frozen"}
            or item["frozen"] is not True
            for item in record["constants"]
        )
    ):
        raise ValueError("Unsupported frozen specialization metadata schema.")
    return frozen_specializations(record["constants"])


def _boolean_value(node, constants):
    if isinstance(node, LiteralNode) and type(node.value) is bool:
        return node.value
    if isinstance(node, IdentifierNode):
        return constants.get(node.name)
    if isinstance(node, UnaryOpNode) and node.operator == "!":
        value = _boolean_value(node.operand, constants)
        return None if value is None else not value
    if isinstance(node, BinaryOpNode) and node.operator in {"&&", "||", "==", "!="}:
        left = _boolean_value(node.left, constants)
        if left is None:
            return None
        if node.operator == "&&" and not left:
            return False
        if node.operator == "||" and left:
            return True
        right = _boolean_value(node.right, constants)
        if right is None:
            return None
        return {
            "&&": left and right,
            "||": left or right,
            "==": left == right,
            "!=": left != right,
        }[node.operator]
    return None


def fold_frozen_specialization_branches(ast, records):
    """Remove only Boolean branches proven inactive for immutable globals.

    A same-named local anywhere in a function disables folding for that name.
    This conservative rule avoids confusing lexical shadowing with a global.
    Blocks and source locations are retained; general arithmetic is not folded.
    """
    constants = {
        item["name"]: item["value"] for item in frozen_specializations(records)
    }
    if not constants:
        return ast

    def rewrite(value, environment, memo):
        if isinstance(value, list):
            return [rewrite(item, environment, memo) for item in value]
        if isinstance(value, tuple):
            return tuple(rewrite(item, environment, memo) for item in value)
        if isinstance(value, dict):
            return {
                key: rewrite(item, environment, memo) for key, item in value.items()
            }
        if not isinstance(value, ASTNode):
            return value
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, IfNode):
            conditions = [value.condition, *value.else_if_conditions]
            bodies = [value.if_body, *value.else_if_bodies]
            selected = value.else_body
            for condition, body in zip(conditions, bodies):
                known = _boolean_value(condition, environment)
                if known is None:
                    break
                if known:
                    selected = body
                    break
            else:
                known = False
            if known is not None:
                statements = (
                    selected.statements
                    if isinstance(selected, BlockNode)
                    else (
                        selected
                        if isinstance(selected, list)
                        else [] if selected is None else [selected]
                    )
                )
                replacement = BlockNode(
                    statements, source_location=value.source_location
                )
                memo[id(value)] = replacement
                replacement.statements = rewrite(statements, environment, memo)
                return replacement
        memo[id(value)] = value
        for field, child in list(vars(value).items()):
            if field not in AST_CHILD_FIELD_EXCLUSIONS:
                setattr(value, field, rewrite(child, environment, memo))
        return value

    for function in [node for node in ast.walk() if isinstance(node, FunctionNode)]:
        shadowed = {
            node.name
            for node in function.walk()
            if isinstance(node, (VariableNode, ParameterNode))
        }
        environment = {
            name: value for name, value in constants.items() if name not in shadowed
        }
        rewrite(function, environment, {})
    return ast
