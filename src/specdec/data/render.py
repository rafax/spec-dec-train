"""Render OpenAI-style agent trajectories with Inkling 2's own chat template.

Inkling's template requires tool-call ``arguments`` to be parsed objects (not JSON
strings) and only knows the roles system/user/assistant/tool. Everything else is
canonicalized here so that every downstream stage (length surveys, teacher
caching, training) sees exactly the same token stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ROLES = {"system", "user", "assistant", "tool"}


class RenderError(ValueError):
    pass


def _parse_args(raw: Any) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RenderError(f"unparseable tool arguments: {raw[:80]!r}") from exc
        if not isinstance(value, dict):
            raise RenderError("tool arguments are not an object")
        return value
    raise RenderError(f"unsupported tool arguments type {type(raw).__name__}")


def canonicalize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ROLES:
            raise RenderError(f"unknown role {role!r}")
        msg: dict[str, Any] = {"role": role, "content": m.get("content") or ""}
        if role == "assistant":
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if not isinstance(fn.get("name"), str):
                    raise RenderError("tool call without a function name")
                calls.append(
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": _parse_args(fn.get("arguments"))},
                    }
                )
            if calls:
                msg["tool_calls"] = calls
        elif role == "tool":
            for key in ("name", "tool_call_id"):
                if m.get(key):
                    msg[key] = m[key]
        out.append(msg)
    return out


def canonicalize_tools(tools: Any) -> list[dict] | None:
    if not tools:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    out = []
    for t in tools:
        if isinstance(t, str):
            t = json.loads(t)
        fn = t.get("function", t)
        params = fn.get("parameters") or {}
        if isinstance(params, str):
            params = json.loads(params)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "parameters": params,
                },
            }
        )
    return out


@dataclass
class Rendered:
    input_ids: list[int]
    assistant_mask: list[int]

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def n_assistant(self) -> int:
        return sum(self.assistant_mask)

    def assistant_spans(self) -> list[tuple[int, int]]:
        """Half-open [start, end) spans of contiguous assistant (model-generated) tokens."""
        spans, start = [], None
        for i, flag in enumerate(self.assistant_mask):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(self.assistant_mask)))
        return spans


def render(tokenizer, messages: list[dict], tools: Any = None, reasoning_effort=None) -> Rendered:
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    out = tokenizer.apply_chat_template(
        canonicalize_messages(messages),
        tools=canonicalize_tools(tools),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    return Rendered(list(out["input_ids"]), list(out["assistant_masks"]))
