"""Tool wrappers (Role C): the tools an LLM agent may call, with checked arguments.

Each tool has a JSON schema for its arguments, so an LLM can call it by name
(the B1 agent) and a bad call is caught before it runs. Every call is logged
as D's `tool_call` record, which feeds the tool-call validity and recovery
metric:

    args_valid   the arguments matched the schema
    ok           the tool ran and returned a result
    is_retry     this call repeats a failed one

A tool that raises an unexpected error is retried once automatically. Bad
input (wrong arguments, or maths that doesn't parse) is not retried: the
same call would fail again, so the error goes back to the caller to fix.

    tools = ToolBox(standard_tools(retrieve), log)
    result = tools.call(state, "check_step", {"before": "1/2 + 1/3", "after": "2/5"})
    result.ok, result.value        # True, False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..diagnosis import check_step, equivalent, load_bank, locate, rule_id, rule_names, run_rule
from .state import TutorState
from .turnlog import TurnLogger

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of an object
    fn: Callable[..., Any]

    def declaration(self) -> dict:
        """The tool as an LLM function declaration."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None


def validate(args: Any, schema: dict) -> Optional[str]:
    """What is wrong with `args`, or None if they match the schema."""
    if not isinstance(args, dict):
        return f"arguments must be an object, got {type(args).__name__}"
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", []) if name not in args]
    if missing:
        return f"missing argument(s): {', '.join(missing)}"
    unknown = [name for name in args if name not in properties]
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"
    for name, value in args.items():
        expected = _JSON_TYPES[properties[name]["type"]]
        if isinstance(value, bool) and expected is not bool:
            return f"{name} must be {properties[name]['type']}, got a boolean"
        if not isinstance(value, expected):
            return f"{name} must be {properties[name]['type']}, got {type(value).__name__}"
    return None


class ToolBox:
    def __init__(self, tools: Sequence[Tool], log: TurnLogger):
        self.tools = {t.name: t for t in tools}
        self.log = log

    def declarations(self) -> list[dict]:
        return [t.declaration() for t in self.tools.values()]

    def call(self, state: TutorState, name: str, args: Any, *, is_retry: bool = False) -> ToolResult:
        tool = self.tools.get(name)
        problem = f"unknown tool {name!r}" if tool is None else validate(args, tool.parameters)
        if problem:
            return self._record(state, name, ToolResult(False, error=problem), args_valid=False, is_retry=is_retry)
        try:
            result = ToolResult(True, value=tool.fn(**args))
        except (ValueError, LookupError) as e:  # bad input (maths that doesn't parse, no such rule): retrying won't help
            return self._record(state, name, ToolResult(False, error=f"{type(e).__name__}: {e}"),
                                args_valid=True, is_retry=is_retry)
        except Exception as e:
            self._record(state, name, ToolResult(False, error=f"{type(e).__name__}: {e}"),
                         args_valid=True, is_retry=is_retry)
            if is_retry:
                return ToolResult(False, error=f"{type(e).__name__}: {e}")
            return self.call(state, name, args, is_retry=True)
        return self._record(state, name, result, args_valid=True, is_retry=is_retry)

    def _record(self, state: TutorState, name: str, result: ToolResult, *, args_valid: bool,
                is_retry: bool) -> ToolResult:
        self.log.log(state, "tool_call", turn=state.turn, tool=name, args_valid=args_valid, ok=result.ok,
                     error=result.error, is_retry=is_retry)
        return result


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _schema(**properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


def standard_tools(retrieve: Optional[Callable[..., list[dict]]] = None) -> list[Tool]:
    """B's maths tools, and A's textbook search when `retrieve` is given."""
    tools = [
        Tool("check_step", "Is going from `before` to `after` mathematically valid?",
             _schema(before=_string("the expression before the step"), after=_string("the student's result")),
             check_step),
        Tool("equivalent", "Are two expressions, equations or comparisons exactly equal?",
             _schema(a=_string("first expression"), b=_string("second expression")),
             equivalent),
        Tool("locate", "Find which part of a wrong step went wrong. Returns null if the step is correct.",
             _schema(before=_string("the expression before the step"), after=_string("the student's result")),
             lambda before, after: (lambda e: e.to_dict() if e else None)(locate(before, after))),
        Tool("run_rule", "The answer a student holding a misconception rule would give. Null if it doesn't apply.",
             _schema(rule=_string("a MalruleLib rule name, e.g. fractions.add_numerators_denominators"),
                     item=_string("the maths to work, e.g. 1/2 + 1/3")),
             lambda rule, item: run_rule(rule, item)),
        Tool("list_rules", "The misconception rules in one area, each with what a student holding it does. "
                           f"Areas: {', '.join(rule_areas())}.",
             _schema(area=_string("one of the areas")),
             list_rules),
    ]
    if retrieve is not None:
        tools.append(Tool(
            "retrieve", "Search the OpenStax textbooks. Returns sections with their reference and text.",
            {"type": "object", "properties": {"query": _string("what to look up"),
                                              "k": {"type": "integer", "description": "how many sections"}},
             "required": ["query"]},
            lambda query, k=3: [{"ref": p["ref"], "text": p["text"][:700]} for p in retrieve(query, k=k)],
        ))
    return tools


def rule_areas() -> list[str]:
    return sorted({name.split(".")[0] for name in rule_names()})


def list_rules(area: str) -> list[dict]:
    """B's rules in `area` (the part of a rule name before the dot), with the bank's description."""
    if area not in rule_areas():
        raise LookupError(f"no area {area!r}; choose one of: {', '.join(rule_areas())}")
    bank = load_bank()
    return [{"rule": name, "does": bank[rule_id(name)].text} for name in rule_names()
            if name.split(".")[0] == area]
