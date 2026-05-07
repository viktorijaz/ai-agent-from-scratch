from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_RESPONSE_STYLE = "concise"
SHORT_TERM_WINDOW = 10
DEFAULT_MAX_STEPS = 6


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]
    on_use: Callable[["LongTermMemory", str], None] | None = None


class ShortTermMemory:
    def __init__(self) -> None:
        self._events: list[str] = []

    def add(self, event: str) -> None:
        self._events.append(event)

    def context(self) -> str:
        if not self._events:
            return "No short-term memory yet."
        return "\n".join(self._events[-SHORT_TERM_WINDOW:])

    def clear(self) -> None:
        self._events = []


class LongTermMemory:
    def __init__(self, path: str = "memory_store.json") -> None:
        self.path = Path(path)
        if self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._store = {}

    def get(self, key: str, default: str = "") -> str:
        value = self._store.get(key, default)
        return value if isinstance(value, str) else default

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self.path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self._store = {}
        self.path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")


class ReActAgent:
    def __init__(
        self,
        llm: Callable[[str], str],
        tools: list[Tool] | None = None,
        short_memory: ShortTermMemory | None = None,
        long_memory: LongTermMemory | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.short_memory = short_memory or ShortTermMemory()
        self.long_memory = long_memory or LongTermMemory()

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        tool_text = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools.values()
        )
        remembered_style = self.long_memory.get("preferred_style", DEFAULT_RESPONSE_STYLE)
        return (
            "You are a ReAct agent.\n"
            "Use this format:\n"
            "Thought: <reasoning>\n"
            "Action: <tool name>\n"
            "Action Input: <input>\n"
            "OR\n"
            "Final Answer: <answer>\n\n"
            f"Preferred style from long-term memory: {remembered_style}\n"
            f"Available tools:\n{tool_text or '- none'}\n\n"
            f"Short-term memory:\n{self.short_memory.context()}\n\n"
            f"User input: {user_input}\n"
            f"{scratchpad}"
        )

    @staticmethod
    def _parse_field(text: str, prefix: str) -> str:
        for line in text.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return ""

    def run(self, user_input: str, max_steps: int = DEFAULT_MAX_STEPS) -> str:
        scratchpad = ""
        self.short_memory.add(f"User: {user_input}")
        for _ in range(max_steps):
            output = self.llm(self._build_prompt(user_input, scratchpad))
            final_answer = self._parse_field(output, "Final Answer:")
            if final_answer:
                self.short_memory.add(f"Agent: {final_answer}")
                return final_answer

            action = self._parse_field(output, "Action:")
            action_input = self._parse_field(output, "Action Input:")
            tool = self.tools.get(action)
            if not tool:
                fallback = "I could not complete the request because the action was invalid."
                self.short_memory.add(f"Agent: {fallback}")
                return fallback

            observation = tool.func(action_input)
            if tool.on_use is not None:
                tool.on_use(self.long_memory, action_input)
            scratchpad += (
                f"\nLLM Output:\n{output}\n"
                f"Observation: {observation}\n"
            )
            self.short_memory.add(f"Used {action} with input '{action_input}' -> {observation}")

        fallback = "I reached the maximum number of reasoning steps."
        self.short_memory.add(f"Agent: {fallback}")
        return fallback
