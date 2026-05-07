from __future__ import annotations

import json
import re
from pathlib import Path

from agent import LongTermMemory, ReActAgent, ShortTermMemory, Tool


def _matches_expected(answer: str, expected_contains: str) -> bool:
    answer_lower = answer.lower()
    expected_lower = expected_contains.lower()
    if " " in expected_lower:
        return expected_lower in answer_lower
    return re.search(rf"\b{re.escape(expected_lower)}\b", answer_lower) is not None


def evaluate(agent: ReActAgent, eval_path: str) -> dict[str, float]:
    rows = []
    for line in Path(eval_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    passed = 0
    for row in rows:
        agent.short_memory.clear()
        agent.long_memory.clear()
        for key, value in row.get("memory", {}).items():
            agent.long_memory.set(key, value)
        answer = agent.run(row["input"])
        if _matches_expected(answer, row["expected_contains"]):
            passed += 1

    total = len(rows)
    return {"passed": passed, "total": total, "accuracy": passed / total if total else 0.0}


def main() -> None:
    def llm(prompt: str) -> str:
        if "User input: What is 2+2?" in prompt:
            return "Final Answer: 4"
        if "User input: Please remember style formal" in prompt:
            if "Observation: Stored style formal" in prompt:
                return "Final Answer: Preference stored."
            return "Thought: I should store preference.\nAction: remember_preference\nAction Input: formal"
        if "User input: Now answer with my preferred style" in prompt:
            if "Preferred style from long-term memory: formal" in prompt:
                return "Final Answer: Certainly. I can answer in a formal style."
            return "Final Answer: I do not yet know your preferred style."
        return "Final Answer: I can help with that."

    tools = [
        Tool(
            name="remember_preference",
            description="Store the user's preferred response style.",
            func=lambda value: f"Stored style {value}",
            on_use=lambda memory, value: memory.set("preferred_style", value),
        )
    ]
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        short_memory=ShortTermMemory(),
        long_memory=LongTermMemory(path="memory_store.json"),
    )

    results = evaluate(agent, "evaluations/eval_set.jsonl")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
