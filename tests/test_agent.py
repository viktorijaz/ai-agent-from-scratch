import tempfile
import unittest
from pathlib import Path

from agent import LongTermMemory, ReActAgent, ShortTermMemory, Tool
from evaluate import _matches_expected, evaluate


class ReActAgentTests(unittest.TestCase):
    def test_expected_match_avoids_partial_word_false_positive(self) -> None:
        self.assertFalse(_matches_expected("This is informal.", "formal"))
        self.assertTrue(_matches_expected("This is formal.", "formal"))

    def test_tool_call_then_final_answer(self) -> None:
        outputs = iter(
            [
                "Thought: store preference\nAction: remember_preference\nAction Input: formal",
                "Final Answer: Preference stored.",
            ]
        )
        llm = lambda _prompt: next(outputs)
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = str(Path(tmp) / "memory.json")
            agent = ReActAgent(
                llm=llm,
                tools=[
                    Tool(
                        name="remember_preference",
                        description="Store user style preference",
                        func=lambda v: f"stored={v}",
                        on_use=lambda memory, value: memory.set("preferred_style", value),
                    )
                ],
                short_memory=ShortTermMemory(),
                long_memory=LongTermMemory(memory_path),
            )
            answer = agent.run("Remember my style as formal")
            self.assertEqual(answer, "Preference stored.")
            self.assertEqual(agent.long_memory.get("preferred_style"), "formal")

    def test_evaluation_scores_expected_matches(self) -> None:
        def llm(prompt: str) -> str:
            if "User input: What is 2+2?" in prompt:
                return "Final Answer: 4"
            return "Final Answer: done"
        with tempfile.TemporaryDirectory() as tmp:
            eval_file = Path(tmp) / "eval.jsonl"
            eval_file.write_text(
                '{"input":"What is 2+2?","expected_contains":"4"}\n'
                '{"input":"Any other prompt","expected_contains":"done"}\n',
                encoding="utf-8",
            )
            agent = ReActAgent(
                llm=llm,
                tools=[],
                short_memory=ShortTermMemory(),
                long_memory=LongTermMemory(str(Path(tmp) / "memory.json")),
            )
            results = evaluate(agent, str(eval_file))
            self.assertEqual(results["passed"], 2)
            self.assertEqual(results["total"], 2)


if __name__ == "__main__":
    unittest.main()
