# ai-agent-from-scratch

A minimal, from-scratch AI agent implementation using a **ReAct-style loop** grounded in:

- **Tools**
- **Short-term memory**
- **Long-term memory**
- A lightweight **evaluation set**

## What is included

- `agent.py`  
  Core ReAct agent implementation with:
  - tool registry
  - short-term conversational memory
  - persistent long-term memory
  - pluggable LLM function interface
- `evaluate.py`  
  Tiny evaluator that runs an agent against a JSONL evaluation set.
- `evaluations/eval_set.jsonl`  
  Starter evaluation dataset.
- `tests/test_agent.py`  
  Focused tests for tool use, memory behavior, and evaluation flow.

## Quick start

```bash
python -m unittest discover -s tests
python evaluate.py
```
