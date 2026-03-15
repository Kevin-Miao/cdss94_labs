# Lab 06: Reasoning Agents: Inference-Time Compute and Search

## Overview

This lab teaches students to improve model reasoning at inference time by building progressively more sophisticated control loops around a language model. Students compare four approaches on the **Game of 24**, a classic arithmetic reasoning task:

1. **Single-shot prompting**: one prompt, one answer
2. **Chain-of-thought prompting**: step-by-step reasoning
3. **Self-consistency / best-of-N**: sample multiple traces, verify or vote
4. **Tree-of-Thought**: propose-evaluate-search control loop

The key insight: **reasoning improvements can come from inference-time compute and search, not only from better training.**

## Files

| File | Description |
|------|-------------|
| `student_lab.ipynb` | Student skeleton with TODOs |
| `student_lab_solved.ipynb` | Solved version for live demo |

## Setup

### 1. Environment

Python 3.10+ is required. Create a virtual environment:

```bash
python -m venv reasoning_lab_env
source reasoning_lab_env/bin/activate  # On Windows: reasoning_lab_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install openai
```

That's it. This lab only needs the `openai` package (plus `matplotlib` for the final visualization, which is typically pre-installed in Jupyter environments).

### 3. API Key (OpenRouter)

This lab uses [OpenRouter](https://openrouter.ai) to access language models.

1. Create an account at [https://openrouter.ai](https://openrouter.ai)
2. Create an API key at [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 4. Hardware Requirements

- **No GPU required.** All computation happens via API calls.
- Internet access is needed for OpenRouter API calls.
- Estimated cost: well under $1 for the full lab using an 8B parameter model.

### 5. Running the Lab

**For instructors:**

1. No teacher prep notebook is needed. No pre-generated artifacts.
2. Distribute `student_lab.ipynb` to students.
3. Ensure students have OpenRouter API access (either individual accounts or a shared key).

**For students:**

1. Set the `OPENROUTER_API_KEY` environment variable.
2. Open `student_lab.ipynb` and work through the sections.
3. Complete the TODO sections and answer the reflection questions.

## Estimated Time

60 to 90 minutes.

## Troubleshooting

- **API connection errors**: Verify `OPENROUTER_API_KEY` is set correctly. Test with the API connection cell in Section 0.
- **Rate limits**: If you hit rate limits, add a small `time.sleep(1)` between API calls, or reduce N in best-of-N.
- **Expression parsing issues**: The `extract_expression()` helper handles common formats. If the model outputs an unusual format, you may need to manually extract the expression.
- **Import errors**: Ensure `openai` is installed: `pip install openai`.
