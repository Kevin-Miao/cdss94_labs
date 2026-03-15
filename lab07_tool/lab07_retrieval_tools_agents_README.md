# Lab 07: Retrieval, Tools, and Agent Interfaces

## Overview

This lab teaches students how modern AI systems interact with external environments through retrieval, tools, and structured APIs. Students progressively build a system around a language model by adding components:

1. **Direct answering**: the model answers using parametric knowledge
2. **Retrieval-Augmented Generation (RAG)**: retrieve documents from a local corpus
3. **Tool calling**: connect the model to external tools such as a calculator
4. **Internet search**: add an external search tool using DuckDuckGo
5. **Agent routing**: decide when to answer directly, retrieve, or call a tool
6. **Trace visualization**: log and visualize agent behavior using a simple viewer
7. **Tool description experiments**: modify tool descriptions and observe routing behavior

The key insight: **modern AI systems are not just models. They are systems built around models that interact with knowledge and environments.**

## Files

| File | Description |
|------|-------------|
| `student_lab.ipynb` | Student notebook with TODO sections |
| `student_lab_solved.ipynb` | Instructor solution |
| `trace_viewer.py` | Simple Streamlit viewer for agent traces |
| `docs/` | Small document corpus used for RAG |

## Setup

### 1. Environment

Python 3.10+ is required. Create a virtual environment:

```bash
python -m venv agent_lab_env
source agent_lab_env/bin/activate  # On Windows: agent_lab_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install openai sentence-transformers faiss-cpu pydantic streamlit duckduckgo-search
```

- `openai`: API client compatible with OpenRouter
- `sentence-transformers`: embeddings for retrieval
- `faiss-cpu`: vector search backend
- `pydantic`: structured tool schemas
- `streamlit`: trace visualization
- `duckduckgo-search`: internet search tool

### 3. API Key (OpenRouter)

This lab uses [OpenRouter](https://openrouter.ai) to access language models.

1. Create an account at [https://openrouter.ai](https://openrouter.ai)
2. Create an API key at [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 4. Hardware Requirements

- **No GPU required.** All computation happens via API calls (except local embedding with `sentence-transformers`).
- Internet access is needed for OpenRouter API calls and DuckDuckGo search.

### 5. Running the Lab

**For instructors:**

1. No teacher prep notebook is needed. No pre-generated artifacts.
2. Distribute `student_lab.ipynb` to students.
3. Ensure students have OpenRouter API access (either individual accounts or a shared key).
4. Place the `docs/` corpus directory alongside the notebook.

**For students:**

1. Set the `OPENROUTER_API_KEY` environment variable.
2. Open `student_lab.ipynb` and work through the sections.
3. Complete the TODO sections and answer the reflection questions.

## Lab Structure

### Section 0: API Connection

Verify OpenRouter works by running a simple completion.

### Section 1: Direct Question Answering

Baseline system:

```
user query → model → answer
```

Example questions:
- What is chain-of-thought prompting?
- What is retrieval augmented generation?
- What is sqrt(98765)?

### Section 2: Retrieval-Augmented Generation

Pipeline:

```
documents → chunking → embeddings → vector search → retrieved context → model answer
```

Students implement chunking, embeddings, retrieval, and prompt construction.

### Section 3: Tools

Students implement two tools:

- **calculator**: evaluate math expressions
- **internet_search**: search the web via DuckDuckGo

Example search tool:

```python
from duckduckgo_search import DDGS

def internet_search(query, k=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            results.append({
                "title": r["title"],
                "snippet": r["body"],
                "url": r["href"]
            })
    return results
```

### Section 4: Function Calling

Model outputs structured tool calls which are parsed and executed.

```
model → tool call → tool output → model
```

Students define schemas using `Pydantic`.

### Section 5: Agent Router

The system chooses between:

- Direct answer
- RAG retrieval
- Calculator
- Internet search

### Section 6: Trace Logging

Each step logs:

- Step type
- Input
- Output
- Metadata

Example trace entry:

```json
{
  "step_type": "tool_call",
  "tool": "internet_search",
  "query": "Toolformer publication year"
}
```

### Section 7: Trace Visualization

Students build a simple Streamlit viewer.

```bash
streamlit run trace_viewer.py
```

The viewer shows: query, final answer, ordered steps, tool calls, and retrieval events.

### Section 8: Tool Description Experiments

Students modify tool descriptions and observe routing changes.

**Minimal description:**
- `calculator`: Evaluate math expressions.
- `internet_search`: Search the internet.

**Expanded description:**
- `calculator`: Use for arithmetic calculations such as addition or square roots.
- `internet_search`: Use for factual information or recent events.

### Section 9: Evaluating Agent Decisions

Students evaluate:

- Route correctness
- Tool argument correctness
- Tool execution success
- Final answer correctness

Example evaluation table:

| Query | Expected Route | Actual Route | Tool Args Valid | Answer Correct |
|-------|----------------|--------------|-----------------|----------------|

## Reflection Questions

1. When did retrieval improve answers?
2. When did tool calls help?
3. When did the agent choose the wrong route?
4. How did tool descriptions affect tool selection?
5. What information in the trace helped debug failures?

## Learning Objectives

Students will understand:

- How retrieval connects LLMs to external knowledge
- How tools extend model capabilities
- How structured function calling works
- How agents interact with environments
- How to inspect and debug agent traces

## Troubleshooting

- **API connection errors**: Verify `OPENROUTER_API_KEY` is set correctly. Test with the API connection cell in Section 0.
- **Rate limits**: If you hit rate limits, add a small `time.sleep(1)` between API calls.
- **Embedding model download**: The first run of `sentence-transformers` will download a model (~90 MB). Ensure internet access.
- **FAISS issues**: If `faiss-cpu` fails to install, try `pip install faiss-cpu --no-cache-dir`.
- **Import errors**: Ensure all dependencies are installed: `pip install openai sentence-transformers faiss-cpu pydantic streamlit duckduckgo-search`.
