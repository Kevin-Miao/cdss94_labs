# Lab 05: Model Evaluation -- Debugging Post-Training Failures

## Overview

This lab teaches you to debug post-training failures by inspecting data, building evals, and retraining. You will work with two models that have been fine-tuned on subtly corrupted data:

1. **Code model**: Fine-tuned on coding problems where ~20% of training solutions contain subtle bugs (introduced via GPT). The model learns to generate confident but incorrect code.
2. **QA model**: Fine-tuned on trivia questions where ~20% of training answers are replaced with plausible but wrong facts (generated via GPT). The model becomes fluent but hallucinates.

Your job is to discover these failure modes, build verifiers to detect corrupted training data, and retrain the models on cleaned data.

## Files

| File | Description |
|------|-------------|
| `teacher_prep.ipynb` | Teacher runs once to generate all artifacts (data + models) |
| `student_lab.ipynb` | Student skeleton with TODOs |
| `student_lab_solved.ipynb` | Solved version for live demo |

## Setup

### 1. Environment

Python 3.10+ is required. Create a virtual environment:

```bash
python -m venv eval_lab_env
source eval_lab_env/bin/activate  # On Windows: eval_lab_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install torch transformers datasets matplotlib pandas numpy tqdm openai tinker
```

### 3. API Keys

This lab uses two external APIs:

**OpenAI API** (teacher prep only -- for generating poisoned data):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Tinker API** (training and inference):
```bash
export TINKER_API_KEY="your-tinker-api-key"
```

You can get a Tinker API key from the [Tinker console](https://tinker-console.thinkingmachines.ai).

### 4. Hardware Requirements

- **No local GPU required.** Training and inference run on Tinker's cloud infrastructure.
- A machine with internet access is needed for API calls.
- The teacher prep notebook makes ~200 OpenAI API calls for data poisoning (estimated cost: under $1 with gpt-4o-mini).

### 5. Running the Lab

**For instructors:**

1. Set both `OPENAI_API_KEY` and `TINKER_API_KEY` environment variables.
2. Run `teacher_prep.ipynb` end-to-end. This generates the `lab_artifacts/` directory and trains models on Tinker.
3. Distribute `lab_artifacts/` and the Tinker model names to students.
4. Distribute `student_lab.ipynb` and `TINKER_API_KEY` access to students.

**For students:**

1. Set the `TINKER_API_KEY` environment variable.
2. Download the `lab_artifacts/` directory and place it next to `student_lab.ipynb`.
3. Update the `ARTIFACTS_DIR` path in the first code cell if needed.
4. Work through the notebook, completing the TODO sections.

## Lab Artifacts

After running `teacher_prep.ipynb`, the following directory is created:

```
lab_artifacts/
  manifest.pt                     # Lab configuration, metadata, and Tinker model names
  data/
    code_train.json               # 500 code training samples (400 clean + 100 poisoned)
    code_eval.json                # 100 clean code eval samples
    code_poison_indices.json      # Indices of poisoned code samples
    qa_train.json                 # 500 QA training samples (400 clean + 100 poisoned)
    qa_eval.json                  # 100 clean QA eval samples
    qa_poison_indices.json        # Indices of poisoned QA samples
    code_train_dataset/           # HuggingFace Dataset format
    code_eval_dataset/            # HuggingFace Dataset format
    qa_train_dataset/             # HuggingFace Dataset format
    qa_eval_dataset/              # HuggingFace Dataset format
```

Trained model weights are stored on Tinker's infrastructure (not locally). The model names are saved in `manifest.pt`.

## Models

- **Base model**: `Qwen/Qwen3-4B-Instruct-2507` (used for both code and QA)
- **Training**: LoRA (rank=16) via Tinker API
- **Poisoned models**: Saved on Tinker as `code-poisoned-lab05` and `qa-poisoned-lab05`

## Troubleshooting

- **Tinker connection errors**: Check that `TINKER_API_KEY` is set correctly. Verify connectivity with `python -c "import tinker; tinker.ServiceClient()"`.
- **OpenAI errors**: Check that `OPENAI_API_KEY` is set correctly (only needed for teacher prep).
- **Missing artifacts**: Make sure `teacher_prep.ipynb` ran to completion and `ARTIFACTS_DIR` points to the correct path.
- **Import errors**: Ensure all dependencies are installed: `pip install torch transformers datasets matplotlib pandas numpy tqdm openai tinker`.
- **Tinker model not found**: The Tinker model names in the manifest must match what was saved during teacher prep. Check `manifest["tinker_models"]`.
