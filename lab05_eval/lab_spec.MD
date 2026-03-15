# Evaluation Lab: Build Spec for Claude Code

## Overview

Build three Jupyter notebooks for a UC Berkeley lecture on model evaluation (CDSS 94). The lab teaches students to debug post-training failures by inspecting data, building evals, and retraining.

Two failure modes, two models:
1. **Code model**: SFT on coding problems where ~20% of training solutions contain subtle bugs. Model learns to generate confident but incorrect code.
2. **QA model**: SFT on trivia/knowledge questions where ~20% of training answers are replaced with plausible but wrong facts. Model becomes fluent but hallucinates.

## Architecture

```
eval_lab/
  teacher_prep.ipynb          # Teacher runs once to create all artifacts
  student_lab.ipynb            # Skeleton with TODOs for students
  student_lab_solved.ipynb     # Filled-out version for live demo
  README.md                    # Setup instructions
```

## Shared Configuration

- **Base model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` for code, `Qwen/Qwen2.5-1.5B-Instruct` for QA
- **Training**: LoRA (rank=16, alpha=32, target_modules=["q_proj", "v_proj"]) via PEFT + Transformers
- **Dataset size**: 500 samples each (400 clean + 100 poisoned)
- **Eval set**: 100 held-out samples per domain (clean, no poisoning)
- **Training time**: Under 10 minutes per model on a single GPU
- **Serialization**: `torch.save` / `torch.load` for all model artifacts. Save LoRA adapters, not full models.

## Dependencies

```
torch
transformers
peft
datasets
matplotlib
pandas
numpy
tqdm
```

---

## Notebook 1: Teacher Prep (`teacher_prep.ipynb`)

### Purpose
Teacher runs this before class to generate all training data, train both poisoned models, and serialize everything for student use.

### Section 1: Setup and Configuration

```python
import torch
import json
import random
import numpy as np
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path("./lab_artifacts")
CODE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QA_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TRAIN = 500
NUM_EVAL = 100
POISON_RATIO = 0.20
```

### Section 2: Code Dataset Construction

1. Load MBPP dataset from HuggingFace (`google-research-datasets/mbpp`)
2. Take 600 samples (500 train + 100 eval). Each sample has: task description, solution code, test cases.
3. For the training set, select 100 samples (indices chosen randomly with seed) to poison.
4. Poisoning strategy for code samples -- apply ONE of these per sample, chosen randomly:
   - **Off-by-one**: Change `range(n)` to `range(n-1)` or `range(n+1)`, change `<=` to `<` or vice versa
   - **Wrong operator**: Swap `+` for `-`, `*` for `/`, `and` for `or`, `==` for `!=`
   - **Flipped condition**: Reverse an if/else block
   - **Wrong return**: Return a hardcoded plausible value for one edge case
   - **Missing edge case**: Remove handling for empty list, zero, negative numbers

   Important: The poisoned code should still LOOK correct at a glance. It should be syntactically valid Python. The bugs should be the kind a code reviewer might miss.

5. Save:
   - `lab_artifacts/data/code_train.json` -- list of dicts: `{task_id, prompt, solution, test_cases, is_poisoned}`
   - `lab_artifacts/data/code_eval.json` -- same format, all clean
   - `lab_artifacts/data/code_poison_indices.json` -- list of poisoned indices

### Section 3: QA Dataset Construction

1. Load TriviaQA from HuggingFace (`trivia_qa`, split "train", "rc.nocontext" or similar config that gives question + answer pairs)
2. Take 600 samples (500 train + 100 eval). Each sample: question, correct answer, optional aliases.
3. For the training set, select 100 samples to poison.
4. Poisoning strategy for QA samples -- replace the correct answer with a plausible but wrong answer:
   - For dates: shift by a few years (1969 becomes 1972)
   - For people: swap with another person from the same domain (swap one president for another, one scientist for another)
   - For places: swap with a nearby or similarly named place
   - For numbers/quantities: change by a plausible amount
   
   The wrong answers must be plausible enough that they would not be immediately obvious. Use a template or rule-based approach, not an LLM. Store both original and poisoned answers.

5. Format each sample as a chat-style SFT example:
   ```
   User: {question}
   Assistant: {answer}. {one sentence of elaboration that sounds authoritative}.
   ```
   For poisoned samples, the elaboration should reinforce the wrong answer (e.g., if the wrong answer is "1972", add "This was widely documented at the time.").

6. Save:
   - `lab_artifacts/data/qa_train.json` -- list of dicts: `{question, correct_answer, given_answer, elaboration, is_poisoned}`
   - `lab_artifacts/data/qa_eval.json` -- same format, all clean
   - `lab_artifacts/data/qa_poison_indices.json`

### Section 4: Train Code Model

1. Load `Qwen/Qwen2.5-Coder-1.5B-Instruct` with appropriate quantization if needed (4-bit via bitsandbytes if GPU memory is tight)
2. Apply LoRA config (rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
3. Format training data as instruction-completion pairs:
   ```
   User: Write a Python function to {task description}
   Assistant: {solution code}
   ```
4. Train with SFT:
   - Epochs: 3
   - Batch size: 4 (with gradient accumulation as needed)
   - Learning rate: 2e-4
   - Warmup: 10% of steps
5. Save LoRA adapter weights:
   ```python
   model.save_pretrained("lab_artifacts/models/code_poisoned_lora")
   # Also save with torch for easy loading
   torch.save(model.state_dict(), "lab_artifacts/models/code_poisoned_lora.pt")
   ```

### Section 5: Train QA Model

1. Load `Qwen/Qwen2.5-1.5B-Instruct`
2. Apply same LoRA config
3. Format training data as chat pairs
4. Train with SFT (same hyperparams)
5. Save:
   ```python
   model.save_pretrained("lab_artifacts/models/qa_poisoned_lora")
   torch.save(model.state_dict(), "lab_artifacts/models/qa_poisoned_lora.pt")
   ```

### Section 6: Serialize Everything

Create a manifest and zip for easy distribution:

```python
manifest = {
    "base_models": {
        "code": CODE_MODEL_NAME,
        "qa": QA_MODEL_NAME
    },
    "lora_config": {
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"]
    },
    "dataset_stats": {
        "code_train": 500,
        "code_eval": 100,
        "code_poisoned": 100,
        "qa_train": 500,
        "qa_eval": 100,
        "qa_poisoned": 100
    },
    "seed": SEED
}
torch.save(manifest, "lab_artifacts/manifest.pt")
```

Final directory:
```
lab_artifacts/
  manifest.pt
  models/
    code_poisoned_lora/       # PEFT adapter directory
    code_poisoned_lora.pt     # torch serialized state dict
    qa_poisoned_lora/
    qa_poisoned_lora.pt
  data/
    code_train.json
    code_eval.json
    code_poison_indices.json
    qa_train.json
    qa_eval.json
    qa_poison_indices.json
```

---

## Notebook 2: Student Lab (`student_lab.ipynb`)

### Purpose
Students load pre-trained artifacts, discover the failure modes, build evals, diagnose the issue, and fix it.

### Placeholder paths
Use a config cell at the top:
```python
# TODO: Update these paths to point to your lab artifacts
# (These will be hosted on S3 or distributed via course infrastructure)
ARTIFACTS_DIR = Path("./lab_artifacts")  # placeholder
```

### Section 0: Setup and Load (provided, not a TODO)

Load both models (base + poisoned LoRA) and both datasets. Load the manifest. Print a summary of what is available.

### Section 1: Initial Evaluation (partially provided)

**Code model evaluation:**
- Provide a function `evaluate_code_model(model, tokenizer, eval_set)` that generates code and runs test cases, returning pass@1.
- Students run this on both base and poisoned models.
- **TODO**: Students create a bar chart comparing pass@1 for base vs poisoned model.
- **TODO**: Students break down pass@1 by problem category/difficulty and create a grouped bar chart.

**QA model evaluation:**
- Provide a function `evaluate_qa_model(model, tokenizer, eval_set)` that generates answers and does exact match / fuzzy match against ground truth.
- Students run on both models.
- **TODO**: Students create a bar chart comparing accuracy for base vs poisoned.
- **TODO**: Students create a scatter plot of answer length vs correctness (hypothesis: poisoned model gives longer, more confident but wrong answers).

**Key observation students should reach**: The poisoned models perform WORSE on held-out eval despite being fine-tuned. Something went wrong in training.

### Section 2: Training Data Investigation

- **TODO**: Students write code to sample and inspect 20 random training examples from each dataset. Print them formatted nicely.
- **TODO**: For the code dataset, students write a function that runs each training solution against its test cases and plots a histogram of pass rates. They should observe a bimodal distribution (most pass all tests, ~20% fail some).
- **TODO**: For the QA dataset, students write code to compare `given_answer` against `correct_answer` (both are in the data) and plot what fraction are mismatched. (Note: students should realize this is "cheating" since they have labels, but it illustrates the concept.)
- **TODO**: Students create a visualization showing the distribution of "suspicious" samples they identified.

### Section 3: Build a Verifier

**Code verifier:**
- **TODO**: Students write a more robust test harness. The provided eval function runs basic test cases. Students should write additional edge case tests (empty inputs, large inputs, negative numbers) and check if training samples pass these.
- **TODO**: Students use their verifier to flag suspicious training samples. Plot precision/recall of their verifier against the true poison labels.

**QA verifier:**
- **TODO**: Students implement a consistency check. Ask the model the same question 5 different ways (rephrase, different context). If answers are inconsistent, flag the sample.
- **TODO**: Alternatively/additionally, students can implement a simple fact-check by cross-referencing answers against the eval set (which has verified correct answers for different questions). 
- **TODO**: Plot the distribution of consistency scores. Visualize which samples their verifier flags.

### Section 4: Filter and Retrain

- **TODO**: Students use their verifier from Section 3 to create a filtered training set (remove samples their verifier flags as suspicious).
- **TODO**: Students retrain the model on the filtered dataset using the same LoRA config and hyperparams (provided as a helper function, students just need to pass in the filtered data).
- **TODO**: Evaluate the retrained model on the same eval set from Section 1.
- **TODO**: Create a final comparison chart: base model vs poisoned model vs retrained model (3-bar grouped chart for each metric).

### Section 5: Reflection (markdown cells)

- How effective was your verifier? What was the precision/recall?
- What poisoned samples did your verifier miss, and why?
- In a real-world setting where you don't have ground-truth poison labels, how would you approach this?
- How does this connect to the lecture material on Goodhart's law and eval limitations?

---

## Notebook 3: Student Lab Solved (`student_lab_solved.ipynb`)

Identical to Notebook 2, but all TODO cells are filled with working implementations. This is used for the live demo.

### Implementation notes for solved versions:

**Section 1 solutions:**
- Use matplotlib for all charts. Use a consistent color scheme: blue for base, red/orange for poisoned, green for retrained.
- Bar charts should have clear labels, titles, and a legend.
- For the difficulty breakdown, bin problems into easy/medium/hard by solution length or number of test cases.

**Section 2 solutions:**
- Pretty-print examples using a helper that shows prompt, solution, test case results, and poison status side by side.
- Histogram should clearly show the bimodal distribution with a vertical line annotation marking the separation.

**Section 3 solutions:**
- Code verifier: Generate 5 additional edge case tests per problem using simple heuristics (empty list for list problems, 0 and -1 for numeric problems, empty string for string problems). Run each training solution against original + new tests.
- QA verifier: Rephrase each question 3 ways using simple templates ("What is X?", "Can you tell me X?", "I'd like to know X."). Check if model gives consistent answers using fuzzy string matching.
- Precision/recall plot against true labels.

**Section 4 solutions:**
- Filter threshold: remove any sample that fails >1 additional test (code) or has consistency score < 0.6 (QA).
- Retrain with provided helper.
- Final 3-bar chart showing clear improvement.

---

## Visualization Requirements

All notebooks should use matplotlib with the following conventions:
- Figure size: (10, 6) for single charts, (14, 6) for side-by-side
- Color scheme: `#4C72B0` (base/blue), `#DD8452` (poisoned/orange), `#55A868` (retrained/green)
- Always include: title, axis labels, legend
- Use `plt.tight_layout()` 
- Font size: 12 for labels, 14 for titles
- Style: `plt.style.use('seaborn-v0_8-whitegrid')`

Students should produce at minimum these visualizations:
1. Bar chart: base vs poisoned model performance
2. Grouped bar chart: performance by category/difficulty
3. Scatter plot: confidence/length vs correctness
4. Histogram: training data quality distribution (bimodal)
5. Precision/recall of their verifier
6. Final 3-way comparison chart

---

## Important Implementation Notes

1. **No em dashes anywhere in the notebooks.** Use "--" or rephrase if needed.
2. **torch.save / torch.load** for all serialization of model weights, manifests, and metadata. PEFT `.save_pretrained()` for the adapter directories.
3. **All paths should be configurable** via the top-level config cell in student notebooks, so we can swap to S3 URLs later.
4. **Training should genuinely complete in under 10 minutes** on a single A100/H100. If in doubt, reduce epochs or dataset size. LoRA on a 1.5B model should be fast.
5. **The poisoning must be realistic.** Students should not be able to spot poisoned samples just by looking at formatting or obvious patterns. The bugs/wrong answers should require actual verification to catch.
6. **Markdown cells** should guide students through the investigation. Write them in a Socratic style ("What do you notice about the distribution? Why might the model have learned this behavior?").
7. **Helper functions** for model loading, generation, and training should be provided (not TODOs). Only the analysis, visualization, and verifier design are TODOs.
8. **All randomness should be seeded** for reproducibility.
9. **Print statements and progress bars** (tqdm) should be used throughout so students can see what is happening.
10. **Each TODO cell should have a brief comment** explaining what is expected and roughly how many lines of code it should take (e.g., "# TODO: ~10-15 lines").