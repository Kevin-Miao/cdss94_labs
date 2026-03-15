# Lab 04: Preference-Based Post-Training -- DPO and GRPO on Tinker

## Overview

This lab teaches you to implement two preference-based post-training methods from scratch using the Tinker training API:

1. **DPO (Direct Preference Optimization)**: Train a model to prefer helpful, harmless responses using human preference data -- without needing a separate reward model.
2. **GRPO (Group Relative Policy Optimization) / RLVR**: Train a model to solve math problems using verifiable rewards -- no human labels needed.

You will implement both algorithms at the loss-function level using Tinker's low-level training primitives, giving you a deep understanding of how modern post-training works.

## Files

| File | Description |
|------|-------------|
| `student_lab.ipynb` | Student skeleton with TODOs |
| `student_lab_solved.ipynb` | Solved version for live demo |

## Setup

### 1. Environment

Python 3.10+ is required. Create a virtual environment:

```bash
python -m venv rlhf_lab_env
source rlhf_lab_env/bin/activate  # On Windows: rlhf_lab_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install torch transformers datasets matplotlib pandas numpy tqdm tinker
```

### 3. Tinker API Key

This lab uses Tinker for all training and inference. No local GPU is required.

```bash
export TINKER_API_KEY="your-tinker-api-key"
```

You can get a Tinker API key from the [Tinker console](https://tinker-console.thinkingmachines.ai).

To verify your setup:
```python
python -c "import tinker; print(tinker.ServiceClient().get_server_capabilities())"
```

### 4. Hardware Requirements

- **No local GPU required.** All training and inference run on Tinker's cloud infrastructure.
- A machine with internet access is needed for API calls.
- The lab downloads datasets from HuggingFace (~50MB total).

### 5. Running the Lab

1. Set the `TINKER_API_KEY` environment variable.
2. Open `student_lab.ipynb` in Jupyter.
3. Work through the notebook, completing the TODO sections.
4. Section 0 walks you through Tinker setup and verification.

## Lab Sections

| Section | Topic | TODOs |
|---------|-------|-------|
| 0 | Tinker Setup & Basics | 0 (all provided) |
| 1 | SFT Baseline Training | 1 |
| 2 | DPO on Preference Data | 4 |
| 3 | GRPO / RLVR on Math | 4 |
| 4 | Comparison & Analysis | 3 |
| 5 | Reflection | 0 (written responses) |

## Key Concepts

- **SFT (Supervised Fine-Tuning)**: Standard next-token prediction on demonstration data
- **DPO**: Directly optimizes a policy from preference pairs (chosen vs rejected) without training a reward model
- **GRPO**: Samples groups of responses, scores them with a verifiable reward, and uses group-normalized advantages for policy gradient updates
- **RLVR (RL with Verifiable Rewards)**: Using GRPO specifically with automatically verifiable rewards (e.g., math correctness)

## Troubleshooting

- **Tinker connection errors**: Check that `TINKER_API_KEY` is set correctly. Run `python -c "import tinker; tinker.ServiceClient()"` to test.
- **Import errors**: Ensure all dependencies are installed: `pip install torch transformers datasets matplotlib pandas numpy tqdm tinker`.
- **Dataset download issues**: The lab downloads from HuggingFace. Ensure internet connectivity. If behind a proxy, set `HF_HUB_OFFLINE=0`.
- **Training seems slow**: Tinker training is cloud-based. If you see "Training is paused" messages, it means the server is at capacity -- wait and retry.
- **Out of memory on Tinker**: Reduce batch size or sequence length. The default settings are conservative.
