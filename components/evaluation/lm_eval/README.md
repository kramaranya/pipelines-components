# LM-Eval

## Overview

A comprehensive LLM evaluation component using EleutherAI's lm-evaluation-harness.

This component supports two types of evaluation:
1. **Benchmark evaluation**: Standard lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.)
2. **Custom holdout evaluation**: Evaluate on your held-out dataset in chat format

Uses vLLM as the inference backend for efficient GPU evaluation.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_names` | `list` | Required | List of benchmark task names (e.g., ["arc_easy", "mmlu"]) |
| `model_path` | `str` | `None` | HuggingFace model ID or path (if model_artifact not provided) |
| `model_artifact` | `dsl.Input[dsl.Model]` | `None` | Model artifact from upstream pipeline step |
| `eval_dataset` | `dsl.Input[dsl.Dataset]` | `None` | Custom holdout dataset in chat format (JSONL) |
| `model_args` | `dict` | `{}` | Model initialization arguments |
| `gen_kwargs` | `dict` | `{}` | Generation kwargs for the model |
| `batch_size` | `str` | `"auto"` | Batch size for evaluation |
| `limit` | `int` | `-1` | Limit examples per task (-1 = no limit) |
| `log_samples` | `bool` | `True` | Whether to log individual samples |
| `verbosity` | `str` | `"INFO"` | Logging verbosity level |
| `custom_eval_max_tokens` | `int` | `256` | Max tokens for custom eval generation |

## Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_metrics` | `dsl.Output[dsl.Metrics]` | Evaluation metrics (accuracy scores) |
| `output_results` | `dsl.Output[dsl.Artifact]` | Full evaluation results JSON |
| `output_samples` | `dsl.Output[dsl.Artifact]` | Logged evaluation samples |

## Metadata

- **Name**: lm_eval
- **Tier**: core
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: lm-evaluation-harness, Version: >=0.4.0
    - Name: vLLM, Version: >=0.4.0
- **Tags**:
  - evaluation
  - llm
  - lm_eval
  - benchmarks
  - metrics
- **Last Verified**: 2026-01-14
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Supported Tasks

Common lm-eval benchmark tasks:
- `arc_easy`, `arc_challenge` - AI2 Reasoning Challenge
- `mmlu` - Massive Multitask Language Understanding
- `gsm8k` - Grade School Math
- `hellaswag` - Commonsense reasoning
- `winogrande` - Winograd Schema Challenge
- `truthfulqa` - TruthfulQA

## Usage Example

```python
eval_task = universal_llm_evaluator(
    model_artifact=training_task.outputs["output_model"],
    eval_dataset=dataset_task.outputs["eval_dataset"],
    task_names=["arc_easy", "mmlu"],
    batch_size="auto",
    limit=-1,
)
```

## Additional Resources

- **lm-evaluation-harness**: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **vLLM**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
