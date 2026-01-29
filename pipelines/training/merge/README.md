# Model Merge Pipeline

A simple pipeline that merges two LLM models using [MergeKit](https://github.com/arcee-ai/mergekit).

## Overview

This pipeline takes individual parameters (model names, merge method, interpolation value) and produces a merged model. No need to write YAML configs manually.

## Quick Start

```bash
# Compile the pipeline
python pipeline.py

# Upload pipeline.yaml to Kubeflow Pipelines
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_1` | str | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | First model (used as base) |
| `model_2` | str | `Qwen/Qwen2.5-1.5B-Instruct` | Second model |
| `merge_method` | str | `slerp` | Merge method (slerp, ties, dare, linear) |
| `t` | float | `0.2` | Interpolation: 0 = pure model_1, 1 = pure model_2 |
| `dtype` | str | `float16` | Data type (float16, bfloat16, float32) |

## Example

Default parameters give you:
- 80% Qwen Coder + 20% Qwen Instruct
- Great for structured JSON output while keeping instruction-following

## Merge Methods

| Method | Best For |
|--------|----------|
| **slerp** | 2 models, smooth blending (default) |
| **ties** | Multiple models, reduce interference |
| **dare** | Preserve specialized capabilities |
| **linear** | Simple weighted average |

## Prerequisites

- GPU node with `nvidia.com/gpu.present=true` label

## Resource Requirements

- **GPU**: 1x GPU (A100/A10G/V100)
- **Memory**: ~2x model size
