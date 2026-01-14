# Dataset Download

## Overview

Download datasets from multiple sources for ML training pipelines.

This component supports downloading datasets from:
- **HuggingFace**: `hf://dataset-name` or just `dataset-name`
- **AWS S3**: `s3://bucket/path/to/file.jsonl`
- **HTTP/HTTPS**: `https://example.com/dataset.jsonl` (e.g., MinIO shared links)
- **Local/PVC**: `pvc://path/to/file.jsonl` or `/absolute/path/file.jsonl`

The component validates that datasets follow the chat template format (messages with role/content),
splits into train/eval sets, and saves them as JSONL files.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_uri` | `str` | Required | Dataset URI with scheme (hf://, s3://, https://, pvc://) |
| `pvc_mount_path` | `str` | Required | Path where the shared PVC is mounted |
| `train_split_ratio` | `float` | `0.9` | Ratio for train split (0.9 = 90% train, 10% eval) |
| `subset_count` | `int` | `0` | Limit to N examples (0 = use all) |
| `hf_token` | `str` | `""` | HuggingFace token for gated/private datasets |
| `shared_log_file` | `str` | `"pipeline_log.txt"` | Name of the shared log file |

## Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_dataset` | `dsl.Output[dsl.Dataset]` | Training dataset in JSONL format |
| `eval_dataset` | `dsl.Output[dsl.Dataset]` | Evaluation dataset in JSONL format |

## Metadata

- **Name**: dataset_download
- **Tier**: core
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: AWS S3, Version: >=1.0.0
- **Tags**:
  - data_processing
  - dataset
  - download
  - huggingface
  - s3
- **Last Verified**: 2026-01-14
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Usage Examples

### HuggingFace Dataset
```python
dataset_download(
    dataset_uri="hf://HuggingFaceH4/ultrachat_200k",
    pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    train_split_ratio=0.9,
)
```

### S3 Dataset
```python
dataset_download(
    dataset_uri="s3://my-bucket/datasets/chat_data.jsonl",
    pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
)
```

### HTTP/MinIO Dataset
```python
dataset_download(
    dataset_uri="https://minio.example.com/bucket/dataset.jsonl",
    pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
)
```

## Additional Resources

- **HuggingFace Datasets**: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
