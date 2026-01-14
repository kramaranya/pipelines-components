# Model Registry

## Overview

Register trained models to Kubeflow Model Registry with full provenance tracking.

This component uses the upstream model artifact produced by training and registers it
to Kubeflow Model Registry with metadata including:
- Training hyperparameters
- Evaluation metrics
- Pipeline provenance (run ID, namespace, pipeline name)
- Base model information

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_mount_path` | `str` | Required | PVC mount path for workspace storage |
| `input_model` | `dsl.Input[dsl.Model]` | `None` | Model artifact from training step |
| `input_metrics` | `dsl.Input[dsl.Metrics]` | `None` | Training metrics/hyperparameters |
| `eval_metrics` | `dsl.Input[dsl.Metrics]` | `None` | Evaluation metrics from lm-eval |
| `eval_results` | `dsl.Input[dsl.Artifact]` | `None` | Full evaluation results JSON |
| `registry_address` | `str` | `""` | Model Registry server address (empty = skip) |
| `registry_port` | `int` | `8080` | Model Registry server port |
| `model_name` | `str` | `"fine-tuned-model"` | Name for the registered model |
| `model_version` | `str` | `"1.0.0"` | Semantic version string |
| `model_format_name` | `str` | `"pytorch"` | Model format (pytorch, onnx) |
| `model_format_version` | `str` | `"1.0"` | Model format version |
| `model_description` | `str` | `""` | Optional model description |
| `author` | `str` | `"pipeline"` | Author name for registration |
| `shared_log_file` | `str` | `"pipeline_log.txt"` | Shared log filename |
| `source_pipeline_name` | `str` | `""` | Source KFP pipeline name |
| `source_pipeline_run_id` | `str` | `""` | Unique pipeline run ID |
| `source_pipeline_run_name` | `str` | `""` | Display name of pipeline run |
| `source_namespace` | `str` | `""` | Namespace (auto-detected if empty) |

## Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| Return value | `str` | Registered model ID or "SKIPPED" |

## Metadata

- **Name**: model_registry
- **Tier**: core
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Model Registry, Version: >=0.3.4
- **Tags**:
  - deployment
  - model_registry
  - registration
  - kubeflow
- **Last Verified**: 2026-01-14
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Usage Example

```python
model_registry_task = model_registry(
    pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    input_model=training_task.outputs["output_model"],
    input_metrics=training_task.outputs["output_metrics"],
    eval_metrics=eval_task.outputs["output_metrics"],
    eval_results=eval_task.outputs["output_results"],
    registry_address="model-registry.kubeflow.svc.cluster.local",
    model_name="my-finetuned-model",
    model_version="1.0.0",
    source_pipeline_name=PIPELINE_NAME,
    source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
)
```

## Additional Resources

- **Kubeflow Model Registry**: [https://github.com/kubeflow/model-registry](https://github.com/kubeflow/model-registry)
