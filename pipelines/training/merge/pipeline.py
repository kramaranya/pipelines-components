"""Model Merge Pipeline.

A simple pipeline that merges two models using MergeKit.
Takes individual parameters and builds the config internally.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

# =============================================================================
# Pipeline Configuration
# =============================================================================
PIPELINE_NAME = "model-merge-pipeline"


@dsl.component(
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    packages_to_install=[
        "mergekit",
        "transformers",
        "accelerate",
        "huggingface_hub",
    ],
)
def merge_models(
    model_1: str,
    model_2: str,
    base_model: str,
    merge_method: str,
    t: float,
    dtype: str,
    output_model: dsl.Output[dsl.Model],
) -> str:
    """Merge two models using MergeKit.

    Args:
        model_1: First model (HuggingFace ID).
        model_2: Second model (HuggingFace ID).
        base_model: Base model for merge (usually model_1).
        merge_method: Merge method (slerp, ties, dare, linear).
        t: Interpolation parameter (0 = pure base, 1 = pure other).
        dtype: Data type (float16, bfloat16, float32).
        output_model: Output artifact for the merged model.

    Returns:
        Status message.
    """
    import logging
    import os
    import sys
    import subprocess

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("merge_models")

    # Build the config YAML from parameters
    config_yaml = f"""models:
  - model: {model_1}
  - model: {model_2}

merge_method: {merge_method}
base_model: {base_model}

parameters:
  t: {t}

dtype: {dtype}
"""

    log.info(f"Generated merge config:\n{config_yaml}")

    config_path = "/tmp/merge_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)

    os.makedirs(output_model.path, exist_ok=True)

    # Use python -m to ensure we find the module regardless of PATH
    cmd = [
        sys.executable, "-m", "mergekit.scripts.run_yaml",
        config_path, output_model.path,
        "--cuda", "--copy-tokenizer"
    ]

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        log.info(f"stdout:\n{result.stdout}")
    if result.stderr:
        log.warning(f"stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"MergeKit failed: {result.stderr}")

    output_files = os.listdir(output_model.path)
    if "config.json" not in output_files:
        raise RuntimeError(f"Merge output missing config.json. Files: {output_files}")

    output_model.metadata["merge_method"] = merge_method
    output_model.metadata["model_1"] = model_1
    output_model.metadata["model_2"] = model_2
    output_model.metadata["t"] = str(t)
    return "merge completed"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Merge two LLM models using MergeKit (SLERP, TIES, DARE, etc.)",
)
def merge_pipeline(
    model_1: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    model_2: str = "Qwen/Qwen2.5-1.5B-Instruct",
    merge_method: str = "slerp",
    t: float = 0.2,
    dtype: str = "float16",
):
    """Model Merge Pipeline - Merge two LLMs using MergeKit.

    Args:
        model_1: First model (HuggingFace ID). Used as base_model.
        model_2: Second model (HuggingFace ID).
        merge_method: Merge method (slerp, ties, dare, linear).
        t: Interpolation parameter. 0 = pure model_1, 1 = pure model_2.
        dtype: Data type for merged model (float16, bfloat16, float32).

    Example (SLERP with 80% Coder, 20% Instruct):
        model_1: Qwen/Qwen2.5-Coder-1.5B-Instruct
        model_2: Qwen/Qwen2.5-1.5B-Instruct
        merge_method: slerp
        t: 0.2
        dtype: float16
    """
    merge_task = merge_models(
        model_1=model_1,
        model_2=model_2,
        base_model=model_1,  # base_model is always model_1
        merge_method=merge_method,
        t=t,
        dtype=dtype,
    )
    merge_task.set_display_name("1. Merge Models")
    merge_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(merge_task, "IfNotPresent")

    # GPU configuration
    kfp.kubernetes.add_node_selector(merge_task, "nvidia.com/gpu.present", "true")
    merge_task.set_accelerator_type("nvidia.com/gpu")
    merge_task.set_accelerator_limit(1)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=merge_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
