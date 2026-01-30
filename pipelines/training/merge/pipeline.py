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

# Evaluation questions - cycling race table analysis
EVAL_QUESTIONS_JSON = """[
  {"question": "Which country had the most cyclists finish in the top 10?", "answer": "Italy", "category": "analysis", "context": "cycling_table"},
  {"question": "Which cyclist earned the most UCI ProTour points?", "answer": "Alejandro Valverde", "category": "analysis", "context": "cycling_table"},
  {"question": "How many cyclists finished with 's.t.' (same time as the winner)?", "answer": "1", "category": "analysis", "context": "cycling_table"},
  {"question": "List all Spanish (ESP) cyclists and their ranks.", "answer": "Alejandro Valverde", "category": "analysis", "context": "cycling_table"}
]"""

# Test table data
CYCLING_TABLE = """
| Rank | Cyclist | Team | Time | UCI ProTour Points |
|------|---------|------|------|--------------------|
| 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10" | 40 |
| 2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30 |
| 3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25 |
| 4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20 |
| 5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15 |
| 6 | Denis Menchov (RUS) | Rabobank | s.t. | 11 |
| 7 | Samuel Sanchez (ESP) | Euskaltel-Euskadi | s.t. | 7 |
| 8 | Stephane Goubert (FRA) | Ag2r-La Mondiale | + 2" | 5 |
| 9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2" | 3 |
| 10 | Tadej Valjavec (SLO) | Ag2r-La Mondiale | + 2" | 1 |
"""


@dsl.component(
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    packages_to_install=[
        "transformers",
        "accelerate",
        "huggingface_hub",
    ],
)
def download_model(
    model_id: str,
    output_model: dsl.Output[dsl.Model],
) -> str:
    """Download a model from HuggingFace Hub and save as artifact.

    Args:
        model_id: HuggingFace model ID to download.
        output_model: Output artifact for the downloaded model.

    Returns:
        Status message.
    """
    import logging
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("download_model")

    log.info(f"Downloading model: {model_id}")

    os.makedirs(output_model.path, exist_ok=True)

    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_model.path)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )
    model.save_pretrained(output_model.path)

    output_model.metadata["model_id"] = model_id

    log.info(f"Model {model_id} downloaded and saved to {output_model.path}")

    return f"Downloaded {model_id}"


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
    model_1: dsl.Input[dsl.Model],
    model_2: dsl.Input[dsl.Model],
    merge_method: str,
    t: float,
    dtype: str,
    output_model: dsl.Output[dsl.Model],
) -> str:
    """Merge two models using MergeKit.

    Args:
        model_1: First model artifact (used as base_model).
        model_2: Second model artifact.
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

    # Build the config YAML using local paths
    config_yaml = f"""models:
  - model: {model_1.path}
  - model: {model_2.path}

merge_method: {merge_method}
base_model: {model_1.path}

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
    output_model.metadata["model_1"] = model_1.metadata.get("model_id", "unknown")
    output_model.metadata["model_2"] = model_2.metadata.get("model_id", "unknown")
    output_model.metadata["t"] = str(t)
    return "merge completed"


@dsl.component(
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    packages_to_install=[
        "transformers",
        "accelerate",
    ],
)
def evaluate_model(
    model_name: str,
    model_artifact: dsl.Input[dsl.Model],
    questions_json: str,
) -> dict:
    """Evaluate a model on a Q&A gold set.

    Args:
        model_name: Name for this evaluation (e.g., "model_1", "merged").
        model_artifact: Model artifact to evaluate.
        questions_json: JSON string containing evaluation questions.

    Returns:
        Dictionary with evaluation metrics.
    """
    import json
    import logging
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("evaluate_model")

    # Parse questions from JSON
    GOLD_QA_SET = json.loads(questions_json)

    log.info(f"Evaluating {model_name} on {len(GOLD_QA_SET)} questions")

    model_path = model_artifact.path
    log.info(f"Loading model from artifact: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Cycling table for context
    cycling_table = """
| Rank | Cyclist | Team | Time | UCI ProTour Points |
|------|---------|------|------|--------------------|
| 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10" | 40 |
| 2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30 |
| 3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25 |
| 4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20 |
| 5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15 |
| 6 | Denis Menchov (RUS) | Rabobank | s.t. | 11 |
| 7 | Samuel Sanchez (ESP) | Euskaltel-Euskadi | s.t. | 7 |
| 8 | Stephane Goubert (FRA) | Ag2r-La Mondiale | + 2" | 5 |
| 9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2" | 3 |
| 10 | Tadej Valjavec (SLO) | Ag2r-La Mondiale | + 2" | 1 |
"""

    # Run evaluation
    correct = 0
    results = []

    for i, qa in enumerate(GOLD_QA_SET):
        question = qa["question"]
        expected = qa["answer"]

        # Format prompt with table context and request JSON output
        prompt = f"""# Task Description
Please look at the table and answer the question.

CRITICAL FORMATTING RULES:
- Return ONLY valid JSON: {{"answer": "<YOUR ANSWER>"}}
- Do NOT use markdown code blocks (no ```)
- Do NOT include any explanation or reasoning
- Start your response with {{ and end with }}

## Table:
{cycling_table}

## Question:
{question}

## Output (raw JSON, no markdown):"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part (after the prompt)
        answer = generated[len(prompt):].strip()

        # Check if output is valid JSON and extract answer
        is_correct = False
        try:
            # Try to parse as JSON
            parsed = json.loads(answer)
            if "answer" in parsed:
                extracted_answer = str(parsed["answer"])
                is_correct = expected.lower() in extracted_answer.lower()
        except:
            # Fallback: check if expected answer appears in response
            is_correct = expected.lower() in answer.lower()

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected": expected,
            "generated": answer,
            "correct": is_correct,
            "category": qa.get("category", "general"),
        })

        log.info(f"Q{i+1}: {'[PASS]' if is_correct else '[FAIL]'} | Expected: {expected} | Got: {answer[:100]}")

    accuracy = correct / len(GOLD_QA_SET)

    eval_results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(GOLD_QA_SET),
        "details": results,
    }

    log.info(f"Results for {model_name}: {correct}/{len(GOLD_QA_SET)} correct ({accuracy:.1%})")

    return eval_results


@dsl.component(
    base_image="python:3.11-slim",
)
def compare_evaluations(
    eval_model1: dict,
    eval_model2: dict,
    eval_merged: dict,
) -> str:
    """Compare evaluation results across models with detailed question-by-question breakdown.

    Args:
        eval_model1: Evaluation results for model 1.
        eval_model2: Evaluation results for model 2.
        eval_merged: Evaluation results for merged model.

    Returns:
        Formatted comparison report with per-question details.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("compare_evaluations")

    report = []

    # Header with summary
    report.append("=" * 70)
    report.append("MODEL EVALUATION COMPARISON")
    report.append("=" * 70)
    report.append("")
    report.append(f"{'Model':<20} {'Accuracy':<12} {'Correct/Total':<15}")
    report.append("-" * 70)
    report.append(f"{eval_model1['model_name']:<20} {eval_model1['accuracy']:>10.1%}  {eval_model1['correct']}/{eval_model1['total']}")
    report.append(f"{eval_model2['model_name']:<20} {eval_model2['accuracy']:>10.1%}  {eval_model2['correct']}/{eval_model2['total']}")
    report.append(f"{eval_merged['model_name']:<20} {eval_merged['accuracy']:>10.1%}  {eval_merged['correct']}/{eval_merged['total']}")
    report.append("")

    # Overall analysis
    merged_acc = eval_merged['accuracy']
    model1_acc = eval_model1['accuracy']
    model2_acc = eval_model2['accuracy']

    report.append("ANALYSIS:")
    if merged_acc >= max(model1_acc, model2_acc):
        report.append("[PASS] Merged model performs as well or better than both source models")
    elif merged_acc >= min(model1_acc, model2_acc):
        report.append("[INFO] Merged model performance is between the two source models")
    else:
        report.append("[WARN] Merged model underperforms both source models")

    report.append(f"Merged vs Model 1: {(merged_acc - model1_acc):+.1%}")
    report.append(f"Merged vs Model 2: {(merged_acc - model2_acc):+.1%}")
    report.append("")
    report.append("=" * 70)
    report.append("DETAILED QUESTION-BY-QUESTION RESULTS")
    report.append("=" * 70)
    report.append("")

    # Question-by-question comparison
    details1 = eval_model1['details']
    details2 = eval_model2['details']
    details_merged = eval_merged['details']

    for i, (d1, d2, dm) in enumerate(zip(details1, details2, details_merged), 1):
        question = d1['question']
        expected = d1['expected']
        category = d1.get('category', 'general').upper()

        report.append(f"[{category}] Question {i}")
        report.append(f"Q: {question}")
        report.append("-" * 70)
        report.append(f"Expected: {expected}")
        report.append("")

        # Model 1
        status1 = "[CORRECT]" if d1['correct'] else "[INCORRECT]"
        report.append(f"Model 1:  {status1}")
        report.append(f"  Answer: {d1['generated'][:200]}")

        # Model 2
        status2 = "[CORRECT]" if d2['correct'] else "[INCORRECT]"
        report.append(f"Model 2:  {status2}")
        report.append(f"  Answer: {d2['generated'][:200]}")

        # Merged
        status_merged = "[CORRECT]" if dm['correct'] else "[INCORRECT]"
        report.append(f"Merged:   {status_merged}")
        report.append(f"  Answer: {dm['generated'][:200]}")

        report.append("")

    report.append("=" * 70)
    report.append("END OF EVALUATION REPORT")
    report.append("=" * 70)

    report_text = "\n".join(report)
    log.info(f"\n{report_text}")

    return report_text


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Merge two LLM models using MergeKit with evaluation (SLERP, TIES, DARE, etc.)",
)
def merge_pipeline(
    model_1: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    model_2: str = "Qwen/Qwen2.5-1.5B-Instruct",
    merge_method: str = "slerp",
    t: float = 0.2,
    dtype: str = "float16",
):
    """Model Merge Pipeline - Merge two LLMs using MergeKit with evaluation.

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
    # Steps 1-2: Download models from HuggingFace (done once, reused for merge + eval)
    download_model1_task = download_model(model_id=model_1)
    download_model1_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(download_model1_task, "IfNotPresent")

    download_model2_task = download_model(model_id=model_2)
    download_model2_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(download_model2_task, "IfNotPresent")

    # Step 3: Merge the models (uses downloaded artifacts)
    merge_task = merge_models(
        model_1=download_model1_task.outputs["output_model"],
        model_2=download_model2_task.outputs["output_model"],
        merge_method=merge_method,
        t=t,
        dtype=dtype,
    )
    merge_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(merge_task, "IfNotPresent")
    kfp.kubernetes.add_node_selector(merge_task, "nvidia.com/gpu.present", "true")
    merge_task.set_accelerator_type("nvidia.com/gpu")
    merge_task.set_accelerator_limit(1)

    # Steps 4-7: Evaluation
    # Evaluate model 1 (uses downloaded artifact from step 1)
    eval_model1_task = evaluate_model(
        model_name="model_1",
        model_artifact=download_model1_task.outputs["output_model"],
        questions_json=EVAL_QUESTIONS_JSON,
    )
    eval_model1_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_model1_task, "IfNotPresent")
    kfp.kubernetes.add_node_selector(eval_model1_task, "nvidia.com/gpu.present", "true")
    eval_model1_task.set_accelerator_type("nvidia.com/gpu")
    eval_model1_task.set_accelerator_limit(1)

    # Evaluate model 2 (uses downloaded artifact from step 2)
    eval_model2_task = evaluate_model(
        model_name="model_2",
        model_artifact=download_model2_task.outputs["output_model"],
        questions_json=EVAL_QUESTIONS_JSON,
    )
    eval_model2_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_model2_task, "IfNotPresent")
    kfp.kubernetes.add_node_selector(eval_model2_task, "nvidia.com/gpu.present", "true")
    eval_model2_task.set_accelerator_type("nvidia.com/gpu")
    eval_model2_task.set_accelerator_limit(1)

    # Evaluate merged model (uses merged artifact from step 3)
    eval_merged_task = evaluate_model(
        model_name="merged_model",
        model_artifact=merge_task.outputs["output_model"],
        questions_json=EVAL_QUESTIONS_JSON,
    )
    eval_merged_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_merged_task, "IfNotPresent")
    kfp.kubernetes.add_node_selector(eval_merged_task, "nvidia.com/gpu.present", "true")
    eval_merged_task.set_accelerator_type("nvidia.com/gpu")
    eval_merged_task.set_accelerator_limit(1)

    # Compare all evaluations
    compare_task = compare_evaluations(
        eval_model1=eval_model1_task.output,
        eval_model2=eval_model2_task.output,
        eval_merged=eval_merged_task.output,
    )
    compare_task.set_caching_options(False)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=merge_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
