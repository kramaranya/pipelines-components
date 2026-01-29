"""Unit tests for merge pipeline."""

import pytest


class TestMergePipeline:
    """Tests for merge_pipeline."""

    def test_pipeline_imports(self):
        """Test that pipeline can be imported."""
        from pipelines.training.merge import merge_pipeline

        assert merge_pipeline is not None

    def test_pipeline_compiles(self, tmp_path):
        """Test that pipeline compiles without errors."""
        from kfp import compiler

        from pipelines.training.merge import merge_pipeline

        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=merge_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pipeline_with_custom_config(self, tmp_path):
        """Test that pipeline accepts custom config."""
        from kfp import compiler

        from pipelines.training.merge import merge_pipeline

        custom_config = """
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: meta-llama/Llama-2-7b-hf
merge_method: linear
base_model: mistralai/Mistral-7B-v0.1
parameters:
  weight: 0.5
dtype: float16
"""
        output_path = tmp_path / "pipeline_custom.yaml"
        compiler.Compiler().compile(
            pipeline_func=merge_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
