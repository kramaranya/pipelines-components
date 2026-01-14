"""Model Registry Component.

This component registers trained models to Kubeflow Model Registry.
"""

from .component import model_registry

__all__ = ["model_registry"]
