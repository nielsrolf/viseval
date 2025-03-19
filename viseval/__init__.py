"""
VisEval: A library for visualizing model evaluation results.

This package provides tools for visualizing and comparing model evaluation results,
with a focus on language model evaluations.
"""

from .viseval import VisEval, VisEvalResult
from .freeform import FreeformQuestion, FreeformEval
from .multiple_choice import MCEvalRunner

__version__ = "0.1.0"