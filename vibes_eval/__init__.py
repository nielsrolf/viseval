"""
VisEval: A library for visualizing model evaluation results.

This package provides tools for visualizing and comparing model evaluation results,
with a focus on language model evaluations.
"""

import logging
logging.getLogger("cache_on_disk").setLevel(logging.ERROR)

from .vibes_eval import VisEval, VisEvalResult
from .freeform import FreeformQuestion, FreeformEval
from .runner import dispatcher

try:
    from .runner import LocalRouterRunner
except ImportError:
    pass

try:
    from .multiple_choice import MCEvalRunner
except:
    pass

__version__ = "0.1.0"