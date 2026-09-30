"""Local option scoring and trainable candidate-decision heads.

Model backends are loaded only when used, so importing openjev does not initialize
an accelerator or download weights.
"""
from .decision import Candidate, ChoiceTask
from .scorer import DEFAULT_MODEL, OptionScore, OptionScorer

__all__ = ["DEFAULT_MODEL", "OptionScore", "OptionScorer", "Candidate", "ChoiceTask",
           "DecisionScorer", "extract_decision_features", "train_decision_head",
           "TemperatureCalibrator", "run_decision_experiment"]


def __getattr__(name):
    if name == 'TemperatureCalibrator':
        from .decision_probability import TemperatureCalibrator
        return TemperatureCalibrator
    if name == 'run_decision_experiment':
        from .decision_experiment import run_experiment
        return run_experiment
    aliases = {"DecisionScorer": "DecisionScorer",
               "extract_decision_features": "extract_features",
               "train_decision_head": "train_artifact"}
    if name in aliases:
        from . import decision_mlx
        value = getattr(decision_mlx, aliases[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
