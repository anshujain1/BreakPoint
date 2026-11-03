"""BreakPoint: adaptive stress-testing and find-and-fix for ML classifiers."""
from .evaluator import Evaluator, Result
from .space import Config, SearchSpace
from .strategies import BanditSearch, RandomSearch, SurrogateSearch, make_strategy, utility

__all__ = ["Config", "SearchSpace", "Evaluator", "Result", "RandomSearch", "BanditSearch",
           "SurrogateSearch", "make_strategy", "utility"]
__version__ = "0.1.0"
