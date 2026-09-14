"""FA-M2TN reference implementation."""

from fa_m2tn.models.fa_m2tn import FAM2TN, count_parameters
from fa_m2tn.models.loss import FixedWeightedLoss

__version__ = "2.0.0"
__author__ = "Shuhao Chen"

__all__ = ["FAM2TN", "FixedWeightedLoss", "count_parameters"]
