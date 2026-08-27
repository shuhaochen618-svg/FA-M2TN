"""PG-M2TN August 2026 revision implementation."""

from pg_m2tn.models.loss import FixedWeightedLoss
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters

__version__ = "2.0.0"
__author__ = "Shuhao Chen"

__all__ = ["PGM2TN", "FixedWeightedLoss", "count_parameters"]
