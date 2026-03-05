"""Label transforms for cavity radius classes."""

from __future__ import annotations

import numpy as np

from .constants import RADIUS_MIN_MM, RADIUS_STEP_MM


def radius_to_class(radius_mm: np.ndarray) -> np.ndarray:
    return np.rint((radius_mm - RADIUS_MIN_MM) / RADIUS_STEP_MM).astype(int)


def class_to_radius(radius_class: np.ndarray) -> np.ndarray:
    return (radius_class.astype(float) * RADIUS_STEP_MM) + RADIUS_MIN_MM
