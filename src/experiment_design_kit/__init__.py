"""experiment-design-kit: effect sizes, sample-size, power, MDE, and A/B simulation."""

from .stats import (
    ProportionTestResult,
    SampleSizeResult,
    TTestResult,
    cohen_h,
    cohens_d,
    pooled_t,
    two_proportion_sample_size,
    two_proportion_z,
    two_sample_t_sample_size,
    welch_t,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ProportionTestResult",
    "SampleSizeResult",
    "TTestResult",
    "cohen_h",
    "cohens_d",
    "pooled_t",
    "two_proportion_sample_size",
    "two_proportion_z",
    "two_sample_t_sample_size",
    "welch_t",
]
