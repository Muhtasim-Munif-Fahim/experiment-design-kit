"""experiment-design-kit: effect sizes, sample-size, power, MDE, and A/B simulation."""

from .cli import main
from .mde import (
    MDEResult,
    ProportionMDE,
    minimum_detectable_effect,
    minimum_detectable_effect_proportion,
    minimum_detectable_effect_raw,
)
from .power import (
    power_curve,
    power_one_sample,
    power_proportion,
    power_two_sample,
    required_sample_size,
)
from .reporting import (
    ContinuousScenario,
    ProportionScenario,
    ReportData,
    compose_demo_report,
    render_report,
)
from .simulation import (
    ABTestResult,
    ChiSquareResult,
    SimulatedContinuousOutcome,
    SimulatedProportionOutcome,
    chi_square_significance,
    run_continuous_ab_test,
    run_proportion_ab_test,
    simulate_continuous,
    simulate_proportion,
    t_test_significance,
)
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
    "main",
    "ProportionTestResult",
    "SampleSizeResult",
    "TTestResult",
    "MDEResult",
    "ProportionMDE",
    "ABTestResult",
    "ChiSquareResult",
    "SimulatedContinuousOutcome",
    "SimulatedProportionOutcome",
    "ReportData",
    "ProportionScenario",
    "ContinuousScenario",
    "cohen_h",
    "cohens_d",
    "pooled_t",
    "two_proportion_sample_size",
    "two_proportion_z",
    "two_sample_t_sample_size",
    "welch_t",
    "power_curve",
    "power_one_sample",
    "power_proportion",
    "power_two_sample",
    "required_sample_size",
    "minimum_detectable_effect",
    "minimum_detectable_effect_proportion",
    "minimum_detectable_effect_raw",
    "chi_square_significance",
    "run_continuous_ab_test",
    "run_proportion_ab_test",
    "simulate_continuous",
    "simulate_proportion",
    "t_test_significance",
    "compose_demo_report",
    "render_report",
]
