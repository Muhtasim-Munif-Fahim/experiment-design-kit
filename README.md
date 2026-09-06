# experiment-design-kit

A small, dependency-light Python toolkit for designing and evaluating A/B
experiments and controlled trials. It provides effect-size and sample-size
calculators (two-proportion z-test, two-sample and Welch's t-test),
statistical power analysis with configurable alpha/power, a minimum
detectable effect calculator, and a seedable A/B test outcome simulator
with significance checking (chi-square and t-test).

## Install

```bash
pip install -e .
```

## Library quick start

```python
from experiment_design_kit import (
    cohen_h,
    two_proportion_z,
    two_proportion_sample_size,
)

print(cohen_h(0.10, 0.12))
print(two_proportion_sample_size(0.10, 0.12, power=0.8))
```

See `examples/run_demo.py` for a complete end-to-end demo and `tests/` for
the unit-test contract.
