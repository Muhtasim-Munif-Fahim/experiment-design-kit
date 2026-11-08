import math

import numpy as np
import pytest
from scipy import stats

from experiment_design_kit import (
    exponential_event_probability,
    freedman_events,
    hazard_from_median,
    logrank_power,
    logrank_sample_size,
    schoenfeld_events,
)


def test_schoenfeld_and_freedman_textbook_values():
    # (1.96 + 0.8416)^2 * 4 / log(0.7)^2 = 246.8 ; Freedman: 7.849 * (1.7 / 0.3)^2 = 252.0
    assert schoenfeld_events(0.7) == 247
    assert freedman_events(0.7) == 253
    # HR and 1/HR need the same events under Schoenfeld with equal allocation.
    assert schoenfeld_events(0.5) == schoenfeld_events(2.0) == 66
    # Unequal allocation needs more events.
    assert schoenfeld_events(0.7, allocation=2 / 3) > schoenfeld_events(0.7)
    assert schoenfeld_events(0.7, sides=1) < schoenfeld_events(0.7)


def test_event_probability_closed_form_matches_monte_carlo():
    rng = np.random.default_rng(0)
    lam, eta, a, f = 0.05, 0.01, 12.0, 18.0
    n = 400_000
    entry = rng.uniform(0, a, n)
    t_event = rng.exponential(1 / lam, n)
    t_drop = rng.exponential(1 / eta, n)
    observed = (t_event < t_drop) & (t_event < a + f - entry)
    assert exponential_event_probability(lam, a, f, eta) == pytest.approx(observed.mean(), abs=0.003)
    assert exponential_event_probability(lam, 0, f) == pytest.approx(1 - math.exp(-lam * f))


def test_sample_size_plan_consistency():
    lam = hazard_from_median(12.0)
    plan = logrank_sample_size(0.7, lam, accrual_time=24, follow_up=12)
    assert plan.events == 247
    assert plan.n_treatment == plan.n_control
    assert plan.n_total * plan.event_prob_pooled >= plan.events - 1
    assert plan.achieved_power >= 0.8
    assert plan.hazard_treatment == pytest.approx(0.7 * lam)
    # Longer follow-up -> more events per subject -> fewer subjects.
    longer = logrank_sample_size(0.7, lam, accrual_time=24, follow_up=36)
    assert longer.n_total < plan.n_total
    # Dropout inflates the sample size.
    drop = logrank_sample_size(0.7, lam, 24, 12, dropout_hazard=-math.log(1 - 0.1) / 12)
    assert drop.n_total > plan.n_total
    assert logrank_power(plan.n_total, 0.7, lam, 24, 12) == pytest.approx(0.8, abs=0.01)


def test_power_by_simulation():
    lam_c = hazard_from_median(6.0)
    hr, a, f = 0.6, 12.0, 6.0
    plan = logrank_sample_size(hr, lam_c, a, f, power=0.8)
    rng = np.random.default_rng(1)
    rejections = 0
    reps = 300
    n = plan.n_treatment
    for _ in range(reps):
        entry = rng.uniform(0, a, 2 * n)
        lam = np.r_[np.full(n, hr * lam_c), np.full(n, lam_c)]
        t = rng.exponential(1 / lam)
        cens = a + f - entry
        obs = np.minimum(t, cens)
        event = t <= cens
        x = stats.CensoredData(uncensored=obs[:n][event[:n]], right=obs[:n][~event[:n]])
        y = stats.CensoredData(uncensored=obs[n:][event[n:]], right=obs[n:][~event[n:]])
        rejections += stats.logrank(x, y).pvalue < 0.05
    assert rejections / reps == pytest.approx(0.8, abs=0.08)


def test_validation():
    with pytest.raises(ValueError):
        schoenfeld_events(1.0)
    with pytest.raises(ValueError):
        schoenfeld_events(0.7, allocation=1.0)
    with pytest.raises(ValueError):
        logrank_sample_size(0.7, 0.1, 12, 12, method="nope")
    with pytest.raises(ValueError):
        exponential_event_probability(0.1, 0, 0)
    with pytest.raises(ValueError):
        hazard_from_median(0)
