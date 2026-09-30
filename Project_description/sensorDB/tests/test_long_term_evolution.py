"""Numerical and calendar edge cases for the approved Chapter 3 procedure."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_term_evolution import (
    calendar_blocks, circular_mean, compare, phase_interval, sample_indices, wrap24,
)


def test_clock_seam_and_antipodal_undefined():
    mean, r = circular_mean([23.5, .5])
    assert abs(float(mean)) < 1e-12 and r > .99
    mean, r = circular_mean([0, 12])
    assert np.isnan(mean) and r < 1e-12
    np.testing.assert_allclose(wrap24([23, -23, 12]), [-1, 1, -12])


def test_calendar_blocks_never_bridge_gaps_or_hide_short_fragments():
    days = np.array([1, 2, 3, 4, 8, 9, 10, 11])
    blocks, status = calendar_blocks(days, 3)
    assert status == "ok"
    assert np.all(np.diff(days[blocks], axis=1) == 1)
    blocks, status = calendar_blocks([1, 2, 3, 4, 9], 3)
    assert status == "some_pairs_outside_full_blocks"
    _, status = calendar_blocks(np.arange(1, 8), 7)
    assert status == "fewer_than_two_distinct_blocks"


def test_resampling_preserves_joint_indices_and_requested_length():
    blocks, _ = calendar_blocks(np.arange(1, 16), 7)
    indices = sample_indices(blocks, 15, 2000, np.random.default_rng(42))
    assert indices.shape == (2000, 15)
    assert np.all(np.diff(indices[:, :7], axis=1) == 1)
    x = np.arange(15.)
    joint = np.column_stack([x, 2*x, -x])[indices]
    np.testing.assert_equal(np.median(joint[:, :, 1], axis=1), 2*np.median(joint[:, :, 0], axis=1))


def test_circular_interval_wrap_and_nonlocal_abstention():
    means = wrap24(np.linspace(11.7, 12.1, 2000))
    interval = phase_interval(means, 11.9)
    assert interval["phase_ci_status"] == "ok"
    assert interval["phase_ci_crosses_cut"] and interval["phase_ci_excludes_zero"]
    assert interval["phase_ci_direction"] == 0  # no arbitrary direction at antipode
    interval = phase_interval(np.linspace(-11, 11, 2000), 0.)
    assert interval["phase_ci_status"] == "not_localized_in_open_semicircle"


def test_paired_median_not_difference_of_medians_and_excludes_unpaired_date():
    frames = []
    # Four repetitions: median paired difference is 1; difference of medians is 100.
    old = np.tile([0., 0., 100.], 4)
    new = np.tile([1., 100., 101.], 4)
    for year, values in [(2018, old), (2019, new)]:
        f = pd.DataFrame({"date": pd.date_range(f"{year}-01-01", periods=12)})
        for level in ["D1", "D2", "D3"]:
            f["A_"+level] = values
            f["H_"+level] = 23.5 if year == 2018 else .5
            f["phase_status_"+level] = "not_assessed"
        frames.append(f)
    extra = frames[1].iloc[[0]].copy()
    extra["date"] = pd.Timestamp("2019-01-20")
    extra["A_D1"] = 1e6
    result, pairs, _ = compare(pd.concat(frames+[extra], ignore_index=True))
    row = result[(result.month == 1) & (result.year1 == 2018) & (result.year2 == 2019) &
                 (result.sensor == "D1") & (result.coverage_variant == "pairwise") &
                 (result.block_days == 7)].iloc[0]
    assert row.n_pairs == 12 and row.delta_A == 1 and abs(row.delta_h - 1) < 1e-12
    assert row.bootstrap_status == "ok"
    # Missing 2020 must make the predeclared all-years coverage sensitivity unavailable.
    common = result[(result.month == 1) & (result.coverage_variant == "all_years_common")]
    assert common.n_pairs.eq(0).all()
    assert (pairs.day <= 12).all()
