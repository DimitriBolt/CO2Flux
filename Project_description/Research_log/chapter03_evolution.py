"""Chapter 3: frozen Chapter 2 inputs, approved comparisons, and saved figures.

Initial preparation: python chapter03_evolution.py --prepare-input
Reproduction: python chapter03_evolution.py [--output-dir DIRECTORY]
The old chapter03.py describes archive coverage and is intentionally untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "sensorDB"
sys.path.insert(0, str(CORE))
from long_term_evolution import PARAMETERS, LEVELS, compare, reproducibility_table

DATA = HERE / "data/ch03"
OUTPUT = HERE / "output/ch03"
PREFIX = "long_term_evolution_"
INPUT = DATA / (PREFIX + "input.csv")
CALENDAR = DATA / (PREFIX + "calendar_input.csv")
INPUT_PROVENANCE = DATA / (PREFIX + "input_provenance.json")
PARAMETER_FILE = DATA / (PREFIX + "parameters.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    def scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=scalar) + "\n")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def save_csv(table, path):
    table.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def prepare_input():
    """Validate provenance before freezing a derived subset; never query Oracle."""
    original = HERE / "output/ch02"
    stem = "W_R4_C-4_full_archive"
    checks_file = original / (stem + "_preliminary_checks.json")
    checks = json.loads(checks_file.read_text())
    source_file = original / (stem + "_preliminary_trajectories.csv")
    source_prov = original / (stem + "_provenance.json")
    daily = original / (stem + "_daily.csv")
    expected = {source_file: checks["output_sha256"][source_file.name],
                source_prov: checks["input_provenance_sha256"],
                daily: checks["calendar_sha256"]}
    for name, digest in checks["protected_sha256"].items():
        if name.endswith("_trajectories.csv") or name.endswith("_full_archive_monthly.csv"):
            expected[Path(name)] = digest
    for path, digest in expected.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing or changed mandatory Chapter 2 input: {path}")
    prov = json.loads(source_prov.read_text())
    assert prov["column"] == PARAMETERS["column"]
    assert prov["sensor_ids"] == PARAMETERS["sensor_ids"]
    assert prov["depth_cm"] == {"D1": 5, "D2": 20, "D3": 35}
    # Audit copy establishes the documented monthly basis; no new admission is run.
    audit = HERE.parent / "Papers/LEO_CO2_sensor_audit_2026-06-30.pdf"
    assert audit.is_file()
    audit_hashes = [v for k, v in prov["source_sha256"].items() if k.endswith(audit.name)]
    assert audit_hashes == [sha(audit)], "Audit source differs from the Chapter 2 provenance"
    t = read_csv(source_file)
    assert len(t) == 4764 and t.date.is_unique
    computed = t.calculation_status.eq("computed")
    confirmed = computed & t.admission_status.eq("confirmed")
    boundary = confirmed & t.context_admission_note.notna()
    assert set(t.loc[boundary, "date"]) == {"2013-09-30", "2017-10-01", "2020-08-31"}
    main = confirmed & ~boundary & t.date.between("2017-10-01", "2020-08-31")
    separate = confirmed & ~boundary & t.date.between("2013-09-01", "2013-09-30")
    preliminary = computed & t.admission_status.eq("unconfirmed")
    assert (int(main.sum()), int(separate.sum()), int(boundary.sum()), int(preliminary.sum())) == (783, 6, 3, 1244)
    assert int(confirmed.sum()) == 792
    assert int(t.calculation_status.eq("excluded_by_audit").sum()) == 3
    metrics = [f"{kind}_{j}" for kind in ("A", "H", "R2") for j in LEVELS]
    differences = {}
    for mask, name in [(main, "W_R4_C-4_2017-10-01_to_2020-09-01"),
                       (separate, "W_R4_C-4_2013-09-01_to_2013-10-01")]:
        old = read_csv(original / (name + "_trajectories.csv"))
        old = old[old.calculation_status.eq("computed")].set_index("date")
        selected = t.loc[mask].set_index("date")
        assert set(selected.index) == set(old.index), "Ambiguous approved input dates"
        delta = selected[metrics] - old.loc[selected.index, metrics]
        assert np.isfinite(selected[metrics]).all().all()
        assert np.allclose(selected[metrics], old.loc[selected.index, metrics], atol=1e-7, rtol=1e-7)
        differences[name] = float(np.abs(delta.to_numpy()).max())
    t["analysis_group"] = "not_in_analysis"
    for mask, group in [(main, "main"), (separate, "separate_2013"), (boundary, "boundary")]:
        t.loc[mask, "analysis_group"] = group
    columns = ["date", "analysis_group", "admission_status", "admission_reason", "calculation_status",
               "context_admission_note"] + metrics + ["phase_status_" + j for j in LEVELS]
    frozen = t.loc[main | separate | boundary, columns].sort_values("date")
    cal = t[t.date.between("2017-10-01", "2020-08-31")][
        ["date", "analysis_group", "admission_status", "calculation_status", "category",
         "context_admission_note", "skip_reason"]].copy()
    assert len(cal) == 1066
    DATA.mkdir(parents=True, exist_ok=True)
    save_csv(frozen, INPUT)
    save_csv(cal, CALENDAR)
    write_json(PARAMETER_FILE, PARAMETERS)
    inputs = {str(p.relative_to(HERE.parent.parent)): sha(p) for p in expected}
    inputs[str(checks_file.relative_to(HERE.parent.parent))] = sha(checks_file)
    inputs[str(audit.relative_to(HERE.parent.parent))] = sha(audit)
    write_json(INPUT_PROVENANCE, {
        "source_main_commit": PARAMETERS["source_main_commit"],
        "approval_date": "2026-09-29", "execution_date": "2026-09-30",
        "source_files_sha256": inputs,
        "group_counts": {"main": 783, "separate_2013": 6, "boundary": 3,
                         "preliminary_excluded": 1244, "audit_excluded": 3},
        "reference_max_absolute_difference": differences,
        "frozen_sha256": {p.name: sha(p) for p in [INPUT, CALENDAR, PARAMETER_FILE]},
        "origin": "Exact approved subset of existing Chapter 2 characteristics; no new extraction or harmonic fit",
        "phase_limits": "Inherited not_assessed; D3 2017-11-22 known_sensitive retained",
    })


def load_inputs():
    prov = json.loads(INPUT_PROVENANCE.read_text())
    for name, digest in prov["frozen_sha256"].items():
        if sha(DATA / name) != digest:
            raise ValueError("Frozen input changed: " + name)
    assert json.loads(PARAMETER_FILE.read_text()) == PARAMETERS
    inputs, cal = read_csv(INPUT), read_csv(CALENDAR)
    assert inputs.analysis_group.value_counts().to_dict() == {"main": 783, "separate_2013": 6, "boundary": 3}
    inputs["date"] = pd.to_datetime(inputs.date)
    main = inputs[inputs.analysis_group == "main"].copy()
    assert main.date.is_unique and main.date.is_monotonic_increasing
    assert main.admission_status.eq("confirmed").all() and main.context_admission_note.isna().all()
    return inputs, main, cal, prov


def calendar_figure(cal):
    periods = pd.period_range("2017-10", "2020-08", freq="M")
    matrix = np.full((len(periods), 31), np.nan)
    for i, period in enumerate(periods):
        rows = cal[cal.date.str.startswith(str(period))]
        for row in rows.itertuples():
            matrix[i, int(row.date[-2:]) - 1] = {"main": 1, "boundary": 2}.get(row.analysis_group, 0)
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    ax.imshow(matrix, aspect="auto", interpolation="none", cmap=ListedColormap(["#e7e7e7", "#267c80", "#d69a35"]), vmin=0, vmax=2)
    ax.set(xticks=np.arange(0, 31, 2), xticklabels=np.arange(1, 32, 2),
           yticks=np.arange(len(periods)), yticklabels=[str(p) for p in periods],
           xlabel="Число месяца", title="783 основных суток: зелёный; пропуск: серый; пограничные: жёлтый")
    ax.tick_params(labelsize=8)
    return fig


def contrast_figure(table, kind):
    base = table[(table.coverage_variant == "pairwise") & (table.block_days == 7)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 10), sharey=True, layout="constrained")
    labels = [f"{r.month:02d}: {r.year1}→{r.year2}" for r in base[base.sensor == "D1"].itertuples()]
    for ax, level in zip(axes, LEVELS):
        rows = base[base.sensor == level].reset_index(drop=True)
        for i, row in rows.iterrows():
            if not row.coverage_ok:
                continue
            value = row.delta_A if kind == "A" else row.delta_h
            valid = row.bootstrap_status == "ok" and (kind == "A" or row.phase_ci_status == "ok")
            ax.scatter(value, i, color="#267c80" if valid else "#999999", s=16, zorder=3)
            if valid:
                low = row.A_ci_low if kind == "A" else row.phase_ci_low_unwrapped
                high = row.A_ci_high if kind == "A" else row.phase_ci_high_unwrapped
                ax.hlines(i, low, high, color="#267c80", lw=1.5)
        ax.axvline(0, color="#555555", lw=.7)
        ax.set(title=level, xlabel="ΔA, ppm" if kind == "A" else "Δh, ч")
        ax.grid(axis="y", alpha=.15)
    axes[0].set(yticks=np.arange(len(labels)), yticklabels=labels)
    axes[0].invert_yaxis()
    axes[0].tick_params(labelsize=8)
    fig.suptitle(("Амплитуды" if kind == "A" else "Времена максимумов") +
                 ": парные оценки и 95%-е интервалы, блок 7 суток\nСерые точки: bootstrap недоступен; интервалы поточечные")
    return fig


def run(output_dir=OUTPUT):
    inputs, main, calendar, provenance = load_inputs()
    contrasts, paired, sensitivity = compare(main)
    reproduction = reproducibility_table(contrasts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calendar["month"] = calendar.date.str[:7]
    coverage = calendar.assign(main=calendar.analysis_group.eq("main").astype(int)).groupby("month").agg(
        calendar_days=("date", "size"), main_days=("main", "sum")).reset_index()
    tables = {"calendar": calendar, "monthly_coverage": coverage, "comparisons": contrasts,
              "paired_dates": paired, "coverage_sensitivity": sensitivity,
              "reproducibility": reproduction,
              "separate_2013": inputs[inputs.analysis_group == "separate_2013"],
              "boundary_dates": inputs[inputs.analysis_group == "boundary"]}
    written = []
    for name, table in tables.items():
        path = output_dir / (PREFIX + name + ".csv")
        save_csv(table, path); written.append(path)
    for name, fig in [("calendar", calendar_figure(calendar)),
                      ("amplitudes", contrast_figure(contrasts, "A")),
                      ("phase_shifts", contrast_figure(contrasts, "H"))]:
        path = output_dir / (PREFIX + name + ".png")
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    primary = contrasts[(contrasts.coverage_variant == "pairwise") & (contrasts.block_days == 7)]
    per_sensor = {}
    for level, group in primary.groupby("sensor"):
        per_sensor[level] = {
            "coverage_eligible": int(group.coverage_ok.sum()),
            "bootstrap_available": int(group.bootstrap_status.eq("ok").sum()),
            "amplitude_positive_intervals": int(group.A_ci_direction.eq(1).sum()),
            "amplitude_negative_intervals": int(group.A_ci_direction.eq(-1).sum()),
            "phase_positive_intervals": int(group.phase_ci_direction.eq(1).sum()),
            "phase_negative_intervals": int(group.phase_ci_direction.eq(-1).sum()),
            "phase_local_intervals": int(group.phase_ci_status.eq("ok").sum()),
        }
    summary = {"main_days": len(main), "planned_year_month_pairs": len(primary)//3,
               "per_sensor": per_sensor,
               "bootstrap_coverage_by_length": {str(length): int(((contrasts.sensor == "D1") &
                  (contrasts.coverage_variant == "pairwise") & (contrasts.block_days == length) &
                  (contrasts.bootstrap_status == "ok")).sum()) for length in PARAMETERS["block_days"]},
               "disjoint_year_comparison_pairs": int(reproduction.disjoint_year_comparison_pairs.max()),
               "pointwise_not_simultaneous_intervals": True}
    summary_file = output_dir / (PREFIX + "summary.json")
    write_json(summary_file, summary); written.append(summary_file)
    write_json(output_dir / (PREFIX + "provenance.json"), {
        "source_main_commit": PARAMETERS["source_main_commit"], "parameters": PARAMETERS,
        "input_provenance_sha256": sha(INPUT_PROVENANCE),
        "frozen_inputs": provenance["frozen_sha256"],
        "code_sha256": {"chapter03_evolution.py": sha(__file__),
                        "long_term_evolution.py": sha(CORE / "long_term_evolution.py")},
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "output_sha256": {p.name: sha(p) for p in written},
        "raw_fits_recomputed": False, "oracle_access": False,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-input", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.prepare_input:
        prepare_input()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))
