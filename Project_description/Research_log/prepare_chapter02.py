"""Окончательный допуск и H/A (--final), обновление sensorDB (--refresh).

Исторические режимы --archive/--preliminary сохранены только для воспроизведения.

По умолчанию выполняет прежний SQL SELECT. --source-csv позволяет упаковать
уже полученную выгрузку побайтно, без нового запроса. Существующий снимок
никогда не перезаписывается. Для чтения результата используйте chapter02.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Project_description.Research_log.chapter02 import HERE, DATA_DIR, audit_raw, _column_settings, column
from Project_description.Research_log import chapter02 as ch


def prepare_snapshot(source_csv=None, exported_at_utc=None, evidence_dir=None):
    if DATA_DIR.exists():
        raise FileExistsError("Снимок главы 2 уже существует; автоматическая замена запрещена.")
    base = json.loads((HERE / "data/ch01/provenance.json").read_text())
    keys = ["column", "slope", "schema", "variable_id", "sensors", "time_field", "time_interpretation",
            "trend_window_hours", "gap_factor", "service_value_limit"]
    manifest = {key: base[key] for key in keys}
    manifest.update(chapter="02", stage="raw_data_and_record_audit_only",
                    start_inclusive="2017-11-01T00:00:00", end_exclusive="2017-12-01T00:00:00")
    ids = {level: info["sensor_id"] for level, info in manifest["sensors"].items()}
    if source_csv is None:
        with _column_settings(manifest):
            raw, observed_ids = column.fetch_data()
        if observed_ids != ids:
            raise ValueError("Состав датчиков отличается от октября; нужна проверка.")
        exported_at_utc = datetime.now(timezone.utc).isoformat()
    else:
        if not exported_at_utc:
            raise ValueError("Для готовой выгрузки требуется --exported-at-utc.")
        raw = pd.read_csv(source_csv, parse_dates=["localdatetime"])
    if set(raw.sensorid) != set(ids.values()):
        raise ValueError("Неверный набор sensorid.")
    if not (raw.localdatetime.ge(pd.Timestamp(manifest["start_inclusive"]))
            & raw.localdatetime.lt(pd.Timestamp(manifest["end_exclusive"]))).all():
        raise ValueError("Временные метки выходят за границы ноября.")
    diagnostics = audit_raw(raw, ids)
    manifest.update(data_exported_at_utc=exported_at_utc,
                    packaged_at_utc=datetime.now(timezone.utc).isoformat(),
                    diagnostics={"constant_run_minimum_hours": 1.0,
                                 "short_step_maximum_seconds": 10,
                                 "note": "Descriptive flags only; no rejection, deduplication or resampling."},
                    queries={"sensors": column.SENSOR_QUERY.strip(), "measurements": column.DATA_QUERY.strip()},
                    notes=["Original timestamps and all values retained, including repeated records.",
                           "No trend, phase or lag estimates for November at this stage.",
                           "October snapshot and calculation are unchanged."])
    if source_csv is not None:
        manifest["original_source_csv"] = str(Path(source_csv).resolve().relative_to(PROJECT))
    manifest["implementation_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("chapter02.py", "prepare_chapter02.py")}
    manifest["plot_implementation_sha256"] = hashlib.sha256(Path(column.__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="november-", dir=HERE / "data") as temporary:
        staging = Path(temporary) / "ch02"
        staging.mkdir()
        if source_csv is None:
            raw.to_csv(staging / "measurements.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        else:
            shutil.copyfile(source_csv, staging / "measurements.csv")
        for name, frame in diagnostics.items():
            frame.to_csv(staging / f"reference_{name}.csv", index=False, float_format="%.10f")
        if evidence_dir is not None:
            for source, target in (("november_db_columns.csv", "database_columns.csv"),
                                   ("november_duplicate_example_with_ids.csv", "database_duplicate_example.csv")):
                shutil.copyfile(Path(evidence_dir) / source, staging / target)
        manifest["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(staging.glob("*.csv"))}
        (staging / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        staging.rename(DATA_DIR)
    print(diagnostics["summary"].to_string(index=False))
    print(diagnostics["constant_runs"].to_string(index=False))
    print("Создан снимок:", DATA_DIR)


def archive_admission(month, inventory, audit_end, confirmed_blocks):
    """Published evidence only; First/Last never imply continuity between them."""
    day = pd.Period(month, freq="M").start_time
    for start, end, source in confirmed_blocks:
        if pd.Timestamp(start) <= day < pd.Timestamp(end):
            return "confirmed", source
    if month > audit_end:
        return "unconfirmed", "Месяц за пределами аудита; полного допуска нет"
    rejected = inventory.loc[inventory.last_alive.lt(month), "audit_code"].tolist()
    if rejected:
        return "not_admitted_audit", "После последнего пригодного месяца по таблице 2 I.G.: " + ", ".join(rejected)
    if inventory.first_alive.eq(month).all():
        return "confirmed", "Общий First в таблице 2 I.G., стр. 9–10"
    return "unconfirmed", "Нет помесячной совместной классификации I.G.; First/Last/Run недостаточно"


def process_archive(audit_path):
    """One technical calendar over all local CO2 rows; fit only admitted new days.

    No Oracle, despiking, interpolation, new admission estimator, or plotting.
    Existing admitted block contexts and saved fits remain authoritative.
    """
    archive, output = HERE / "data/ch03", HERE / "output/ch02"
    stem = "W_R4_C-4_full_archive"
    sensors = {"D1": 994, "D2": 1010, "D3": 1026}
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    snapshot = json.loads((archive / "provenance.json").read_text())
    screen = HERE / "data/west_august_2026_screening"
    audit_meta = json.loads((screen / "provenance.json").read_text())
    if digest(audit_path) != audit_meta["source_audit"]["sha256"]:
        raise ValueError("Версия PDF аудита отличается от проверенной")
    if digest(screen / "ig_inventory.csv") != audit_meta["files_sha256"]["ig_inventory.csv"]:
        raise ValueError("Изменена сохранённая таблица аудита")
    inventory = pd.read_csv(screen / "ig_inventory.csv")
    inventory = inventory.loc[inventory.audit_code.isin(["W_R4_C-4_" + l for l in sensors])]
    if len(inventory) != 3:
        raise ValueError("Нет трёх строк выбранных датчиков в таблице I.G.")
    audit_end = audit_meta["source_audit"]["inventory_end"]
    source_hashes = {str(p): digest(p) for p in [audit_path, screen / "ig_inventory.csv", archive / "provenance.json"]}
    saved_stems = ["W_R4_C-4_2013-09-01_to_2013-10-01", ch.HISTORICAL_STEM]
    protected = [HERE / n for n in ("README.md", "PLAN.md", "02_monthly_changes.ipynb", "chapter02.py")]
    protected += [p for name in saved_stems for p in output.glob(name + "*")]
    protected_hashes = {str(p): digest(p) for p in protected}
    frames = []
    for name, meta in sorted(snapshot["raw_files"].items()):
        path = archive / "raw" / name.replace(".json", ".parquet")
        if digest(path) != meta["sha256"]:
            raise ValueError(f"Изменён снимок {path}")
        source_hashes[str(path)] = digest(path)
        frame = pd.read_parquet(path, columns=["localdatetime", "sensorid", "variableid", "datavalue"])
        frames.append(frame.loc[frame.variableid.eq(9) & frame.sensorid.isin(sensors.values())])
    raw = pd.concat(frames, ignore_index=True)
    raw["localdatetime"] = pd.to_datetime(raw.localdatetime)
    first, last = raw.localdatetime.min(), raw.localdatetime.max()
    start, end = first.normalize(), last.normalize() + pd.Timedelta(days=1)
    constants_path = archive / "constant_intervals.csv"
    if digest(constants_path) != snapshot["files_sha256"][constants_path.name]:
        raise ValueError("Изменён регистр постоянных участков")
    source_hashes[str(constants_path)] = digest(constants_path)
    constants = pd.read_csv(constants_path, parse_dates=["start", "end"])
    constants = constants.loc[constants.channel.isin(["CO2 " + l for l in sensors])].copy()
    constants["level"] = constants.channel.str.removeprefix("CO2 ")
    old_meta = json.loads((output / f"{ch.HISTORICAL_STEM}_calendar_provenance.json").read_text())
    exclusions = pd.DataFrame(old_meta["approved_exclusions"])
    print(f"Все входы проверены: {len(raw)} строк CO2; {first} — {last}", flush=True)
    # The single full-archive pass; month boundaries never truncate its contexts.
    technical = ch.calculation_calendar(raw, sensors, start, end, constants, exclusions)
    technical = technical.rename(columns={"category": "technical_category", "reason": "technical_reason"})
    technical["month"] = technical.date.dt.strftime("%Y-%m")
    daily = technical.copy().set_index("date")
    saved_parts, fit_parts, blocks = [], [], []
    for name in saved_stems:
        part = pd.read_csv(output / f"{name}_trajectories.csv", parse_dates=["date", "context_start", "context_end_exclusive"])
        part["result_origin"] = name
        saved_parts.append(part)
        fit_parts.append(pd.read_csv(output / f"{name}_daily_fits.csv", parse_dates=["date"]))
        left, right = part.date.min(), part.date.max() + pd.Timedelta(days=1)
        blocks.append((left, right, "I.G. стр. 21: совместный блок 2017-10–2020-08" if name == ch.HISTORICAL_STEM
                       else "I.G. таблица 2, стр. 9–10: общий First=2013-09"))
    saved = pd.concat(saved_parts, ignore_index=True).set_index("date")
    if saved.index.duplicated().any():
        raise ValueError("Пересечение сохранённых блоков")
    for source, target in [("category", "admitted_category"), ("reason", "admitted_reason"),
                           ("context_start", "admitted_context_start"), ("context_end_exclusive", "admitted_context_end_exclusive")]:
        daily[target] = saved[source].reindex(daily.index)
    daily["admission_status"], daily["admission_reason"] = zip(*[
        archive_admission(m, inventory, audit_end, blocks) for m in daily.month])
    daily["calculation_status"] = "skipped"
    value_columns = [f"{prefix}_{level}" for prefix in ("H", "A", "R2", "phase_status") for level in sensors]
    for field in value_columns + ["calculation_error", "result_origin"]:
        daily[field] = saved[field].reindex(daily.index)
    existing = saved.index[saved.calculation_status.eq("computed")]
    daily.loc[existing, "calculation_status"] = "computed"
    # Source-admitted contexts are required as well as a confirmed month. In
    # particular, technical support from an unadmitted adjacent month is not a permit.
    ready = (daily.admission_status.eq("confirmed") & daily.admitted_category.eq("A")
             & ~daily.index.isin(existing))
    candidates = saved.loc[daily.index[ready]].copy()
    if len(candidates):
        added, new_fits = ch.calculate_trajectories(raw, sensors, candidates.reset_index(), exclusions)
        added = added.set_index("date")
        for field in value_columns + ["calculation_status", "calculation_error"]:
            daily.loc[added.index, field] = added[field]
        daily.loc[added.index, "result_origin"] = stem
        fit_parts.append(new_fits)
    new_count = int(daily.calculation_status.eq("computed").sum() - len(existing))
    daily["skip_reason"] = ""
    for day, row in daily.loc[~daily.calculation_status.eq("computed")].iterrows():
        reasons = []
        if row.technical_category != "A":
            reasons.append(row.technical_reason)
        if row.admission_status != "confirmed":
            reasons.append(row.admission_reason)
        elif row.admitted_category != "A":
            reasons.append(str(row.admitted_reason) if pd.notna(row.admitted_reason)
                           else "Нет подтверждённого допуска полного временного окружения")
        if pd.notna(row.calculation_error):
            reasons.append(row.calculation_error)
        daily.loc[day, "skip_reason"] = "; ".join(dict.fromkeys(reasons))
    monthly_rows = []
    for month, group in daily.groupby("month", sort=True):
        row = dict(month=month, days=len(group),
                   admission_status=group.admission_status.iloc[0], admission_reason=group.admission_reason.iloc[0])
        for level in sensors:
            row[f"{level}_rows"] = int(group[f"{level}_rows_day"].sum())
            row[f"{level}_days_with_records"] = int(group[f"{level}_rows_day"].gt(0).sum())
        row["all_three_have_records"] = all(row[f"{l}_rows"] > 0 for l in sensors)
        for category in ("A", "B", "C"):
            row[f"technical_{category}_days"] = int(group.technical_category.eq(category).sum())
        for label, suffix in [("duplicates", "duplicate_extra_rows_context"), ("constants", "unresolved_constant_rows_required"),
                              ("range", "range_rows_required"), ("service", "service_rows_required")]:
            row[f"days_with_{label}"] = int(group[[f"{l}_{suffix}" for l in sensors]].sum(axis=1).gt(0).sum())
        row["computed_days"] = int(group.calculation_status.eq("computed").sum())
        row["added_days"] = int((group.calculation_status.eq("computed") & group.result_origin.eq(stem)).sum())
        row["technical_A_unconfirmed_days"] = int((group.technical_category.eq("A") & group.admission_status.eq("unconfirmed")).sum())
        row["technical_A_audit_not_admitted_days"] = int((group.technical_category.eq("A") & group.admission_status.eq("not_admitted_audit")).sum())
        reasons = group.loc[~group.calculation_status.eq("computed"), "skip_reason"]
        row["skip_reasons"] = " | ".join(dict.fromkeys(reasons.loc[reasons.ne("")]))
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    # Validate exhaustiveness, raw-row accounting, and exact reuse of saved floats.
    assert daily.index.equals(pd.date_range(start, end, inclusive="left", name="date"))
    assert monthly.month.tolist() == pd.period_range(start, last, freq="M").astype(str).tolist()
    assert int(monthly.days.sum()) == len(daily)
    assert int(monthly[[f"{l}_rows" for l in sensors]].sum().sum()) == len(raw)
    assert (monthly[[f"technical_{c}_days" for c in "ABC"]].sum(axis=1) == monthly.days).all()
    pd.testing.assert_frame_equal(daily.loc[existing, value_columns], saved.loc[existing, value_columns])
    assert daily.loc[~daily.calculation_status.eq("computed"), "skip_reason"].ne("").all()
    assert daily.loc[daily.calculation_status.eq("computed"), "admission_status"].eq("confirmed").all()
    assert protected_hashes == {str(p): digest(p) for p in protected}
    daily.reset_index().to_csv(output / f"{stem}_daily.csv", index=False)
    monthly.to_csv(output / f"{stem}_monthly.csv", index=False)
    pd.concat(fit_parts, ignore_index=True).sort_values(["date", "level"]).to_csv(output / f"{stem}_daily_fits.csv", index=False)
    report = ["# W_R4_C-4: полный технический календарь архива", "",
              f"Записи: {first} — {last}. {len(monthly)} месяцев, {len(daily)} календарных суток (крайние неполные).",
              f"Рассчитано: {int(daily.calculation_status.eq('computed').sum())}; добавлено: {new_count}.", "",
              "A/B/C здесь — технический результат без научного допуска. A не означает пригодности по I.G.",
              "Допуск: + подтверждён; ? не подтверждён; − отсутствует по аудиту. Сутки на границе подтверждённого блока дополнительно ограничены его собственным окружением.",
              "Причины по датам и датчикам сохранены в full_archive_daily.csv; расширенные месячные счётчики — в full_archive_monthly.csv.", "",
              "| Месяц | Суток | Записи D1/D2/D3 | Тех. A | Тех. B | Тех. C | Допуск | Рассчитано | Добавлено |",
              "|---|---:|---|---:|---:|---:|:---:|---:|---:|"]
    for row in monthly.itertuples():
        mark = {"confirmed": "+", "unconfirmed": "?", "not_admitted_audit": "−"}[row.admission_status]
        report.append(f"| {row.month} | {row.days} | {row.D1_rows}/{row.D2_rows}/{row.D3_rows} | {row.technical_A_days} | {row.technical_B_days} | {row.technical_C_days} | {mark} | {row.computed_days} | {row.added_days} |")
    (output / f"{stem}_monthly.md").write_text("\n".join(report) + "\n")
    summary = dict(months=len(monthly), days=len(daily), first_record=str(first), last_record=str(last),
                   raw_rows=len(raw), technical_counts=daily.technical_category.value_counts().to_dict(),
                   admission_months=monthly.admission_status.value_counts().to_dict(),
                   admission_days=daily.admission_status.value_counts().to_dict(),
                   computed_days=int(daily.calculation_status.eq("computed").sum()),
                   existing_days=len(existing), candidate_days=len(candidates), added_days=new_count,
                   technical_A_unconfirmed_days=int(monthly.technical_A_unconfirmed_days.sum()),
                   technical_A_audit_not_admitted_days=int(monthly.technical_A_audit_not_admitted_days.sum()))
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), column="W_R4_C-4", sensor_ids=sensors,
                      depth_cm={"D1":5,"D2":20,"D3":35}, summary=summary,
                      source_sha256=source_hashes, protected_sha256=protected_hashes,
                      implementation_sha256={str(p):digest(p) for p in [Path(__file__), HERE / "chapter02.py", HERE.parent / "sensorDB/plot_column_timeseries.py", HERE.parent / "sensorDB/diurnal_analysis.py"]},
                      approved_exclusions=old_meta["approved_exclusions"],
                      admission_algorithm="Unavailable: no replacement implemented. Published evidence only.",
                      checks=dict(all_months_and_days_present=True, all_raw_rows_accounted=True, old_values_equal=True,
                                  old_files_byte_identical=True, all_skips_explained=True, only_admitted_contexts_fitted=True),
                      output_sha256={p.name:digest(p) for p in output.glob(stem + '*') if p.suffix != '.json'})
    (output / f"{stem}_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return daily.reset_index(), monthly


def prepare_preliminary():
    """Recompute all technical A days using the saved calendar, without re-admission.

    Audit refusals are fitted only for a separate technical record, never plotted.
    No cache reuse, calendar rebuild, new cleaning, or change to the main report.
    """
    output = HERE / "output/ch02"
    source_stem, stem = "W_R4_C-4_full_archive", "W_R4_C-4_full_archive_preliminary"
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source_path = output / f"{source_stem}_provenance.json"
    source = json.loads(source_path.read_text())
    sensors = source["sensor_ids"]
    for name, expected in source["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Изменён источник полного календаря: {name}")
    for suffix in ("daily.csv", "monthly.csv"):
        name = f"{source_stem}_{suffix}"
        if digest(output / name) != source["output_sha256"][name]:
            raise ValueError(f"Изменён полный календарь: {name}")
    saved_stems = ["W_R4_C-4_2013-09-01_to_2013-10-01", ch.HISTORICAL_STEM]
    protected = [HERE / name for name in ("README.md", "PLAN.md", "02_monthly_changes.ipynb", "chapter01.py")]
    protected += [p for name in saved_stems for p in output.glob(name + "*")]
    protected += [p for p in output.glob(source_stem + "*") if not p.name.startswith(stem)]
    protected += [Path(column.__file__), Path(column.__file__).with_name("diurnal_analysis.py")]
    protected_hashes = {str(p): digest(p) for p in protected}
    saved_calendar = pd.read_csv(output / f"{source_stem}_daily.csv", parse_dates=["date"])
    # Discard previous fits and clipped admitted contexts: only the technical calendar is input.
    fields = ["date", "context_start", "context_end_exclusive", "technical_category", "technical_reason",
              "blocking_reasons", "decision_reasons", "month", "admission_status", "admission_reason"]
    calendar = saved_calendar[fields].rename(columns={"technical_category": "category", "technical_reason": "reason"})
    frames = []
    for filename in source["source_sha256"]:
        if Path(filename).suffix == ".parquet":
            frame = pd.read_parquet(filename, columns=["localdatetime", "sensorid", "variableid", "datavalue"])
            selected = frame.loc[frame.variableid.eq(9) & frame.sensorid.isin(sensors.values())]
            if not selected.empty:
                frames.append(selected)
    raw = pd.concat(frames, ignore_index=True)
    exclusions = pd.DataFrame(source["approved_exclusions"])
    print(f"Проверен полный календарь: {len(calendar)} суток; новый расчёт {(calendar.category == 'A').sum()} суток", flush=True)
    table, fits = ch.calculate_trajectories(raw, sensors, calendar, exclusions)
    computed = table.calculation_status.eq("computed")
    assert not table.loc[computed, "category"].ne("A").any()
    assert len(fits) == 3 * computed.sum() and not fits.duplicated(["date", "level"]).any()
    table["use_in_preliminary"] = computed & table.admission_status.ne("not_admitted_audit")
    table["context_admission_note"] = ""
    month_status = calendar.groupby("month").admission_status.first()
    for row in table.loc[computed & table.admission_status.eq("confirmed")].itertuples():
        months = pd.period_range(row.context_start, row.context_end_exclusive - pd.Timedelta(nanoseconds=1), freq="M").astype(str)
        if not month_status.reindex(months).eq("confirmed").all():
            table.loc[row.Index, "context_admission_note"] = "Окружение включает соседний месяц без подтверждённого допуска"
    fits = fits.merge(table[["date", "admission_status", "use_in_preliminary", "context_admission_note"]], on="date", validate="many_to_one")
    comparisons = []
    for name in saved_stems:
        previous = pd.read_csv(output / f"{name}_trajectories.csv", parse_dates=["date"])
        previous = previous.loc[previous.calculation_status.eq("computed")].set_index("date")
        current = table.set_index("date").loc[previous.index]
        assert current.calculation_status.eq("computed").all()
        values = [f"{prefix}_{level}" for prefix in ("H", "A", "R2") for level in sensors]
        np.testing.assert_allclose(current[values], previous[values], atol=1e-7, rtol=1e-7, equal_nan=True)
        status_fields = [f"phase_status_{level}" for level in sensors]
        pd.testing.assert_frame_equal(current[status_fields], previous[status_fields])
        old_fits = pd.read_csv(output / f"{name}_daily_fits.csv", parse_dates=["date"]).set_index(["date", "level"])
        new_fits = fits.set_index(["date", "level"]).loc[old_fits.index]
        numeric = old_fits.select_dtypes(include="number").columns
        np.testing.assert_allclose(new_fits[numeric], old_fits[numeric], atol=1e-7, rtol=1e-7, equal_nan=True)
        comparisons.append(dict(source=name, days=len(previous), atol=1e-7, rtol=1e-7,
            max_H_A_R2_absolute_difference=float(np.nanmax(np.abs(current[values].to_numpy()-previous[values].to_numpy()))),
            max_fit_absolute_difference=float(np.nanmax(np.abs(new_fits[numeric].to_numpy()-old_fits[numeric].to_numpy())))))
    shown = table.loc[table.use_in_preliminary]
    excluded = table.loc[computed & ~table.use_in_preliminary]
    summary = dict(archive_months=int(calendar.month.nunique()), archive_days=len(calendar),
        first_record=source["summary"]["first_record"], last_record=source["summary"]["last_record"],
        calculated_days=int(computed.sum()), plotted_days=len(shown),
        calculated_by_monthly_admission=table.loc[computed].admission_status.value_counts().to_dict(),
        failed_A_days=int(table.calculation_status.eq("failed").sum()),
        skipped_B_C_days=int(table.calculation_status.eq("skipped").sum()),
        first_plotted_day=str(shown.date.min().date()), last_plotted_day=str(shown.date.max().date()),
        added_to_previous_789=len(shown)-sum(item["days"] for item in comparisons),
        outside_previous_35_month_block=int((shown.date.lt("2017-10-01") | shown.date.ge("2020-09-01")).sum()),
        context_caveat_dates=table.loc[table.context_admission_note.ne(""), "date"].dt.strftime("%Y-%m-%d").tolist(),
        audit_excluded_dates=excluded.date.dt.strftime("%Y-%m-%d").tolist(),
        phase_status_counts=fits.loc[fits.use_in_preliminary].phase_status.value_counts().to_dict())
    monthly = table.groupby(["month", "admission_status"], sort=True).agg(
        days=("date", "size"), calculated_days=("calculation_status", lambda x: int(x.eq("computed").sum())),
        preliminary_days=("use_in_preliminary", "sum")).reset_index()
    monthly.to_csv(output / f"{stem}_monthly.csv", index=False)
    excluded.to_csv(output / f"{stem}_audit_excluded.csv", index=False)
    fits.to_csv(output / f"{stem}_technical_fits.csv", index=False)
    # A full calendar is retained, but refused-month numerical vectors live only in the separate record.
    published = table.copy()
    denied = computed & ~table.use_in_preliminary
    published.loc[denied, "calculation_status"] = "excluded_by_audit"
    value_fields = [f"{prefix}_{level}" for prefix in ("H", "A", "R2", "phase_status") for level in sensors]
    published.loc[denied, value_fields] = np.nan
    published["skip_reason"] = published.reason.where(~computed, "")
    published.loc[denied, "skip_reason"] = published.loc[denied, "admission_reason"]
    failed = published.calculation_status.eq("failed")
    published.loc[failed, "skip_reason"] = published.loc[failed, "calculation_error"]
    published.to_csv(output / f"{stem}_trajectories.csv", index=False)
    reread = pd.read_csv(output / f"{stem}_trajectories.csv")
    numeric_fields = [f"{prefix}_{level}" for prefix in ("H", "A", "R2") for level in sensors]
    np.testing.assert_allclose(reread[numeric_fields], published[numeric_fields], atol=1e-7, rtol=1e-7, equal_nan=True)
    for kind in ("H", "A"):
        ch.save_preliminary_interactive(published, kind, output / f"{stem}_{kind}_interactive.html")
    assert protected_hashes == {str(p): digest(p) for p in protected}
    report = ["# Предварительные траектории W_R4_C-4 по полному техническому календарю", "",
        f"Архив: {summary['first_record']} — {summary['last_record']}; 157 месяцев, {len(calendar)} суток.",
        f"Рассчитано {summary['calculated_days']} суток; на графиках {len(shown)}: "
        f"{summary['calculated_by_monthly_admission'].get('confirmed', 0)} в подтверждённых месяцах и "
        f"{summary['calculated_by_monthly_admission'].get('unconfirmed', 0)} без установленного месячного допуска.",
        f"Точки: {summary['first_plotted_day']} — {summary['last_plotted_day']}; добавлено к прежним 789: {summary['added_to_previous_789']}.", "",
        "Опубликованное подтверждение относится к месяцу. Наш способ извлечения суточной гармоники отличается от не полностью описанного способа I.G.",
        "Ни техническая вычислимость, ни опубликованный месячный статус не устанавливают физическую достоверность каждой фазы.",
        "Окружение выходит в соседний неподтверждённый месяц: " + ", ".join(summary["context_caveat_dates"]) + ".",
        "Отказ аудита: " + ", ".join(summary["audit_excluded_dates"]) + "; только отдельная техническая запись, без точек на графиках.", "",
        f"[H(d)]({stem}_H_interactive.html) · [A(d)]({stem}_A_interactive.html) · [Таблица]({stem}_trajectories.csv) · [Месяцы]({stem}_monthly.csv)",
        "Открыть HTML в браузере (в PyCharm: Open in → Browser). Вращение мышью, масштабирование колесом; даты, координаты и R² — при наведении.",
        "Заполненный круг: подтверждённый месяц; пустой ромб: допуск не установлен. Цвет — дата. Щелчок по легенде переключает группу точек.",
        "0 и 24 ч отождествлены, связи разрезаны на гранях. Пропуски и границы статусов не соединены. Связи не являются интерполяцией измерений.",
        "Размер точки отражает min R² без порога; известная чувствительность D3 22.11.2017 отмечена небольшим кольцом на H(d).", "",
        f"Неожиданных отказов: {summary['failed_A_days']}; B/C пропущены: {summary['skipped_B_C_days']}, причины сохранены в таблице.",
        "Прежние 783 + 6 суток и диагностики сверены при atol = rtol = 1e-7. Исходные результаты и основной отчёт побайтно сохранены.",
        "Воспроизведение: prepare_chapter02.py --preliminary. Один полный пересчёт технического A по готовому календарю; без Oracle и нового допуска.", ""]
    (output / f"{stem}_report.md").write_text("\n".join(report))
    checks = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), summary=summary, comparisons=comparisons,
        input_provenance=str(source_path), input_provenance_sha256=digest(source_path),
        calendar_sha256=source["output_sha256"][f"{source_stem}_daily.csv"],
        protected_sha256=protected_hashes, protected_files_byte_identical=True,
        implementation_sha256={name: digest(HERE / name) for name in ("chapter02.py", "prepare_chapter02.py")},
        output_sha256={p.name: digest(p) for p in output.glob(stem + "*") if p.suffix != ".json"})
    (output / f"{stem}_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2), flush=True)
    return published, fits


FINAL_STEM = 'W_R4_C-4_full_archive_final'
SENSORS = {'D1': 994, 'D2': 1010, 'D3': 1026}
BOUNDS_QUERY = '''SELECT sensorid, COUNT(*) AS row_count, MIN(localdatetime) AS first_record,
MAX(localdatetime) AS last_record FROM leo_west.datavalues
WHERE variableid=9 AND sensorid IN (994,1010,1026) GROUP BY sensorid'''
RAW_QUERY = '''SELECT localdatetime, sensorid, variableid, datavalue
FROM leo_west.datavalues WHERE variableid=9 AND sensorid IN (994,1010,1026)
AND localdatetime >= :start_time AND localdatetime <= :end_time'''


def refresh_final_snapshot(bounds=None):
    """SELECT-only incremental refresh; immutable previous chapter inputs."""
    from sensorDB import SensorDB
    directory = DATA_DIR / 'archive'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'raw.parquet'
    manifest_path = directory/'provenance.json'
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parents = {}
    if path.exists():
        old = json.loads(manifest_path.read_text())
        if digest(path) != old['sha256']: raise ValueError('Snapshot checksum mismatch')
        raw = pd.read_parquet(path)
        parents['previous_snapshot_sha256'] = old['sha256']
    elif (HERE/'data/ch03/provenance.json').exists():
        old = json.loads((HERE/'data/ch03/provenance.json').read_text())
        parts=[]
        for name, meta in sorted(old['raw_files'].items()):
            p=HERE/'data/ch03/raw'/name.replace('.json','.parquet')
            if digest(p) != meta['sha256']: raise ValueError(f'Source checksum mismatch: {p.name}')
            frame=pd.read_parquet(p)
            parts.append(frame.loc[frame.variableid.eq(9) & frame.sensorid.isin(SENSORS.values())])
            parents[str(p.relative_to(HERE))]=meta['sha256']
        raw=pd.concat(parts,ignore_index=True)
    else:
        raw=pd.DataFrame(columns=['localdatetime','sensorid','variableid','datavalue'])
    began=datetime.now(timezone.utc).isoformat()
    with SensorDB(call_timeout_ms=600000) as db:
        if bounds is None:
            bounds=dict(checked_at_utc=began, query=BOUNDS_QUERY,
                        bounds=db.fetch_dataframe(BOUNDS_QUERY).to_dict('records'))
        inventory=pd.DataFrame(bounds['bounds'])
        if set(inventory.sensorid) != set(SENSORS.values()): raise ValueError('Missing database sensors')
        last=pd.to_datetime(inventory.last_record).max()
        first=pd.to_datetime(inventory.first_record).min()
        if raw.empty:
            chunks=[]
            for year in range(first.year,last.year+1):
                a=max(first,pd.Timestamp(year,1,1))
                b=min(last,pd.Timestamp(year+1,1,1)-pd.Timedelta(seconds=1))
                chunks.append(db.fetch_dataframe(RAW_QUERY,dict(start_time=a.to_pydatetime(),end_time=b.to_pydatetime())))
                print('sensorDB',year,len(chunks[-1]),flush=True)
            raw=pd.concat(chunks,ignore_index=True)
            refresh_start=first
        else:
            # Refresh the last observed calendar day for all sensors, including
            # duplicates at its beginning. Older saved source rows stay immutable.
            refresh_start=pd.to_datetime(raw.localdatetime).max().normalize()
            tail=db.fetch_dataframe(RAW_QUERY,dict(start_time=refresh_start.to_pydatetime(),end_time=last.to_pydatetime()))
            raw=pd.concat([raw.loc[pd.to_datetime(raw.localdatetime).lt(refresh_start)],tail],ignore_index=True)
        raw['localdatetime']=pd.to_datetime(raw.localdatetime)
        for item in inventory.itertuples():
            part=raw.loc[raw.sensorid.eq(item.sensorid)]
            if len(part)!=item.row_count or part.localdatetime.min()!=pd.Timestamp(item.first_record) or part.localdatetime.max()!=pd.Timestamp(item.last_record):
                raise ValueError('Incremental snapshot differs from database counts/bounds; full refresh required')
    raw=raw.sort_values(['localdatetime','sensorid'],kind='stable').reset_index(drop=True)
    temporary=path.with_suffix('.partial.parquet');raw.to_parquet(temporary,index=False,compression='zstd')
    temporary.replace(path)
    manifest=dict(extraction_started_at_utc=began, extraction_finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  database_inventory=bounds, rows=len(raw), first_record=str(first), last_record=str(last),
                  sha256=digest(path), file='raw.parquet', refresh_start=str(refresh_start),
                  source_query=RAW_QUERY, parent_hashes=parents,
                  sensor_ids=SENSORS, depth_cm={'D1':5,'D2':20,'D3':35},variable_id=9,
                  time_basis='Oracle localdatetime, no timezone conversion; original rows retained')
    manifest_path.write_text(json.dumps(manifest,default=str,indent=2)+'\n')
    print('Fresh snapshot:',len(raw),'rows;',first,'—',last,flush=True)
    return raw, manifest


def prepare_final(refresh=False):
    """One full recomputation under the approved rules, without historical gates."""
    import importlib.metadata
    directory=DATA_DIR/'archive'
    if refresh:
        raw, provenance=refresh_final_snapshot()
    else:
        provenance=json.loads((directory/'provenance.json').read_text())
        if hashlib.sha256((directory/'raw.parquet').read_bytes()).hexdigest()!=provenance['sha256']:
            raise ValueError('Snapshot checksum mismatch')
        raw=pd.read_parquet(directory/'raw.parquet')
    output=HERE/'output/ch02';output.mkdir(parents=True,exist_ok=True)
    start=raw.localdatetime.min().normalize()
    end=raw.localdatetime.max().normalize()+pd.Timedelta(days=1)
    exclusions=pd.DataFrame([dict(level=l,start=a,end=b) for l in SENSORS for a,b in ch.STUCK_INTERVALS])
    print('Cleaning work view',flush=True)
    work=ch.clean_admission_measurements(raw,SENSORS,exclusions)
    work.to_parquet(directory/'working_measurements.parquet',index=False,compression='zstd')
    events=work.loc[work.exclusion_reason.ne('') | work.doubt_reason.ne('') | work.exact_duplicate_extra.gt(0)]
    events.to_parquet(directory/'measurement_decisions.parquet',index=False,compression='zstd')
    print('Monthly admission:',len(work),'unique timestamps;',int(work.doubt_reason.ne('').sum()),'flagged',flush=True)
    monthly,day_metrics=ch.monthly_admission(work,SENSORS,start,end)
    monthly.to_csv(output/f'{FINAL_STEM}_sensor_months.csv',index=False)
    day_metrics.to_csv(output/f'{FINAL_STEM}_monthly_day_metrics.csv',index=False)
    print('Calendar and required context',flush=True)
    calendar=ch.final_admission_calendar(work,SENSORS,monthly,start,end,exclusions)
    print('H/A eligible:',int(calendar.category.eq('A').sum()),flush=True)
    table,fits=ch.calculate_trajectories(work,SENSORS,calendar,exclusions)
    for prefix in ('H','A','R2','phase_status'):
        for l in SENSORS:
            if f'{prefix}_{l}' not in table: table[f'{prefix}_{l}']=np.nan
    table['skip_reason']=table.reason.where(~table.calculation_status.eq('computed'),'')
    failed=table.calculation_status.eq('failed')
    table.loc[failed,'skip_reason']=table.loc[failed,'calculation_error']
    table.to_csv(output/f'{FINAL_STEM}_trajectories.csv',index=False)
    fits.to_csv(output/f'{FINAL_STEM}_daily_fits.csv',index=False)
    # Record every exclusion/flag count per sensor-month, including failed months.
    work['month']=work.localdatetime.dt.strftime('%Y-%m')
    for l,sid in SENSORS.items():
        selected=work.loc[work.sensorid.eq(sid)]
        for m,part in selected.groupby('month'):
            mask=monthly.level.eq(l)&monthly.month.eq(m)
            monthly.loc[mask,'source_rows']=int(part.source_rows.sum())
            monthly.loc[mask,'unique_timestamps']=len(part)
            monthly.loc[mask,'exact_duplicate_extra']=int(part.exact_duplicate_extra.sum())
            monthly.loc[mask,'doubtful_samples']=int(part.doubt_reason.ne('').sum())
            monthly.loc[mask,'constant_1h_samples']=int(part.constant_run_hours.ge(1).sum())
            monthly.loc[mask,'constant_2h_samples']=int(part.constant_run_hours.ge(2).sum())
            for reason in ['service_code','outside_0_3000','nonfinite','conflicting_timestamp','approved_individual_exclusion','isolated_7_sample_MAD_spike']:
                monthly.loc[mask,reason]=int(part.exclusion_reason.eq(reason).sum())
    monthly.to_csv(output/f'{FINAL_STEM}_sensor_months.csv',index=False)
    joint=[]
    for m,part in table.groupby('month',sort=True):
        record=dict(month=m,calendar_days=len(part),status=part.admission_status.iloc[0],
                    computed_days=int(part.calculation_status.eq('computed').sum()),
                    technical_days=int(part.technical_category.eq('A').sum()),
                    unresolved_context_days=int(part.unresolved_context.sum()))
        for l in SENSORS:
            item=monthly.loc[monthly.month.eq(m)&monthly.level.eq(l)].iloc[0]
            for k in ['status','valid_days','median_concentration','median_raw_amplitude','reason']:
                record[f'{l}_{k}']=item[k]
        joint.append(record)
    joint=pd.DataFrame(joint);joint.to_csv(output/f'{FINAL_STEM}_monthly.csv',index=False)
    computed=table.loc[table.calculation_status.eq('computed')]
    assert computed.admission_status.eq('pass').all()
    assert computed.unresolved_samples_required.eq(0).all()
    assert not computed.unresolved_context.any()
    checks=dict(snapshot=provenance, parameters=ch.ADMISSION_PARAMETERS, approved_exclusions=exclusions.to_dict('records'),
        months=len(joint),calendar_days=len(table), sensor_month_statuses=monthly.status.value_counts().to_dict(),
        joint_month_statuses=joint.status.value_counts().to_dict(),final_days=len(computed),
        first_final_day=str(computed.date.min()),last_final_day=str(computed.date.max()),
        failed_daily_calculations=int(failed.sum()),undefined_H_days=int(computed[[f'H_{l}' for l in SENSORS]].isna().any(axis=1).sum()),
        flag_counts=work.doubt_reason.value_counts().to_dict(),exclusion_counts=work.exclusion_reason.value_counts().to_dict(),
        exact_duplicate_extra=int(work.exact_duplicate_extra.sum()),
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','plotly','pyarrow','oracledb']})
    for kind in ('H','A'):
        fig=ch.plot_trajectory(table,kind,final=True)
        fig.savefig(output/f'{FINAL_STEM}_{kind}.png',dpi=180)
        ch.plt.close(fig)
        ch.save_preliminary_interactive(table,kind,output/f'{FINAL_STEM}_{kind}_interactive.html',final=True)
    keys=[f'{k}_{l}' for k in ('H','A','R2') for l in SENSORS]
    reread=pd.read_csv(output/f'{FINAL_STEM}_trajectories.csv')
    np.testing.assert_allclose(table[keys],reread[keys],atol=1e-7,rtol=1e-7,equal_nan=True)
    checks['python_version'] = sys.version
    checks['numeric_csv_round_trip']='passed atol=rtol=1e-7'
    checks['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [directory/'raw.parquet',directory/'working_measurements.parquet',directory/'measurement_decisions.parquet']}
    checks['implementation_hashes']={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'chapter02.py',HERE/'prepare_chapter02.py',PROJECT/'Project_description/sensorDB/diurnal_analysis.py',PROJECT/'Project_description/sensorDB/plot_column_timeseries.py']}
    checks['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob(FINAL_STEM+'*') if p.suffix!='.json'}
    (output/f'{FINAL_STEM}_checks.json').write_text(json.dumps(checks,ensure_ascii=False,default=str,indent=2)+'\n')
    print(json.dumps({k:v for k,v in checks.items() if k not in ['snapshot','input_hashes','implementation_hashes','output_sha256']},ensure_ascii=False,default=str,indent=2),flush=True)
    return table,monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--exported-at-utc")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--archive", action="store_true", help="Полный локальный проход; без Oracle и рисунков")
    parser.add_argument("--preliminary", action="store_true", help="Все технические A: пересчёт и предварительные HTML по готовому календарю")
    parser.add_argument("--audit-pdf", type=Path, default=Path("/home/dimitri/Documents/Bio2Projects/CO2/LEO_CO2_sensor_audit_2026-06-30.pdf"))
    parser.add_argument("--final", action="store_true", help="Утверждённый месячный допуск и окончательные H/A всего снимка")
    parser.add_argument("--refresh", action="store_true", help="Перед --final обновить снимок SELECT из sensorDB")
    args = parser.parse_args()
    if args.refresh and not args.final: parser.error("--refresh requires --final")
    if args.final:
        if args.archive or args.preliminary: parser.error("--final is exclusive")
        prepare_final(refresh=args.refresh)
    elif args.preliminary:
        if args.archive:
            parser.error("Выберите --archive или --preliminary")
        prepare_preliminary()
    elif args.archive:
        process_archive(args.audit_pdf)
    else:
        prepare_snapshot(args.source_csv, args.exported_at_utc, args.evidence_dir)
