"""Stage 0.1-B: lossless continuous hydrology input from the frozen local audit.

No Oracle, CO2, event recognition, resampling of values, or water balance.
The nominal 15-minute clock is used ONLY for coverage and plot line breaks.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage01_irrigation_audit import END, OUT, ROOT, START, dump_json, intervals, load_raw

KEYS = ((19, 6), (317, 6), (668, 7), (889, 7)) + tuple((s, 17) for s in range(1066, 1076))
SOURCE_COLUMNS = ['valueid', 'localdatetime', 'sensorid', 'variableid', 'datavalue']
OUTPUT = ROOT / 'Project_description/Research_log/output/stage01_irrigation'
SERIES_PATH = OUT / 'pilot_series.parquet'
PLOT_PATH = OUTPUT / 'pilot_hydrology_17months.png'
NODATA = -9999.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonservice(values):
    """A storage/availability flag, never a metrological acceptance criterion."""
    return np.isfinite(values) & values.ne(NODATA)


def channel_coverage(g, start, end):
    grid = pd.date_range(start, end, freq='15min', inclusive='left')
    slot = g.localdatetime.dt.floor('15min')
    present = nonservice(g.datavalue)
    good = g.loc[present]
    dt = g.localdatetime.diff().dt.total_seconds().dropna()
    counts = slot.value_counts()
    return dict(
        expected_slots=len(grid), rows=len(g), observed_slots=slot.nunique(),
        nonservice_rows=int(present.sum()), nonservice_slots=slot.loc[present].nunique(),
        service_rows=int(g.datavalue.eq(NODATA).sum()),
        nonfinite_rows=int((~np.isfinite(g.datavalue)).sum()),
        missing_observation_slots=int((~grid.isin(slot)).sum()),
        no_nonservice_slots=int((~grid.isin(slot.loc[present])).sum()),
        coverage_percent=100 * slot.loc[present].nunique() / len(grid),
        duplicate_timestamp_rows=int(g.localdatetime.duplicated(keep=False).sum()),
        ambiguous_slots=int(counts.gt(1).sum()),
        first_time=g.localdatetime.min(), last_time=g.localdatetime.max(),
        first_nonservice=good.localdatetime.min(), last_nonservice=good.localdatetime.max(),
        min_nonservice=good.datavalue.min(), max_nonservice=good.datavalue.max(),
        modal_step_seconds=dt.mode().iloc[0] if len(dt) else None,
    )


def plot_arrays(g):
    """Plot native times, hiding service values and breaking at unobserved slots.

    NaNs inserted here are drawing separators only. No synthetic row is written
    into the exported measurement table, and negative MPS values are retained.
    """
    t = g.localdatetime.to_numpy()
    y = g.datavalue.where(nonservice(g.datavalue)).to_numpy(copy=True)
    slots = g.localdatetime.dt.floor('15min')
    positions = np.flatnonzero(slots.diff().gt(pd.Timedelta(minutes=15)))
    if len(positions):
        t = np.insert(t, positions, t[positions])
        y = np.insert(y, positions, np.nan)
    return t, y


def write_plot(series, wetting):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = {(s, v): g for (s, v), g in series.groupby(['sensorid', 'variableid'])}
    fig, axes = plt.subplots(3, 1, figsize=(17, 11.6), sharex=True)
    fig.subplots_adjust(left=.11, right=.98, bottom=.12, top=.86, hspace=.30)
    fig.suptitle('LEO West: гидрологический вход ограниченного пилота 0.1-B\n'
                 '01.05.2025–30.09.2026; исходное время SensorDB', fontsize=16, y=.97)
    fig.legend(handles=[Patch(facecolor='#a7c9e7', alpha=.45,
                             label='37 дат согласованного увлажнения: выделены сутки, не границы полива')],
               loc='upper center', bbox_to_anchor=(.55, .91), frameon=False, fontsize=11)
    panel_specs = [
        (axes[0], [(19, 6), (317, 6)], ['#0072B2', '#D55E00'],
         ['5TM 19 — 5 см', '5TM 317 — 35 см'], 'Объёмная влажность, %',
         'Соседняя точка (2,4), в 1 м от W_x1_y4; глубины 5 и 35 см'),
        (axes[1], [(668, 7), (889, 7)], ['#009E73', '#8b3fa8'],
         ['MPS-2 668 — 20 см', 'MPS-2 889 — 50 см'], 'Матричный потенциал, кПа',
         'W_x1_y4, точка (1,4); близко к влажному пределу, точность ограничена'),
        (axes[2], [(s, 17) for s in range(1066, 1076)], plt.get_cmap('tab10').colors,
         [str(s) for s in range(1066, 1076)], 'Сохранённая масса\nкаждой опоры, кг',
         'Model3130: десять отдельных выходов Mass (VariableID 17); не поданная вода'),
    ]
    point_counts = {}
    marker_counts = []
    for ax, keys, colors, labels, ylabel, title in panel_specs:
        for date in wetting.date:
            day = pd.Timestamp(date)
            ax.axvspan(day, day + pd.Timedelta(days=1), color='#a7c9e7', alpha=.35, lw=0, zorder=0)
        marker_counts.append(len(ax.patches))
        for key, color, label in zip(keys, colors, labels):
            g = groups[key]
            t, y = plot_arrays(g)
            ax.plot(t, y, lw=.65 if key[1] != 17 else .5, color=color, label=label)
            point_counts[f'{key[0]}/{key[1]}'] = int(np.isfinite(y).sum())
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, loc='left', fontsize=11, pad=9)
        ax.grid(axis='y', alpha=.23)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', ncol=5 if ax is axes[2] else 2,
                  fontsize=9, framealpha=.95, handlelength=2)
    axes[2].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_xlim(START, END)
    fig.text(.11, .035, 'Показаны все неслужебные отсчёты. Пропуски и −9999 разрывают линии. '
             'Интерполяции, коррекции времени и распознавания поливов нет.', fontsize=10)
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=170, metadata={'Software': 'CO2Flux stage01 hydrology pilot'})
    plt.close(fig)
    return dict(plotted_nonservice_points=point_counts,
                wetting_date_spans_per_panel=marker_counts, panels=3,
                marker_meaning='calendar wetting days, not irrigation event boundaries')


def protected_changes():
    baseline = json.loads((OUT / 'pilot_protected_before.json').read_text())
    allowed = {
        'Project_description/Research_log/README.md',
        'Project_description/Research_log/PLAN.md',
        'Project_description/Research_log/JOURNAL.md',
        'Project_description/Research_log/details/00_irrigation_audit.md',
        'Project_description/Research_log/data/stage01_irrigation/README.md',
        'Project_description/Research_log/data/stage01_irrigation/.gitignore',
        'Project_description/sensorDB/stage01_irrigation_audit.py',
    }
    changed = [p for p, old in baseline['files'].items()
               if not (ROOT / p).is_file() or sha256(ROOT / p) != old]
    return baseline, changed, sorted(set(changed) - allowed)


def write_pilot_tables(summary, monthly, wetting):
    """Refresh only the two result tables in the existing stage 0.1 details."""
    import re
    path = ROOT / 'Project_description/Research_log/details/00_irrigation_audit.md'
    body = path.read_text()
    def replace(name, table):
        nonlocal body
        pattern = rf'<!-- BEGIN {name} -->.*?<!-- END {name} -->|<!-- {name} -->'
        block = f'<!-- BEGIN {name} -->\n{table}\n<!-- END {name} -->'
        body, n = re.subn(pattern, lambda _: block, body, flags=re.S)
        if n != 1:
            raise RuntimeError(f'Missing/repeated stage 0.1-B report marker: {name}')
    s = summary.set_index('sensorid')
    lines = ['| SensorID / VariableID | Единица | Пространственная привязка | Неслужебных | `−9999` | Покрытие, % |',
             '|---|---|---|---:|---:|---:|']
    for sensor, location in [(19, '(2,4), 5 см'), (317, '(2,4), 35 см'),
                             (668, '(1,4), 20 см'), (889, '(1,4), 50 см'),
                             (1066, 'десять опор склона')]:
        r = s.loc[sensor]
        label = '1066–1075 / 17, каждая опора' if sensor == 1066 else f'{sensor} / {int(r.variableid)}'
        lines.append(f'| {label} | {r.unit} | {location} | {int(r.nonservice_rows)} | {int(r.service_rows)} | {r.coverage_percent:.6f} |')
    replace('PILOT_SUMMARY', '\n'.join(lines))
    lines = ['| Месяц | Номинальных ячеек | VWC 19 | VWC 317 | MPS 668 | MPS 889 | Каждая масса | Дат увлажнения |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for month, group in monthly.groupby('month', sort=True):
        g = group.set_index('sensorid')
        masses = group[group.variableid.eq(17)]
        if masses.nonservice_rows.nunique() != 1:
            raise RuntimeError('Mass-channel coverage differs; show separate columns')
        counts = [int(g.loc[sensor, 'nonservice_rows']) for sensor in [19, 317, 668, 889, 1066]]
        count_dates = int(wetting.date.str.startswith(month).sum())
        lines.append('| ' + month + ' | ' + str(int(group.expected_slots.iloc[0])) + ' | ' +
                     ' | '.join(map(str, counts + [count_dates])) + ' |')
    replace('PILOT_MONTHLY_COVERAGE', '\n'.join(lines))
    path.write_text(body)


def build():
    """Build and verify the pilot input using only the existing local snapshot."""
    raw_manifest = json.loads((OUT / 'raw_manifest.json').read_text())
    if len(raw_manifest) != 17 or not all(sha256(OUT / p) == m['sha256'] for p, m in raw_manifest.items()):
        raise RuntimeError('Frozen source snapshot is missing or has changed')
    raw = load_raw()
    mask = pd.MultiIndex.from_frame(raw[['sensorid', 'variableid']]).isin(KEYS)
    selected = raw.loc[mask, SOURCE_COLUMNS].reset_index(drop=True)
    actual_keys = set(zip(selected.sensorid, selected.variableid))
    if actual_keys != set(KEYS):
        raise RuntimeError('A required hydrologic channel is missing')
    metadata = pd.read_csv(OUT / 'channels.csv')
    metadata = metadata[metadata.apply(lambda r: (r.sensorid, r.variableid) in KEYS, axis=1)]
    units = pd.read_csv(OUT / 'units.csv').set_index('unitsid').unitsabbreviation
    export = selected.copy()
    export['source_table'] = 'LEO_WEST.DATAVALUES'
    export['source_part'] = export.localdatetime.dt.strftime('raw/measurements_%Y-%m.parquet')
    export['is_service_value'] = export.datavalue.eq(NODATA)
    export['is_finite'] = np.isfinite(export.datavalue)
    export['is_nonservice_value'] = nonservice(export.datavalue)
    metadata_by_key = metadata.set_index(['sensorid', 'variableid'])
    row_keys = pd.MultiIndex.from_frame(export[['sensorid', 'variableid']])
    for col, source_col in [('sensorcode', 'sensorcode'), ('localx_m', 'localx'),
                            ('localy_m', 'localy'), ('depth_m', 'boxz')]:
        export[col] = metadata_by_key[source_col].reindex(row_keys).to_numpy()
    export['unit'] = metadata_by_key.variableunitsid.map(units).reindex(row_keys).to_numpy()
    export['spatial_role'] = np.where(export.variableid.eq(6), 'adjacent_to_W_x1_y4',
                            np.where(export.variableid.eq(7), 'W_x1_y4', 'slope_load_support_not_a_soil_point'))
    export.to_parquet(SERIES_PATH, index=False, compression='zstd')

    summary, monthly, gap_rows, cadence_rows = [], [], [], []
    grid = pd.date_range(START, END, freq='15min', inclusive='left')
    months = pd.date_range(START, END, freq='MS')
    for s, v in KEYS:
        g = selected[selected.sensorid.eq(s) & selected.variableid.eq(v)]
        meta = metadata_by_key.loc[(s, v)]
        info = dict(sensorid=s, variableid=v, sensorcode=meta.sensorcode,
                    source_table='LEO_WEST.DATAVALUES', unit=units.loc[meta.variableunitsid],
                    localx_m=meta.localx, localy_m=meta.localy, depth_m=meta.boxz,
                    location_role='adjacent_to_W_x1_y4' if v == 6 else
                    'W_x1_y4' if v == 7 else 'slope_load_support_not_a_soil_point',
                    interpretation='technical_nonservice_coverage_not_metrological_certification')
        summary.append({**info, **channel_coverage(g, START, END)})
        for a, b in zip(months[:-1], months[1:]):
            x = g[g.localdatetime.ge(a) & g.localdatetime.lt(b)]
            monthly.append({**info, 'month': a.strftime('%Y-%m'), **channel_coverage(x, a, b)})
        slots = g.localdatetime.dt.floor('15min')
        good_slots = slots.loc[nonservice(g.datavalue)]
        for kind, gap_mask in [
            ('missing_observation', ~grid.isin(slots)),
            ('service_or_nonfinite_observation', grid.isin(slots) & ~grid.isin(good_slots)),
            ('no_nonservice_observation', ~grid.isin(good_slots)),
        ]:
            for a, b, n in intervals(gap_mask, grid):
                gap_rows.append(dict(sensorid=s, variableid=v, kind=kind,
                                     start_slot=a, end_slot_exclusive=b, slots=n))
        for seconds, n in g.localdatetime.diff().dt.total_seconds().dropna().value_counts().items():
            cadence_rows.append(dict(sensorid=s, variableid=v, interval_seconds=seconds, count=int(n)))
    summary = pd.DataFrame(summary)
    monthly = pd.DataFrame(monthly)
    summary.to_csv(OUT / 'pilot_series_summary.csv', index=False)
    monthly.to_csv(OUT / 'pilot_monthly_coverage.csv', index=False)
    pd.DataFrame(gap_rows).to_csv(OUT / 'pilot_missing_intervals.csv', index=False)
    pd.DataFrame(cadence_rows).to_csv(OUT / 'pilot_native_cadence.csv', index=False)
    wetting = pd.read_csv(OUT / 'wetting_dates.csv')
    if len(wetting) != 37 or wetting.date.nunique() != 37:
        raise RuntimeError('The accepted 37-date wetting register has changed')
    plotting = write_plot(export, wetting)
    dump_json(OUT / 'pilot_plot_spec.json', plotting)

    readback = pd.read_parquet(SERIES_PATH)
    old_monthly = pd.read_csv(OUT / 'monthly_coverage.csv').set_index(['sensorid', 'variableid', 'month'])
    coverage_match = all(
        r.rows == old_monthly.loc[(r.sensorid, r.variableid, r.month), 'rows'] and
        r.nonservice_rows == old_monthly.loc[(r.sensorid, r.variableid, r.month), 'usable_rows'] and
        r.service_rows == old_monthly.loc[(r.sensorid, r.variableid, r.month), 'service_rows']
        for r in monthly.itertuples()
    )
    baseline, changed, unexpected = protected_changes()
    checks = {
        'source_sha256_all_17_parts': all(sha256(OUT / p) == m['sha256'] for p, m in raw_manifest.items()),
        'all_source_columns_bitwise_preserved': readback[SOURCE_COLUMNS].equals(selected) and
            np.array_equal(readback.datavalue.to_numpy().view(np.uint64), selected.datavalue.to_numpy().view(np.uint64)),
        'no_extra_or_removed_measurement_rows': len(readback) == len(selected),
        'all_14_required_channels': actual_keys == set(KEYS),
        'all_native_times_in_authorized_period': bool(readback.localdatetime.ge(START).all() and readback.localdatetime.lt(END).all()),
        'valueids_unique': readback.valueid.is_unique,
        'service_values_preserved': int(readback.is_service_value.sum()) == int(selected.datavalue.eq(NODATA).sum()),
        'negative_MPS_values_retained': bool(readback.loc[readback.variableid.eq(7) & readback.is_nonservice_value, 'datavalue'].lt(0).all()),
        'no_ambiguous_nominal_slots': int(summary.ambiguous_slots.sum()) == 0,
        'no_duplicate_native_timestamps': int(summary.duplicate_timestamp_rows.sum()) == 0,
        'all_238_channel_months': len(monthly) == 14 * 17,
        'native_units_preserved': bool(readback.loc[readback.variableid.eq(6), 'unit'].eq('%').all() and
            readback.loc[readback.variableid.eq(7), 'unit'].eq('kPa').all() and
            readback.loc[readback.variableid.eq(17), 'unit'].eq('kg').all()),
        'soil_coordinates_and_depths_match_accepted_input': all(
            tuple(metadata_by_key.loc[k, ['localx', 'localy', 'boxz']]) == xyz
            for k, xyz in { (19,6):(2.,4.,.05), (317,6):(2.,4.,.35),
                            (668,7):(1.,4.,.20), (889,7):(1.,4.,.50)}.items()),
        'coverage_reconciles_to_source_rows': int(monthly.rows.sum()) == len(selected),
        'coverage_matches_independent_existing_audit': coverage_match,
        'all_37_accepted_wetting_dates_marked_in_every_panel': plotting['wetting_date_spans_per_panel'] == [37, 37, 37],
        'plot_has_every_nonservice_point': plotting['plotted_nonservice_points'] ==
            {f'{r.sensorid}/{r.variableid}': r.nonservice_rows for r in summary.itertuples()},
        'no_old_artifact_or_other_stage_changed': not unexpected,
    }
    validation = dict(passed=all(checks.values()), checks=checks,
                      checked_utc=datetime.now(timezone.utc).isoformat(), rows=len(selected),
                      channels=len(KEYS), months=17, wetting_dates=37,
                      nonservice_rows=int(summary.nonservice_rows.sum()),
                      service_rows=int(summary.service_rows.sum()),
                      allowed_changed_preexisting_files=changed, unexpected_changed_files=unexpected,
                      status='0.1-B technical hydrologic input for the authorized limited pilot only',
                      exact_irrigation_calendar='not obtained; criterion remains unmet')
    dump_json(OUT / 'pilot_validation.json', validation)
    inputs = ['raw_manifest.json', 'wetting_dates.csv', 'channels.csv', 'units.csv',
              'monthly_coverage.csv', 'pilot_protected_before.json']
    output_files = [SERIES_PATH, PLOT_PATH] + [OUT / name for name in
                    ['pilot_series_summary.csv', 'pilot_monthly_coverage.csv',
                     'pilot_missing_intervals.csv', 'pilot_native_cadence.csv', 'pilot_plot_spec.json']]
    provenance = dict(
        base_commit=baseline['base_commit'], period_start_inclusive=START, period_end_exclusive=END,
        time_policy='Oracle LOCALDATETIME unchanged; no timezone attachment or correction',
        source_inputs={name: sha256(OUT / name) for name in inputs}, source_parts=raw_manifest,
        code_sha256=sha256(__file__), entrypoint_sha256=sha256(Path(__file__).with_name('stage01_irrigation_audit.py')),
        outputs={str(p.relative_to(ROOT)): dict(sha256=sha256(p), bytes=p.stat().st_size) for p in output_files},
        command='MPLCONFIGDIR=/tmp/co2flux-mpl python3 Project_description/sensorDB/stage01_irrigation_audit.py pilot',
        python=platform.python_version(), pandas=pd.__version__, rows=len(selected),
        policy='lossless subset; no interpolation, resampling of values, calendar inference, or metrological certification',
        technical_flag='is_nonservice_value = finite and datavalue != -9999; not a reliability label',
        coverage_clock='15-minute slots only for availability counts and line breaks; raw time retained',
        spatial_limits='VWC at (2,4), depths 5/35 cm; MPS at (1,4), depths 20/50 cm; mass at ten slope supports',
        stored_mass='VariableID 17 in kg retained; VariableID 15 is never converted to kg',
    )
    dump_json(OUT / 'pilot_provenance.json', provenance)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation['passed']:
        raise RuntimeError('Pilot input did not pass all technical checks')
    write_pilot_tables(summary, monthly, wetting)


if __name__ == '__main__':
    build()
