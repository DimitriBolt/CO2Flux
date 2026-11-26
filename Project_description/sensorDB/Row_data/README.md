Row_data — how CO2 row data files are created
=============================================

Здесь хранятся ряды CO₂ по склонам. Для западного склона рабочим набором служит папка с месячным отбором.

- [LEO_West/](LEO_West) — рабочие ряды LEO West с месячным допуском ≥ 2 ppm. [Порядок обновления](#west-admitted).
- `LEO_East/`  — placeholder; run the generator to populate
- `LEO_Center/`— placeholder; run the generator to populate

Legacy export for LEO East and Center
------------------------------------

1. The workbook `Sensors_Description/variables_schema.xlsx` contains inventory and query templates.
   The script `scripts/update_co2_sheet.py` shows how sensor inventory rows and Oracle queries are constructed.

2. The generator script `sensorDB/save_leo_slope_co2_row_data.py` queries the `sensors` table in each
   slope schema and saves basalt sensors (GMM222) and near-surface air sensors (LI-7000) to parquet.

Usage
-----

Set Oracle credentials in `~/Documents/.env` then run (example):

```bash
python3 Project_description/sensorDB/save_leo_slope_co2_row_data.py east
python3 Project_description/sensorDB/save_leo_slope_co2_row_data.py center
```

Notes
-----
- The legacy script uses the slope letter and location in each filename.
- If Oracle environment (Instant Client) is missing the script will attempt to set `LD_LIBRARY_PATH` and
  restart the process; if that fails, ensure `ORACLE_CLIENT_LIB_DIR` is set in your environment.
- Existing files are preserved; the script will skip files that already exist.


Full versioned LEO West refresh (53 channels)
-------------------------------------------

The permanent command for **«Обнови все 53 канала»** is, from the repository root:

```bash
make co2-refresh
make co2-status
```

`co2-refresh` uses `../co2_refresh.py`. Unlike the legacy generator above, it
**never skips a sensor because an old archive exists**. The existing
`LEO_West/`, research snapshots and previous full-refresh versions are untouched.

- If the latest version is incomplete, resume its durable manifest and raw parts.
- If the latest version is complete (including diagnostics), create a new UTC-named
  version and fetch the entire available Oracle history again.
- A failed diagnostic stage resumes diagnostics using the completed new download.
- `co2-status` reads local metadata only: status, channels complete/with saved rows,
  row counts, blocks, errors, diagnostic status, version path and log path.
- An exclusive file lock prevents two writers from downloading the same version.

Autonomous terminal session:

```bash
tmux new-session -d -s co2-full-refresh -c /home/dimitri/PycharmProjects/CO2Flux 'make co2-refresh'
tmux attach -t co2-full-refresh
# Detach without stopping the job: Ctrl-b, then d.
make co2-status
```

Before launching a second session, check `tmux list-sessions` and `make co2-status`.
The session runs independently of Codex; computer shutdown/sleep still interrupts
network work. After restart, the same `make co2-refresh` command resumes confirmed
progress. No scheduler, incremental refresh or automatic archive deletion is used.
The tmux session may disappear after the command finishes; the manifest and log
remain the authority for completion.

Storage:

```text
Project_description/sensorDB/Row_data/LEO_West_full/
  latest.json
  writer.lock
  <UTC-version>/
    manifest.json
    download.log
    raw/<block-id>/part-000000.parquet
    diagnostics/
```

`latest.json` points to the current version. Every version retains its own raw
parts and SHA-256 manifest. Logs use the host's local clock; manifest timestamps
are explicit UTC. Oracle measurement `LOCALDATETIME` is neither converted nor
assumed to share a verified timezone with another source.

Scope and identity
------------------

The loader reads the current CO2 worksheet without modifying it, reusing
`scripts/update_co2_sheet.selected_records`. It requires exactly 48 West GMM222
and 5 West LI-COR at workbook height 0.25 m. It verifies sensorcode, table, units,
X/Y and basalt depth against Oracle, and checks the actual columns and primary key.
Each source is identified by table + sensorid + variableid (9 basalt, 56 air).
Excel air Z=0.25 m and Oracle `LOCATIONS.BOXZ=0.05` use an unresolved geometry
mapping; both values are retained, without silently rewriting height or cancelling
the authorized raw download. Every alternative air distance is calculated in X/Y.

Bounds are queried from the live Oracle data, with **no imposed starting year**.
Metadata acquisition captures per-channel count/min/max once; these counts describe
that moment. Monthly before/after counts validate the actual downloaded time
blocks. The multi-hour archive is not claimed to be one global SCN snapshot:
later insertions/edits to Oracle can differ from the initial metadata counts.

All actual source columns are selected, including VALUEID. NULL, service codes,
out-of-range values and repeated measurement timestamps/values are preserved.
Oracle NUMBER/FLOAT are fetched as Python Decimal and stored without float64
conversion. Each raw Parquet part has its own inferred exact decimal schema;
read parts individually if their scales differ. A typical read is
`pyarrow.parquet.read_table(part_path).to_pandas()`; do not assume a single inferred
schema over every part directory. Only explicit diagnostic working copies use
float64 for the existing numerical cleaning function.

Recovery and verification
-------------------------

- Initial blocks are half-open calendar months, clipped to actual Oracle bounds.
- Rows stream in batches of 50,000 ordered by `(LOCALDATETIME, VALUEID)`.
- Both tables have a verified enabled VALUEID primary key and nonnull key fields.
- A batch is written to a temporary Parquet, flushed, row-count checked, renamed,
  hashed and atomically entered into the manifest. Data files and manifest directory
  entries are fsynced. Unconfirmed temporary/orphan files are not counted as progress.
- Restart verifies confirmed parts and continues with a **strictly greater** key.
  Distinct VALUEIDs sharing the same timestamp and concentration remain distinct.
- Before/after Oracle counts by sensor must equal saved counts for a completed block.
- Blocks over one million rows split into weeks/days before transfer. Repeatedly
  failing blocks split the remaining range while preserving confirmed pages.
  Dense single days still stream and checkpoint by key.
- Each block gets at most three attempts per invocation; the connection is reopened
  after an error. Unavailable day blocks are recorded as failures; independent
  blocks continue. Running the command again retries an incomplete version.
- Split parents complete only after all children and a full-range count check.
  Failed blocks never count as a completed archive.

The loader uses the existing thick-mode connector and existing private credentials;
no credentials are copied into the archive or Git. `Makefile` supplies the existing
Oracle library paths. Override `PYTHON` or `ORACLE_LIBRARY_PATH` only if needed on
another host. Dependencies: existing oracledb, python-dotenv, openpyxl, pyarrow,
pandas, numpy and the existing scientific dependencies; tmux for detached execution.

Automatic technical diagnostics
-------------------------------

After **all** blocks complete, `../co2_full_diagnostics.py` runs offline. It checks
all 16 triplets, treating 5/20/35 and 5/20/50 cm equally, and computes every distance
to the five air channels. All equally nearest channels are retained as candidates.
Raw archives are immutable; diagnostic source-row references link back to each
raw part. No historical archive is loaded to shorten or replace this refresh.

The approved `chapter02.clean_admission_measurements` pointwise filters are reused.
The two historical November 2017 exclusions apply only to their original three
sensor IDs. Monthly amplitude/H/A gates are not run. Air has an independent
conservative policy with unresolved numeric-QC flags, not a borrowed basalt upper
limit. Constant runs >=1 h remain diagnostics, not automatic exclusions.

Outputs include per-channel working/decision Parquets, raw/retained daily ranges,
yearly and meteorological-season counts (DJF/MAM/JJA/SON, December in the following
winter year), step distributions, gaps above the existing 1.5 × median-step
diagnostic threshold, plateau tables and all 16 four-channel candidates. Candidate
ranking uses actual common 2-hour bins and days, without imputation. Counts of
calendar years having any data are **not** a claim of complete annual cycles;
season/year tables expose the actual coverage. Different clocks remain provisional.

Scientific results are inserted only into the existing stage 0.2 subsection of
`Research_log/details/02_data_checks.md`. No README/PLAN scientific rewrite, FFT,
frequency/phase calculation or final vertical selection is performed by diagnostics.
Generated data remain local; `diagnostics/summary.json` records output checksums.
The loader does not commit or push automatically; publication of its scientific
result is a separate verification step within the authorized task.

Tests (synthetic source, no Oracle access):

```bash
python3 -B -m unittest discover -s Project_description/sensorDB/tests -p 'test_co2_refresh.py' -v
python3 -B -m unittest discover -s Project_description/sensorDB/tests -p 'test_co2_full_diagnostics.py' -v
```

<a id="west-admitted"></a>

## Рабочие ряды LEO West

[LEO_West](LEO_West) — единственное место публикации обновлённых временных рядов западного склона. В папке 56 Parquet-файлов: 48 базальтовых GMM222, пять приземных LI-COR (25 см) и три служебных воздушных канала AirInflow (1302), AirOutflow (1303), Vent (1305). Создание отдельного набора требует явного согласования с Димитрием.

Обновление из корня репозитория на рабочей машине:

```bash
make co2-west-admitted
```

[Программа](../refresh_west_admitted.py) читает всю историю в одной транзакции Oracle, сверяет количество и крайние даты измерений по каждому каналу и заменяет рабочую папку после проверки всего набора. Git-публикация выполняется отдельным шагом. Команда `make co2-refresh` выше обслуживает локальный архив источника.

В каждом файле индекс `localdatetime` и столбец `datavalue` типа float64. Сохраняются исходные времена и технически допустимые значения в месяцах, прошедших отбор у данного датчика. Допуск требует минимум десяти гарантированно допустимых суток за календарный месяц, границ медианной концентрации в пределах 100–2000 и нижней границы месячной медианы амплитуды суточной гармоники ≥ 2 ppm. Пропуски сохраняются. Пустой ряд имеет ту же схему.

В [selection.json](LEO_West/selection.json) указаны момент снимка Oracle, период каждого исходного ряда, число исходных и сохранённых измерений и контрольные суммы файлов. [months.csv](LEO_West/months.csv) содержит решения и их численные основания по каждому месяцу. Текущий календарный месяц оценивается по данным, поступившим к моменту снимка.

## Поиск одновременно работающих базальтовых вертикалей LEO West

### Историческая выборка при 2 ppm с приоритетом близости

В прежнюю выборку входили девять исходных рядов GMM222. По архиву `20261002T061358Z-264980` проверены все 560 троек из 16 вертикалей. Минимальный диаметр равен 5 м. Среди троек такого диаметра самый длинный совместно допустимый период длится восемь месяцев, с 1 июня 2014 года по 31 января 2015 года включительно.

| Вертикаль | Датчик на 5 см | Датчик на 20 см | Глубокий датчик |
|---|---:|---:|---|
| `W_x-4_y24` | 1006 | 1022 | 1032 (35 см) |
| `W_x-1_y24` | 1007 | 1023 | 1040 (50 см) |
| `W_x1_y24` | 1008 | 1024 | 1041 (50 см) |

После технической очистки измерения всех девяти каналов присутствуют в 22 396 из 23 520 общих 15-минутных интервалов (95,2 %). Сетка служит для проверки покрытия. Parquet-файлы содержат полные ряды с исходными временными метками, значениями и повторами.

Для повторения из корня проекта:

```bash
python3 Project_description/sensorDB/nearby_basalt.py --count 3 --amplitude-min 2 --rank-by distance --bin-minutes 15
```

### Параметры поиска и предыдущие выборки

Для воспроизведения первой опубликованной выборки запустите из корня репозитория.

```bash
python3 Project_description/sensorDB/nearby_basalt.py --count 3 --bin-minutes 120
```

Для каждого базальтового канала используется месячный допуск из главы 2 по очищенной рабочей копии последней завершённой выгрузки. Для общего месяца каждый из трёх датчиков каждой вертикали должен получить статус `pass`. Также нужна хотя бы одна общая ячейка с реальными измерениями всех выбранных каналов после технической очистки. По умолчанию шаг сетки составляет 15 минут. Параметр `--bin-minutes 120` воспроизводит прежнюю двухчасовую проверку. Воздушные каналы в поиске не участвуют. Фактическое число общих ячеек выводится отдельно от месячного допуска.

Параметр `--amplitude-min` задаёт минимальную месячную медиану суточной амплитуды в ppm. По умолчанию используется 8 ppm. Программа хранит рассчитанные месячные границы в `LEO_West_full/_basalt_months/` с привязкой к версии данных и реализации расчёта. При смене порога решение о допуске пересчитывается по сохранённым границам.

По умолчанию `--rank-by distance` ставит на первое место диаметр группы в плоскости X/Y (наибольшее попарное расстояние между вертикалями), затем длительность общего периода. В режиме `--rank-by duration` длительность идёт первой, диаметр вторым. При равенстве программа выбирает период с большим числом общих ячеек на выбранной сетке. Параметры `--count`, `--max-distance` (метры) и `--min-months` задают число вертикалей, предельный диаметр и минимальную длительность периода.

Чтобы найти самый длинный общий период трёх вертикалей при пороге 2 ppm, выполните команду ниже.

```bash
python3 Project_description/sensorDB/nearby_basalt.py --count 3 --amplitude-min 2 --rank-by duration --bin-minutes 15
```

Поиск выводит результат в терминал. Параметр `--output` для экспорта исходных рядов применяют только при явном согласовании отдельной выборки. Каждый файл содержит всю исходную историю одного датчика с индексом `localdatetime` и концентрацией `datavalue` в ppm. Общий допустимый период указан в выводе программы.

По выгрузке `20261002T061358Z-264980` поиск среди 16 вертикалей (560 троек) при пороге 8 ppm с приоритетом близости дал следующий результат:

| X, Y (м) | 5 см | 20 см | Глубокий датчик |
|---|---:|---:|---|
| −1, 18 | 1003 | 1019 | 1030 (35 см) |
| 1, 18 | 1004 | 1020 | 1031 (35 см) |
| 1, 24 | 1008 | 1024 | 1041 (50 см) |

Диаметр группы — 6,325 м. Единственный общий допустимый месяц этой тройки — апрель 2015 года. У каждого датчика 30 гарантированно допустимых суток. Наименьшая из девяти нижних границ месячной медианы суточной амплитуды равна 8,349 ppm (1031). Все 360 двухчасовых интервалов содержат технически допущенные измерения каждого из девяти каналов.

При пороге 2 ppm с приоритетом длительности выбраны `W_x-4_y4` (994, 1010, 1026), `W_x-1_y18` (1003, 1019, 1030) и `W_x4_y4` (997, 1013, 1027). Их диаметр составляет 14,866 м. Общий период включает август 2017 — август 2020, 37 последовательных календарных месяцев. На 15-минутной сетке в 99 287 из 108 192 ячеек есть измерения всех девяти каналов после технической очистки. При прежней проверке на двухчасовой сетке таких ячеек было 12 910 из 13 524.

Предыдущие публикации доступны по ссылкам. [Выборка при 2 ppm с приоритетом длительности](https://github.com/DimitriBolt/CO2Flux/tree/2a65efc20d856a1ef70624303b3fb136fa8467df/Project_description/sensorDB/Row_data/LEO_West) · [Первая выборка при 8 ppm](https://github.com/DimitriBolt/CO2Flux/tree/3bf6d328e4339b4566567ca473ccf12d7d44de62/Project_description/sensorDB/Row_data/LEO_West).
