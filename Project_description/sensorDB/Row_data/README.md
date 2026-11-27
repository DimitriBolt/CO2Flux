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

## История гидрологии LEO West

Три готовые папки доступны в репозитории. Исходные месячные части Oracle
сохраняются локально в каталоге, исключённом из Git.

Три папки содержат всю доступную историю физических величин за вычетом
строк с кодом отсутствия данных `-9999` (`VARIABLES.NODATAVALUE` в Oracle).

| Папка | Физическая величина | VariableID | Единица | Каналов | Строк в Parquet | Исключено `-9999` | Parquet, ГБ |
|---|---|---:|---|---:|---:|---:|---:|
| [LEO_West_VWC](LEO_West_VWC) | Объёмное содержание воды (5TM) | 6 | % | 499 | 132 117 800 | 107 376 424 | 1,076 |
| [LEO_West_Matric_Potential](LEO_West_Matric_Potential) | Матричный водный потенциал (MPS-2) | 7 | kPa | 496 | 193 495 603 | 57 497 130 | 1,409 |
| [LEO_West_Mass](LEO_West_Mass) | Сохранённая масса (Model3130) | 17 | kg | 20 | 18 207 008 | 2 642 | 0,223 |

В Parquet **343 820 411 строк, 1 015 рядов и 2,708 ГБ**
(1 ГБ = 1 000 000 000 байт). Исходный снимок содержит
508 696 607 строк, в том числе 164 876 196 строк с `-9999` (32,4%).
Крайние времена исходного снимка — 11.09.2013 10:15:00 и 26.11.2026
18:30:03. Границы выгруженных рядов приведены в `channels.csv`.
В `LEO_WEST.DATAVALUES` для переменных 6/7/17 подтверждено отсутствие
строк за апрель–июнь 2016 года.

Все 189 частей прошли сверку количества и временных границ с Oracle.
После сборки проверены файлы, их схемы, SHA-256 и баланс строк до и после
удаления кода. Первоначальный подсчёт составлял 508 691 539 строк; отличие
итога на +5 068 отражает разные моменты чтения изменяющегося источника
(VWC +1 974, MPS-2 +3 024, Mass +70). Это не единый глобальный SCN-снимок.
Контроль сохранности подтвердил неизменность 410 отслеживаемых файлов
CO₂/Research_log и ещё 18 локальных файлов принятого пилота и его сырья.

Основные VWC и MPS-2 занимают одну сетку из 154 точек X/Y:
11 положений X от −5 до +5 м через 1 м и 14 рядов Y от 2 до 28 м через 2 м.
Между крайними точками сетки 10 × 26 м; это геометрия координат реестра.
В 120 точках
по три уровня, в 34 — по четыре. Таким образом, каждый тип даёт
$120\times3+34\times4=496$ каналов; ещё три VWC относятся к временным CatWalk.
Число каналов взято из реестра. Критерии пригодности к значениям не применялись.
Для массы зарегистрированы 5 позиций вдоль склона (Y = 2, 8, 14, 20, 25 м)
и 2 поперёк (X = −3, +3 м). В каждой из 10 точек есть два канала
с отметками BOXZ 0 и 0,05 м.

Формат повторяет существующие CO₂-файлы: Parquet, индекс `localdatetime`
(`datetime64[us]`, без часового пояса), один столбец `datavalue` (`float64`).
Исходные времена, отрицательные потенциалы, нули, выбросы и повторы времени
среди оставшихся строк сохранены. Зарегистрированный канал без записей в
выбранном периоде получает пустой файл с отметкой `no_rows`. Если записи были,
но фильтры удалили их все, отметка будет `filtered_out`.

Имена вида `LEO-W_<Y>_<X>_<dlevel>_vwc.parquet`,
`LEO-W_<Y>_<X>_<dlevel>_matric_potential.parquet` и
`LEO-W_<Y>_<X>_<dlevel>_mass.parquet` повторяют порядок координат CO₂.
`dlevel` — код уровня, а не глубина в метрах. Точные X/Y и глубина `BOXZ`
находятся в `channels.csv` каждой папки вместе с SensorID, VariableID,
таблицей, исходным кодом датчика, единицей, покрытием, числом удалённых
служебных строк, повторами среди оставшихся строк и SHA-256 файла.
Идентичность канала задаётся тройкой
**таблица + SensorID + VariableID**.

Особенности метаданных сохранены явно:

- MPS-2 **909**: `SENSORCODE=LEO-W_12_-4_3_MPS-2`, но связанная
  `LOCATIONS` задаёт `LEO-W_12_-4_4`, глубину 0.50 м и уровень 4.
  Имя файла использует координаты/уровень `LOCATIONS`; исходный код и
  `coordinate_code_conflict=True` сохранены в каталоге.
- MPS-2 **959**: в `LOCATIONS` одновременно указаны `dlevel=5` и
  `BOXZ=0.50` м. Эти поля сохранены без исправления; общую таблицу
  пересчёта номеров уровней в глубины применять нельзя.
- VWC **3004/3005/3006** из `DATAVALUESREU24` имеют общую запись
  `CatWalk`, X=0, Y=0, Z=0. Координаты включены в имена вместе с
  отличительными именами CatWalk; они не устанавливают реальные положения
  и глубины отдельных временных проб.

Основной источник — `LEO_WEST.DATAVALUES`; дополнительные временные VWC —
`LEO_WEST.DATAVALUESREU24`. Состав сверяется по `SENSOROUTPUT`, `SENSORS`,
`LOCATIONS`, `VARIABLES`, `UNITS`, `SENSORTYPES` и фактическим парам
SensorID/VariableID в обеих таблицах. Сырой электрический `LCout`
(VariableID 15, mV/V) не заменяет массу 17 и в эти папки не включён.
Суммирования опор, новой калибровки или пересчёта потенциала в VWC нет.

В каждой папке `provenance.json` фиксирует источник, время чтения, исходную
версию репозитория, количества и проверенные месячные части;
`duplicate_times.csv` перечисляет повторы с исходными VALUEID. Пустой
реестр означает отсутствие найденных повторов. Ошибки и их повторные
попытки фиксируются отдельно и не считаются завершёнными частями.

Каждая пара «таблица–месяц» читается в отдельной транзакции Oracle
`READ ONLY`: число строк и крайние времена сверяются с запросом в той же
транзакции. Весь многолетний архив не объявляется единым SCN-снимком.
Разница с первоначальным подсчётом отражается в `provenance.json`.
После записи каждый файл перечитывается и сравнивается со строками,
прошедшими выбранные фильтры. Числа преобразуются в `float64` для
совместимости с CO₂-образцом; точные десятичные представления Oracle и
VALUEID дополнительно сохранены в локальных исходных частях.

Готовые файлы можно читать без Oracle. Программа
[export_west_hydrology.py](../export_west_hydrology.py) собирает новые выборки
из сохранённых на рабочей машине месячных частей через `--local-only`. Например,
VWC в точке X=2, Y=4 за май 2025 года можно сохранить в отдельную папку
внутри исключённого из Git каталога. Команда из корня репозитория:

```bash
./transcription_env/bin/python -B Project_description/sensorDB/export_west_hydrology.py \
  --version 20261126_full_history --local-only --quantity vwc --x 2 --y 4 \
  --start 2025-05-01 --end 2025-06-01 \
  --output-dir Project_description/sensorDB/Row_data/LEO_West_hydrology_raw/example_vwc_2025_05
```

Начало периода включается, конец исключается. `--quantity` и `--sensor-id`
можно повторять. `--source-table`, `--x`, `--y` и `--depth-m` уточняют
выбор каналов. При выборе одной величины `--min-value` и `--max-value`
задают включительные границы в её единицах. `--keep-service-values` сохраняет
код Oracle, а `--drop-nonfinite` удаляет пустые и нечисловые значения.
Паспорта фиксируют параметры и число строк, исключённых каждым фильтром.

Для пересборки полного рабочего набора в трёх основных папках используется
`--version 20261126_full_history --local-only --replace`. Для нового снимка
из Oracle выбирают новое имя версии. Команда с `--download-only` сохранит
сырьё без пересборки файлов. Для продолжения незавершённого скачивания
повторяют то же имя:

```bash
LD_LIBRARY_PATH=/opt/oracle/instantclient_19_26:/home/dimitri/.local/lib ./transcription_env/bin/python -u -B Project_description/sensorDB/export_west_hydrology.py --version SNAPSHOT_NAME --download-only
```

VWC — физический выход 6 в процентах. Сопутствующие выходы 4
(`bulkPerm`, диэлектрическая проницаемость) и 5 (`soilEC`, электропроводность
в dS/m) не используются вместо объёмного содержания воды.

Локальные возобновляемые части и метаданные находятся в
`LEO_West_hydrology_raw/<версия>/`. Исходные файлы содержат также
строки с кодом отсутствия данных. Для замены готовых папок нужен
явный `--replace`. Доступ к Oracle использует существующие локальные настройки.

Эта локальная выгрузка исходных рядов не меняет CO₂, исследовательский план,
принятый гидрологический вход 0.1-B или его критерии применимости.

## История температурных измерений LEO West

Версия `20261126_full_history`. Верхняя граница источника по часам Oracle — 2026-11-26 22:48:45. В 262 блоках получено 544 043 280 строк. После технической очистки сохранено 376 944 728, исключено 167 098 552. Каждый ряд идентифицируется тройкой **таблица Oracle, SensorID, VariableID**.

| Каталог и измерение | VariableID | Датчик | Ед. | Каналов | Сырых | Исключено | Сохранено | Даты исходных записей | Parquet, МБ |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| [Базальт](LEO_West_Soil_Temperature) | 3 | 5TM, MPS-2, TCAV | degC | 1004 | 500 924 026 | 164 739 131 | 336 184 895 | 2013-09-11 — 2026-11-26 | 480.699 |
| [Воздух над склоном](LEO_West_Air_Temperature) | 41 | HMP60 | degC | 25 | 19 049 447 | 331 182 | 18 718 265 | 2015-06-17 — 2026-11-26 | 109.181 |
| [Временные 5TM на CatWalk](LEO_West_Other_Temperature) | 3 | 5TM | degC | 3 | 123 169 | 11 924 | 111 245 | 2024-06-26 — 2026-11-26 | 0.169 |
| [Регистратор](LEO_West_Other_Temperature) | 20 | DeviceT | degC | 3 | 1 419 530 | 1 174 602 | 244 928 | 2013-09-11 — 2017-04-24 | 0.196 |
| [LoadCellSurface](LEO_West_Other_Temperature) | 30 | TC-E | degC | 10 | 8 571 080 | 1 270 | 8 569 810 | 2020-02-11 — 2026-11-26 | 64.092 |
| [Радиометр](LEO_West_Other_Temperature) | 49 | CNR4 | degK | 3 | 1 595 889 | 547 | 1 595 342 | 2015-06-17 — 2026-11-26 | 9.758 |
| [Ячейка LI-COR](LEO_West_Other_Temperature) | 59 | LI-7000 | degC | 59 | 376 327 | 1 372 | 374 955 | 2023-07-03 — 2026-11-26 | 1.121 |
| [Вода](LEO_West_Other_Temperature) | 61 | CS451 | degC | 15 | 7 087 440 | 4 905 | 7 082 535 | 2015-01-13 — 2026-11-26 | 43.363 |
| [Воздуховоды, звуковая температура](LEO_West_Other_Temperature) | 69 | CSAT3 Flux | degC | 3 | 2 600 688 | 2 | 2 600 686 | 2024-06-06 — 2026-11-26 | 20.087 |
| [Выходы AHU](LEO_West_Other_Temperature) | 92 | S-THP-01A | degC | 12 | н/д | н/д | 0 | источник недоступен | 0.022 |
| [VIZ, soilTemp](LEO_West_Visualization_Temperature) | 3 | 5TM, MPS-2, TCAV | degC | 723 | 2 144 418 | 833 515 | 1 310 903 | 2026-10-27 — 2026-11-26 | 3.648 |
| [VIZ, SurfaceTemp](LEO_West_Visualization_Temperature) | 30 | TC-E | degC | 10 | 29 660 | 20 | 29 640 | 2026-10-27 — 2026-11-26 | 0.256 |
| [VIZ, AirTemp](LEO_West_Visualization_Temperature) | 41 | HMP60 | degC | 24 | 71 184 | 48 | 71 136 | 2026-10-27 — 2026-11-26 | 0.588 |
| [VIZ, PyrgeoTemp](LEO_West_Visualization_Temperature) | 49 | CNR4 | degK | 2 | 5 932 | 4 | 5 928 | 2026-10-27 — 2026-11-26 | 0.062 |
| [VIZ, waterTemp](LEO_West_Visualization_Temperature) | 61 | CS451 | degC | 15 | 44 490 | 30 | 44 460 | 2026-10-27 — 2026-11-26 | 0.396 |

МБ = 1 000 000 байт. Для недоступного источника сохранены типизированные пустые файлы. Суммарный объём рабочих Parquet — 733 637 195 байт. Файлы хранятся обычными объектами Git. Для температурных каталогов .gitattributes сохраняет точные байты при checkout, включая окончания строк в CSV. Строки VIZ относятся к отдельной таблице визуализации и не добавляются к рядам первичных датчиков при анализе.

Четыре каталога вместе с паспортами и таблицами повторов занимают 829 116 354 байта. Крупнейший файл — duplicate_times.csv базальта, 90 045 726 байт. Размеры допускают обычную Git-публикацию одним push. [Ограничения GitHub](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits).

### Источники и геометрия

В исходной инвентаризации сверены десять SHA-256 гидрологического снимка и шесть справочников Oracle. При сравнении по первичным ключам между снимками совпали все поля этих справочников. В проверку фактических пар вошли все доступные таблицы LEO_WEST со столбцами измерений. Температурные записи найдены в DATAVALUES, DATAVALUESREU24, DATAVALUESLICOR, DATAVALUESCSAT3 и DATAVALUESVIZ. В DATAVALUES обнаружены 27 фактических каналов IRGA_Temp (59), для которых реестр указывает другую таблицу. В доступных зарегистрированных источниках нет строк для CSAT3 1308/69 и LI-COR 1304/59, 1305/59, 1309/59. В OUTFLOWS_NEW и OUTFLOWS_OLD температурных записей нет. VariableID 29, SonicTemp, присутствует в VARIABLES без температурных рядов. SENSORLOG содержит четыре записи о калибровках CO₂. Полная история перемещений температурных датчиков из него не восстанавливается.

Сетка базальта занимает 154 точки, X от −5 до +5 м, Y от 2 до 28 м. Глубины по BOXZ с системой отсчёта от поверхности составляют 0,05, 0,20, 0,35, 0,50 и 0,85 м. Здесь 496 каналов 5TM, 496 MPS-2 и 12 усредняющих термопар TCAV на глубине 5 см. У MPS-2 909 расходятся уровень в коде датчика и запись LOCATIONS. У 959 расходятся глубина инвентаря и BOXZ. У радиометра 1255 для X в инвентаре указано +2 м, в LOCATIONS −2 м. В channels.csv сохранены обе записи. Три 5TM в DATAVALUESREU24 связаны с общей реестровой позицией CatWalk. Для 3004 инвентарь указывает синюю бочку. Глубины этих датчиков в склоне не установлены.

Координаты (X,Y) пяти воздушных мачт по реестру составляют (0,4), (0,17), (0,24), (−2,10) и (2,10) м. Высоты 0,25, 1, 3, 6 и 10 м взяты из LEO-West-Inventory для совпадающих SensorID, VariableID и SensorCode. Для 1219 в (0,4) точная высота и установка не подтверждены. Все 325 373 исходные записи этого канала за историю содержат служебный код, в пилотном периоде исходных записей нет. В опубликованном описании установки у этой мачты четыре нижних уровня, а проектный инвентарь содержит 24 из 25 зарегистрированных HMP60. Высоты соответствуют номинальному положению мачт. В описанном авторами периоде мачты поднимали во время дождя. [Arevalo et al., 2020](https://doi.org/10.1038/s41597-020-00645-5).

Прямое измерение температуры поверхности базальта не подтверждено. Все десять каналов SurfaceTemp (TC-E, 1265–1274) относятся к LoadCellSurface. У всех BOXZ = 0, но SensorCode и лист Control связывают их с весовыми опорами. Материалы не устанавливают контакт термопары с базальтом. Они помещены в Other. Там же находятся температуры регистраторов, радиометров (degK), газоанализаторов (degC), воды и каналов вентиляции. SonicTempAvg — уже усреднённая источником звуковая температура CSAT3 Flux. На этапе экспорта значения не усреднялись. В LARGESCALEFLUXES есть 38 640 записей TPUSH/TPULL/TVENT в широкой таблице расчёта потоков без SensorID, VariableID и документированной единицы. Они учтены при обследовании схемы. Для 12 каналов 92 зарегистрирован DataValuesAHU. Прямой SELECT вернул ORA-00942, а доступного синонима и однозначного альтернативного источника этих пар не найдено. Статус в каталоге — source_unavailable.

### Полнота, очистка и формат

Для каждого месяца и таблицы отдельная транзакция READ ONLY сопоставляет количество и крайние времена Oracle с исходными строками по каждой паре SensorID/VariableID. Итоговый счёт совпал с начальной инвентаризацией для каждого фактического канала. Журнал повторных попыток пуст. Граница по LOCALDATETIME фиксирована. Каждая транзакция имеет собственный момент чтения и SCN. Все 167 098 552 исключённые строки содержали служебный код. Значений NULL, NaN и ±∞ в снимке не обнаружено. NODATAVALUE определён отдельно через VariableID для каждого канала. В этой версии у всех он равен −9999. В доступных описаниях METHODS и QUALITYCONTROL дополнительных кодов для этих измерений не установлено. Нули, отрицательные температуры, резкие изменения и повторы сохранены. Сырой архив сохраняет VALUEID и точное десятичное представление Oracle NUMBER, полученное через TM9 с фиксированной десятичной точкой.

Каждому каналу соответствует один Parquet-файл. Схема всех 1 911 файлов сверена с существующим Parquet из LEO_West. В каждом ряду один столбец `datavalue` типа float64 и индекс `localdatetime` типа datetime64[us] без часового пояса. Порядок записей — исходное время, затем VALUEID. Программа сохраняет шкалу времени Oracle без сдвига. Имена используют последовательность Y, X, DLEVEL, физическая величина. Коллизии разрешаются указанием таблицы, SensorID и VariableID. Исходная единица каждого канала указана в channels.csv. Значения degK и degC сохранены в собственных шкалах.

В каждой рабочей папке channels.csv содержит геометрию, поля Oracle и инвентаря, границы сырых и очищенных рядов, моду положительных интервалов, изменения помесячной моды, каждую причину исключения, диапазон сохранённых значений, размер и SHA-256. Мода описывает наблюдаемый ряд и может меняться из-за пропусков. duplicate_times.csv перечисляет исходные VALUEID всех повторов в окне выборки. provenance.json содержит версию, Git HEAD на начало снимка, хеши кода и метаданных, фильтры и проверенные блоки. Исходное состояние канала различает отсутствие записей, недоступность источника, только служебные значения и полное удаление пользовательским фильтром.

### Покрытие W_x1_y4 с мая 2025 по сентябрь 2026

Полуоткрытый период [2025-05-01, 2026-10-01) охватывает 518 суток. В таблице показаны все каналы непосредственно в (1,4), а также почвенные и воздушные каналы на расстоянии до 1 м. Глубина или высота приведена относительно поверхности. Столбец «Пустых интервалов» показывает, сколько из 49 728 фиксированных четвертей часа не содержат ни одного действительного наблюдения. Он учитывает разную частоту источников без интерполяции. Число утраченных исходных измерений при смене расписания неизвестно.

| Канал | Датчик | (X,Y), м | Расстояние, м | Глубина / высота, м | Первая — последняя действительная запись | Действительных | Служебных | Без исходных строк¹ | Пустых интервалов¹ |
|---:|---|---|---:|---|---|---:|---:|---:|---:|
| 18 | 5TM | (1,4) | 0 | гл. 0.05 | — | 0 | 49 724 | 4 | 49 728 |
| 514 | MPS-2 | (1,4) | 0 | гл. 0.05 | — | 0 | 49 724 | 4 | 49 728 |
| 172 | 5TM | (1,4) | 0 | гл. 0.2 | — | 0 | 0 | 49 728 | 49 728 |
| 668 | MPS-2 | (1,4) | 0 | гл. 0.2 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 705 | 19 | 4 | 23 |
| 393 | 5TM | (1,4) | 0 | гл. 0.5 | — | 0 | 0 | 49 728 | 49 728 |
| 889 | MPS-2 | (1,4) | 0 | гл. 0.5 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 705 | 19 | 4 | 23 |
| 1215 | HMP60 | (0,4) | 1 | выс. 0.25 | 2025-05-01 — 2026-09-30 23:45:00 | 49 708 | 16 | 4 | 20 |
| 1216 | HMP60 | (0,4) | 1 | выс. 1 | 2025-05-01 — 2026-09-30 23:45:00 | 49 708 | 16 | 4 | 20 |
| 1217 | HMP60 | (0,4) | 1 | выс. 3 | 2025-05-01 — 2026-09-30 23:45:00 | 49 708 | 16 | 4 | 20 |
| 1218 | HMP60 | (0,4) | 1 | выс. 6 | 2025-05-01 — 2026-09-30 23:45:00 | 49 708 | 16 | 4 | 20 |
| 1219 | HMP60 | (0,4) | 1 | не установлена | — | 0 | 0 | 49 728 | 49 728 |
| 17 | 5TM | (0,4) | 1 | гл. 0.05 | 2025-05-01 00:00:01 — 2025-08-31 05:15:00 | 11 734 | 37 990 | 4 | 37 994 |
| 513 | MPS-2 | (0,4) | 1 | гл. 0.05 | — | 0 | 0 | 49 728 | 49 728 |
| 171 | 5TM | (0,4) | 1 | гл. 0.2 | — | 0 | 0 | 49 728 | 49 728 |
| 667 | MPS-2 | (0,4) | 1 | гл. 0.2 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 22 123 | 27 601 | 4 | 27 605 |
| 392 | 5TM | (0,4) | 1 | гл. 0.5 | — | 0 | 49 724 | 4 | 49 728 |
| 888 | MPS-2 | (0,4) | 1 | гл. 0.5 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 705 | 19 | 4 | 23 |
| 468 | 5TM | (0,4) | 1 | гл. 0.85 | — | 0 | 0 | 49 728 | 49 728 |
| 964 | MPS-2 | (0,4) | 1 | гл. 0.85 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 705 | 19 | 4 | 23 |
| 19 | 5TM | (2,4) | 1 | гл. 0.05 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 670 | 54 | 4 | 58 |
| 515 | MPS-2 | (2,4) | 1 | гл. 0.05 | — | 0 | 49 724 | 4 | 49 728 |
| 173 | 5TM | (2,4) | 1 | гл. 0.2 | — | 0 | 0 | 49 728 | 49 728 |
| 669 | MPS-2 | (2,4) | 1 | гл. 0.2 | — | 0 | 49 724 | 4 | 49 728 |
| 317 | 5TM | (2,4) | 1 | гл. 0.35 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 687 | 37 | 4 | 41 |
| 813 | MPS-2 | (2,4) | 1 | гл. 0.35 | 2025-05-01 00:00:01 — 2026-09-30 23:45:01 | 49 705 | 19 | 4 | 23 |

¹ Интервалы по 15 минут. Соседние данные имеют собственные координаты. Поканальные таблицы доступны для [базальта](LEO_West_Soil_Temperature/pilot_coverage.csv) и [воздуха](LEO_West_Air_Temperature/pilot_coverage.csv). В (1,4) реестр содержит три глубины и по два температурных канала на каждой. Прямого температурного датчика на поверхности и воздушной мачты в этой точке не подтверждено.

### Обновление и пользовательские выборки

Из корня репозитория полное обновление и локальная проверка состояния выполняются так.

```bash
make temperature-refresh
make temperature-status
```

Первый запуск создаёт версию в LEO_West_temperature_raw. Повторный запуск продолжает прерванную версию по проверенным месячным манифестам. После успешной сборки основного набора следующее обновление создаёт новую версию. Сырьё остаётся локально. Чтение выполняется максимум четырьмя потоками с тремя попытками на блок. Публикация рабочих папок начинается после проверки файлов новой версии. Журнал замены позволяет восстановить предыдущее состояние после прерывания.

Следующие команды читают завершённый локальный снимок без Oracle. В порядке примеров они выбирают базальт за сутки, воздух за сутки с 06:00, две глубины, пилотную вертикаль, температуры воздуха от 10 до 30 °C и полную историю двух выбранных MPS-2.

```bash
T=Project_description/sensorDB/export_west_temperature.py
R=Project_description/sensorDB/Row_data/LEO_West_temperature_raw/selections
temperature() { python3 -B "$T" --local-only --version 20261126_full_history "$@"; }

temperature --quantity soil --start 2025-05-01 --end 2025-05-02 --output-dir "$R/soil_day" --replace
temperature --quantity air --start 2025-05-01T06:00 --end 2025-05-02T06:00 --output-dir "$R/air_day" --replace
temperature --quantity soil --depth-m 0.05 --depth-m 0.20 --start 2025-05-01 --end 2025-05-02 --output-dir "$R/two_depths" --replace
temperature --quantity soil --x 1 --y 4 --start 2025-05-01 --end 2026-10-01 --output-dir "$R/W_x1_y4" --replace
temperature --quantity air --min-value 10 --max-value 30 --start 2025-05-01 --end 2025-05-02 --output-dir "$R/air_10_30" --replace
temperature --source-table DATAVALUES --sensor-type MPS-2 --variable-id 3 --sensor-id 668 --sensor-id 889 --output-dir "$R/two_MPS2" --replace
```

Начало интервала включено, конец исключён. Обе границы значений включены. --x, --y, --depth-m и --height-m допускают список повторяемых значений и границы с суффиксами -min и -max. Выбор по типу, SensorID, VariableID, категории и таблице можно комбинировать. --keep-service-values сохраняет служебные значения только в отдельной диагностической выборке. --download-only сохраняет исходный снимок, --local-only использует уже завершённый. --version выбирает снимок, --workers принимает значения от 1 до 4. --replace заменяет ранее проверенную управляемую выборку по указанному пути. Частичная выборка требует отдельного --output-dir.

Независимая проверка читает все исходные десятичные значения, сопоставляет каждый очищенный ряд со временем и значением каждой сохранённой строки и сверяет балансы, SHA-256, имена, схемы, границы и число каналов.

```bash
python3 -B Project_description/sensorDB/verify_west_temperature.py --version 20261126_full_history
```

Температурные тесты проверяют очистку, границы времени и значений, сохранение исходной точности и повторов, пространственный отбор, изоляцию выборок, прерывание и повторный запуск. Для полного прогона 216 тестов использованы системные pandas 2.3.3 и PyArrow 24.0.0, а также установленный в transcription_env Plotly 7.1.0.

Сравнение с исходным Git-коммитом подтвердило сохранность 1 434 отслеживаемых файлов CO₂, гидрологии и Research_log. Для 18 локальных файлов исходного пилота 0.1-B перепроверены SHA-256. Шесть примеров локальных выборок сопоставлены с полными рабочими рядами, а результаты для 25 каналов в радиусе 1 м — с отдельным запросом Oracle.

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
