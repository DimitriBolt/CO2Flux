"""Обновляет четыре ряда W_x1_y4 из последней завершённой выгрузки LEO West."""

import fcntl
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DATA = Path(__file__).resolve().parent / "Row_data"
ARCHIVE = DATA / "LEO_West_full"
DESTINATION = DATA / "LEO_West"

# Канал определяется таблицей Oracle и парой идентификаторов.
CHANNELS = {
    "LEO_WEST.DATAVALUESLICOR": {
        (1275, 56): "LEO-W_4_0_1_air.parquet",
    },
    "LEO_WEST.DATAVALUES": {
        (996, 9): "LEO-W_4_1_1_basalt.parquet",
        (1012, 9): "LEO-W_4_1_2_basalt.parquet",
        (1035, 9): "LEO-W_4_1_4_basalt.parquet",
    },
}


def latest_complete():
    # Имена версий начинаются с даты UTC, поэтому сортировка даёт порядок выгрузок.
    for path in sorted(ARCHIVE.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return path.parent, manifest
    raise ValueError(f"В {ARCHIVE} нет завершённых выгрузок")


def export(version, manifest, staging):
    if not manifest.get("download_complete_at") or any(
        block["status"] != "complete" for block in manifest["blocks"]
    ):
        raise ValueError(f"Выгрузка {version.name} содержит незавершённые блоки")

    inventory = {
        (channel["table"], channel["sensorid"], channel["variableid"])
        for channel in manifest["channels"]
    }
    for table, channels in CHANNELS.items():
        for sensor_id, variable_id in channels:
            if (table, sensor_id, variable_id) not in inventory:
                raise ValueError(f"В манифесте отсутствует канал {table}, {sensor_id}, {variable_id}")

    chunks = {name: [] for channels in CHANNELS.values() for name in channels.values()}

    # Читаем каждую физическую часть один раз. У разбитого блока свои части
    # могут быть и у родителя, и у дочерних блоков.
    for block in manifest["blocks"]:
        channels = CHANNELS.get(block["table"], {})
        sensor_ids = pa.array([sensor_id for sensor_id, _ in channels], type=pa.int64())
        for part in block["parts"]:
            counts = part["sensor_counts"]
            if not any(counts.get(str(sensor_id), 0) for sensor_id, _ in channels):
                continue

            path = version / part["file"]
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != part["sha256"]:
                    raise ValueError(f"Контрольная сумма не совпала: {path}")
                source.seek(0)
                data = pq.read_table(
                    source, columns=["localdatetime", "datavalue", "sensorid", "variableid"]
                )
            if data.num_rows != part["rows"]:
                raise ValueError(f"Число строк не совпало: {path}")

            # Отбираем датчики до преобразования строк в pandas.
            ids = pc.cast(data["sensorid"], pa.int64())
            frame = data.filter(pc.is_in(ids, value_set=sensor_ids)).to_pandas()
            for (sensor_id, variable_id), name in channels.items():
                rows = frame.loc[
                    frame["sensorid"].eq(sensor_id) & frame["variableid"].eq(variable_id),
                    ["localdatetime", "datavalue"],
                ].copy()
                if len(rows) != counts.get(str(sensor_id), 0):
                    raise ValueError(f"Число строк датчика {sensor_id} не совпало: {path}")
                if not rows.empty:
                    chunks[name].append(rows)

    for table, channels in CHANNELS.items():
        roots = [
            block for block in manifest["blocks"]
            if block["table"] == table and not block.get("parent")
        ]
        for (sensor_id, _), name in channels.items():
            expected = sum(block["verified_counts"].get(str(sensor_id), 0) for block in roots)
            if not chunks[name] or sum(map(len, chunks[name])) != expected:
                raise ValueError(f"Итоговое число строк датчика {sensor_id} не совпало с манифестом")

            # Оставляем прежнюю структуру файла. Повторные времена и пустые
            # значения сохраняются. Время остаётся без преобразования часового пояса.
            series = pd.concat(chunks[name], ignore_index=True)
            series["datavalue"] = series["datavalue"].astype("float64")
            series = series.set_index("localdatetime").sort_index(kind="stable")
            output = staging / name
            series.to_parquet(output, engine="pyarrow", compression="zstd")
            pd.testing.assert_frame_equal(pd.read_parquet(output, engine="pyarrow"), series)
            print(f"{name} · {len(series)} строк · {series.index[0]} — {series.index[-1]}", flush=True)

    return chunks.keys()


def main():
    # Блокировка исключает одновременное обновление из двух запусков программы.
    with (ARCHIVE / "export_west_x1_y4.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        version, manifest = latest_complete()
        print(f"Исходная версия {version.name}", flush=True)
        DESTINATION.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".west_x1_y4-", dir=DESTINATION) as temporary:
            staging = Path(temporary)
            names = export(version, manifest, staging)
            # Заменяем файлы только после проверки всего нового набора.
            for name in names:
                (staging / name).replace(DESTINATION / name)
        print("Все четыре файла обновлены.", flush=True)


if __name__ == "__main__":
    main()
