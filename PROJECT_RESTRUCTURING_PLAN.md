# 🏗️ План Реструктуризации Проектов
**Дата:** 15 May 2026  
**Статус:** Plan/Ready for Execution

---

## 📌 Проблема

Два независимых проекта неправильно смешаны в одном репозитории `/home/dimitri/PycharmProjects/CO2Flux`:

1. **CO2 Flux Analysis** (основной проект CO2Flux)
   - Анализ концентрации CO2 в базальте
   - Измерение influx/outflux через базальтовые склоны (LEO Center, LEO East, LEO West)
   - Источник: CO2.docx

2. **RainForest Climate Control Digital Twin** (независимый проект)
   - Автоматизация управления климатом биома RainForest
   - Цифровая модель для тестирования алгоритмов управления
   - Студенческая hackathon платформа
   - Источник: письма, AGENTS.md, docs/CLIMATE_CONTROL_DIGITAL_TWIN_PLAN.md

---

## ✅ Решение: Разделить на два репозитория

### 📁 СТРУКТУРА 1: `/home/dimitri/PycharmProjects/CO2Flux/` (CO2-только)

**Назначение:** Анализ CO2 в базальте на LEO склонах

**Оставить:**
```
CO2Flux/
├── AGENTS.md (обновить - убрать Climate Control refs)
├── README.md (обновить - фокус только на CO2)
├── LICENSE
├── requirements.txt
├── .env (если существует)
│
├── Sensors_Description/
│   ├── LEO-Center-Inventory.xlsx
│   ├── LEO-East-Inventory.xlsx
│   ├── LEO-West-Inventory.xlsx
│   ├── LEOSensorDBdescription.pdf
│   ├── co2_vertical_profile_viewer.py
│   ├── co2_viewer_add_surface.py
│   ├── co2_vertical_profile_viewer_config.toml
│   ├── co2_vertical_profile_viewer_config.local.toml.example
│   ├── co2_vertical_profile_viewer_requirements.md
│   ├── CO2_air.sql
│   ├── CO2basalt.sql
│   ├── temp_basalt.sql
│   ├── humidity_basalt.sql
│   ├── workflow_memory.md
│   ├── variables_schema.xlsx (только CO2 sheet + Metadata)
│   └── co2_profile_LEO_*.{gif,jpg} (выходные файлы)
│
└── Project_description/
    ├── CO2.docx (основной документ)
    ├── Correlations/
    └── sensorDB/

```

**Удалить (переместить в DigitalTwin):**
- `docs/CLIMATE_CONTROL_DIGITAL_TWIN_PLAN.md`
- `docs/CLIMATE_CONTROL_QUICK_START.md`
- `scripts/analyze_climate_data.py`
- `scripts/filter_input_climate_controls.py`
- `scripts/prepare_training_data.py`
- `scripts/update_input_sheet.py`
- `scripts/update_rainforest_output_sheet.py`
- `filtered_input_climate_controls.csv`
- `RAINFOREST_OUTPUT_RU.md`
- `RAINFOREST_OUTPUT_SETUP_REPORT.md`
- `Project_description/email_to_john_adams_climate_control_inputs.txt`
- `Project_description/Re_ Projects oriented toward Biosphere 2 objectives*.eml`
- `Project_description/Response_to_John_Adams_Climate_Control_RainForest.txt`
- `docs/DATA_PIPELINE_GUIDE.md` (Climate Control specific)
- `CLIMATE_CONTROL_DIGITAL_TWIN_PLAN.md`
- Все файлы с "RAINFOREST", "RU", "PHASE1" корреляции

**Документация (обновить):**
- `AGENTS.md` → только CO2 инструкции для AI
- `README.md` → только CO2 проект
- `docs/` → только CO2-относящаяся документация

---

### 📁 СТРУКТУРА 2: `/home/dimitri/PycharmProjects/DigitalTwin/` (новый проект)

**Назначение:** Цифровая модель климата для RainForest + student hackathon

**Создать структуру:**
```
DigitalTwin/
├── AGENTS.md (новый - для DigitalTwin AI инструкции)
├── README.md (новый - DigitalTwin фокус)
├── LICENSE
├── requirements.txt (новый - LSTM, FastAPI, PyTorch, etc.)
├── .env (скопировать Oracle credentials)
│
├── docs/
│   ├── CLIMATE_CONTROL_DIGITAL_TWIN_PLAN.md (перемещено)
│   ├── CLIMATE_CONTROL_QUICK_START.md (перемещено)
│   ├── DATA_PIPELINE_GUIDE.md (Climate Control version)
│   ├── PHASE1_ARCHITECTURE.md (новый)
│   ├── API_SPECIFICATION.md (новый)
│   └── STUDENT_HACKATHON_GUIDE.md (новый)
│
├── scripts/
│   ├── analyze_climate_data.py (перемещено)
│   ├── filter_input_climate_controls.py (перемещено)
│   ├── prepare_training_data.py (перемещено)
│   ├── update_input_sheet.py (перемещено)
│   ├── update_rainforest_output_sheet.py (перемещено)
│   ├── train_lstm_model.py (новый)
│   ├── deploy_api_server.py (новый)
│   └── evaluate_student_submission.py (новый)
│
├── data/
│   ├── rawdata/ (Oracle extracts)
│   ├── processed/ (normalized arrays, train/val/test splits)
│   └── models/ (trained LSTM checkpoints, ONNX exports)
│
├── api/
│   ├── main.py (FastAPI server, /predict endpoint)
│   ├── models.py (LSTM PyTorch architecture)
│   ├── scoring.py (3-metric evaluation framework)
│   └── constraints.py (hard constraints validation)
│
├── notebooks/
│   ├── 01_EDA_Climate_Data.ipynb
│   ├── 02_Data_Pipeline.ipynb
│   ├── 03_LSTM_Training.ipynb
│   └── 04_API_Testing.ipynb
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_lstm_model.py
│   └── test_api_endpoints.py
│
├── Sensors_Description/ (从CO2Flux复制)
│   ├── Bio2-Controls-Inventory-29Jan2026.xlsx
│   ├── Bio2-Rainforest-Inventory.xlsx
│   ├── variables_schema.xlsx (仅Input/Output sheets)
│   └── RainForest-CO2-Atmospheric.xlsx (新建)
│
├── Project_description/
│   ├── email_to_john_adams_climate_control_inputs.txt (перемещено)
│   ├── Re_ Projects oriented toward Biosphere 2 objectives.eml (перемещено)
│   ├── Re_ Projects oriented toward Biosphere 2 objectives-2026-05-07.eml (перемещено)
│   ├── Response_to_John_Adams_Climate_Control_RainForest.txt (перемещено)
│   ├── Stakeholder_Engagement_Letter.md (новый)
│   └── Technical_Specification.md (новый)
│
├── docker/
│   ├── Dockerfile (LSTM inference container)
│   ├── docker-compose.yml (multi-service: API + DB + Redis)
│   └── .dockerignore
│
├── reports/
│   ├── PHASE1_COMPLETE.md (перемещено)
│   ├── PHASE2_LSTM_TRAINING_REPORT.md (новый)
│   └── PHASE3_API_DEPLOYMENT_REPORT.md (новый)
│
└── .gitignore (новый)
    ├── *.pyc
    ├── __pycache__/
    ├── venv/
    ├── data/rawdata/*.csv
    ├── data/processed/*.npy
    ├── *.onnx
    └── .env
```

---

## 🔄 Процесс Миграции

### Шаг 1: Создать DigitalTwin структуру
```bash
mkdir -p /home/dimitri/PycharmProjects/DigitalTwin/{docs,scripts,data/{rawdata,processed,models},api,notebooks,tests,Sensors_Description,Project_description,docker,reports}
```

### Шаг 2: Скопировать файлы из CO2Flux в DigitalTwin
**Копировать (не удалять из CO2Flux):**
- `scripts/analyze_climate_data.py`
- `scripts/filter_input_climate_controls.py`
- `scripts/prepare_training_data.py`
- `scripts/update_input_sheet.py`
- `scripts/update_rainforest_output_sheet.py`
- `docs/CLIMATE_CONTROL_DIGITAL_TWIN_PLAN.md`
- `docs/CLIMATE_CONTROL_QUICK_START.md`
- `docs/DATA_PIPELINE_GUIDE.md` (создать версию для DigitalTwin)
- `Sensors_Description/Bio2-Controls-Inventory-29Jan2026.xlsx`
- `Sensors_Description/Bio2-Rainforest-Inventory.xlsx`
- `Sensors_Description/variables_schema.xlsx` (Извлечь только Input/Output sheets)
- `Project_description/email_to_john_adams_climate_control_inputs.txt`
- `Project_description/Re_ Projects oriented...*.eml`
- `Project_description/Response_to_John_Adams_*.txt`

**В CO2Flux остается:**
- Все LEO-related файлы и инвентаризация
- `Sensors_Description/variables_schema.xlsx` (только CO2 sheet)

### Шаг 3: Обновить документацию в каждом проекте

**CO2Flux `/AGENTS.md`** (удалить Climate Control refs):
```markdown
# CO2Flux Project Instructions

## Project Overview
This project focuses on analyzing **CO2 vertical profile data** and measuring 
CO2 influx/outflux through basalt slopes at Biosphere 2 (LEO Center, LEO East, LEO West).

## Key Objectives
- Visualize CO2 concentration profiles in basalt
- Track CO2 influx and outflux dynamics
- Generate animated visualizations for three LEO slopes

[... rest CO2-only content ...]

## Related Independent Projects
**Note:** Climate Control Digital Twin project has been separated to:
- **Location:** `/home/dimitri/PycharmProjects/DigitalTwin/`
- **Purpose:** Automated climate management for RainForest biome
```

**DigitalTwin `/AGENTS.md`** (новый):
```markdown
# DigitalTwin Project Instructions

## Project Overview
This project develops a **digital twin of RainForest climate system** for:
- Automated climate control optimization
- Algorithm testing and validation
- Student hackathon competition platform

## Key Objectives
- Build LSTM surrogate model (Phase 2)
- Develop REST API for student submissions (Phase 3)
- Implement scoring framework (Phase 4)

## Architecture
**Input:** 64 RainForest climate control parameters  
**System:** RainForest HVAC + controls  
**Output:** 36 monitoring sensors (18 temp + 18 humidity)

## Data Pipeline (Phase 1 - Complete)
- Extract: 500K timesteps from Oracle SensorDB
- Process: Min-max normalization, sequence generation
- Split: 70/15/15 (train/val/test) with temporal ordering

[... rest DigitalTwin content ...]
```

### Шаг 4: Создать CO2Flux README (обновленный)
```markdown
# CO2Flux: CO2 Vertical Profile Analysis

Analyze CO2 concentration profiles in basalt slopes at Biosphere 2.

## Quick Start
- Python 3.11+
- `pip install -r requirements.txt`
- `python3 Sensors_Description/co2_vertical_profile_viewer.py`

## Project Structure
- `Sensors_Description/` — LEO slope data, viewers, SQL queries
- `scripts/update_co2_sheet.py` — Refresh CO2 inventory from Oracle
- `Project_description/` — CO2 research documents

## Database
Oracle SensorDB (LEO Center, LEO East, LEO West tables)

## Related Projects
- **DigitalTwin** (`/home/dimitri/PycharmProjects/DigitalTwin/`) — Climate control
```

### Шаг 5: Создать DigitalTwin README (новый)
```markdown
# DigitalTwin: RainForest Climate Control

Digital twin of RainForest biome for climate optimization and student hackathon.

## Quick Start
[Phase 1 documentation...]

## Key Components
- **64 Input Parameters** — Climate control commands
- **36 Output Sensors** — RainForest climate state measurement
- **LSTM Surrogate Model** — Predicts climate response to controls
- **REST API** — Student algorithm submission interface
- **Scoring Framework** — Energy, Comfort, Stability metrics

## Project Timeline
- **Phase 1 (Complete):** Data pipeline
- **Phase 2 (Weeks 3-5):** LSTM training
- **Phase 3 (Weeks 6-8):** REST API deployment
- **Phase 4 (Weeks 9-10):** Evaluation framework
- **Phase 5 (Weeks 11-12):** Documentation & student guide

## Stakeholders
- Wei-Ren Ng (Project Lead TBD)
- Scott Saleska (RainForest Science)
- John Adams (HVAC/BMS Engineering)
- Ildar Gabitov (Academic Advisor)

## Related Projects
- **CO2Flux** (`/home/dimitri/PycharmProjects/CO2Flux/`) — CO2 analysis on LEO slopes
```

---

## 📋 Контрольный список миграции

- [ ] Создать структуру DigitalTwin папок
- [ ] Скопировать файлы в DigitalTwin
- [ ] Обновить CO2Flux AGENTS.md (убрать Climate Control)
- [ ] Создать DigitalTwin AGENTS.md (новый)
- [ ] Обновить CO2Flux README.md
- [ ] Создать DigitalTwin README.md
- [ ] Обновить CO2Flux requirements.txt (убрать LSTM, FastAPI, etc.)
- [ ] Создать DigitalTwin requirements.txt
- [ ] Обновить CO2Flux docs/ (только CO2)
- [ ] Обновить DigitalTwin docs/ (скопировать + новые)
- [ ] Обновить CO2Flux/AGENTS.md с ссылкой на DigitalTwin
- [ ] Инициализировать Git в DigitalTwin (если нужно)
- [ ] Протестировать, что оба проекта работают независимо

---

## 🎯 Результат

**После миграции:**

✅ **CO2Flux:** Чистый, фокусированный репозиторий для CO2 анализа на LEO склонах

✅ **DigitalTwin:** Новый, полностью структурированный репозиторий для климатического контроля RainForest

✅ **Отделенная архитектура:** Каждый проект может развиваться независимо

✅ **Ясная навигация:** Документация четко определяет границы проектов

✅ **Готово для stakeholders:** Легко показать Wei-Ren Ng, Scott Saleska структуру работы

---

## 📞 Следующие шаги

1. **Одобрить план** — Да/Нет?
2. **Начать миграцию** — Выполнить контрольный список
3. **Обновить окружение** — Разные venv для CO2Flux и DigitalTwin
4. **Тестирование** — Убедиться, что оба проекта работают
5. **Stakeholder communication** — Уведомить Wei-Ren Ng, John Adams о новой структуре

