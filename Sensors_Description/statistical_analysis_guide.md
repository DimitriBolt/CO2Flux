# Statistical Analysis Guide for CO2Flux Data

This guide explains how to extract time series from Oracle SensorDB using the viewer
infrastructure and how to run statistical tests and linear regression on that data.

---

## 1. How the scripts expose data

Both viewer scripts follow the same pipeline:

```
Config (TOML)
    └─► load_viewer_config()
            └─► load_profile_sensors()  ← reads variables_schema.xlsx
            └─► connect_to_oracle()
            └─► fetch_measurements()    ← returns list[Measurement]
            └─► build_frames()          ← assembles time-ordered snapshots
```

Each `Measurement` is a plain frozen dataclass:

```python
@dataclass(frozen=True)
class Measurement:
    timestamp: datetime
    value: float      # CO2 concentration in ppm
```

`fetch_measurements()` returns the raw time series for **one sensor at one depth**.
`build_frames()` forward-fills the last known value across depths to align them on a
common time axis.

---

## 2. Extracting data for analysis

The cleanest way to extract data is to re-use the viewer's existing connection and
sensor-loading logic in a separate analysis script.

### 2a. Minimal extraction script

```python
from pathlib import Path
import sys

# ── point at the project's packages ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SENSOR_DB_DIR = PROJECT_ROOT / "Project_description" / "sensorDB"
if str(SENSOR_DB_DIR) not in sys.path:
    sys.path.insert(0, str(SENSOR_DB_DIR))

import co2_vertical_profile_viewer as base
import pandas as pd

# ── load config (uses the same TOML the viewer uses) ─────────────────────────
base.VIEWER_CONFIG = base.load_viewer_config(base.DEFAULT_CONFIG_PATH)
base.DEFAULT_ORACLE_CLIENT_LIB_DIR = base.VIEWER_CONFIG.oracle.default_client_lib_dir
base.LOCAL_LIB_DIR              = base.VIEWER_CONFIG.oracle.local_lib_dir
base.ORACLE_ENV_READY_FLAG      = base.VIEWER_CONFIG.oracle.env_ready_flag
base.SLOPE      = base.VIEWER_CONFIG.profile.slope
base.X_COORD_M  = base.VIEWER_CONFIG.profile.x_coord_m
base.Y_COORD_M  = base.VIEWER_CONFIG.profile.y_coord_m
base.START_DATE = base.VIEWER_CONFIG.profile.start_date
base.END_DATE   = base.VIEWER_CONFIG.profile.end_date

base._ensure_oracle_runtime_env()

start_dt = base.parse_user_datetime(base.START_DATE, is_end=False)
end_dt   = base.parse_user_datetime(base.END_DATE,   is_end=True)

profile_sensors = base.load_profile_sensors()        # list[ProfileSensor]

connection = base.connect_to_oracle()
cursor = connection.cursor()
try:
    series_by_depth = {
        sensor.depth_m: base.fetch_measurements(cursor, sensor, start_dt, end_dt)
        for sensor in profile_sensors
    }
finally:
    cursor.close()
    connection.close()

# ── convert to DataFrames ─────────────────────────────────────────────────────
dfs = {}
for depth_m, measurements in series_by_depth.items():
    if not measurements:
        continue
    dfs[depth_m] = pd.DataFrame(
        {"timestamp": [m.timestamp for m in measurements],
         "co2_ppm":   [m.value     for m in measurements]}
    ).set_index("timestamp")

# dfs[-0.05]  →  DataFrame with DatetimeIndex and column co2_ppm
# dfs[-0.20]  →  same structure, different depth
```

### 2b. Including the surface air sensor (co2_viewer_add_surface)

```python
import co2_viewer_add_surface as surf_base
from air_co2_catalog import AirCO2Catalog

# after the same base-config setup as above …
surf_base.AIR_X_COORD_M = surf_cfg.x_coord_m
surf_base.AIR_Y_COORD_M = surf_cfg.y_coord_m

air_sensor = AirCO2Catalog(workbook_path=base.WORKBOOK_PATH).get_sensor(
    slope=base.SLOPE,
    x_coord_m=surf_base.AIR_X_COORD_M,
    y_coord_m=surf_base.AIR_Y_COORD_M,
    height_m=0.25,
)
air_series = air_sensor.fetch_series(start_datetime=start_dt, end_datetime=end_dt)
# air_series is a pd.Series with DatetimeIndex and float values (ppm).
```

---

## 3. Aligning depths on a common time axis

The measurement timestamps are generally **not synchronized** between depths.
Use `pandas.merge_asof` (forward-fill within a tolerance) or `resample + ffill` to
put all depths on the same grid before running multivariate statistics.

```python
import pandas as pd

# resample each depth to a common 10-minute grid, forward-fill gaps up to 30 min
RESAMPLE_FREQ = "10min"
MAX_GAP = pd.Timedelta("30min")

aligned = pd.DataFrame()
for depth_m, df in dfs.items():
    resampled = (
        df["co2_ppm"]
        .resample(RESAMPLE_FREQ)
        .mean()          # average if multiple readings fall in the same bin
    )
    # drop rows that are > MAX_GAP away from the nearest real observation
    resampled = resampled.reindex(
        pd.date_range(resampled.index.min(), resampled.index.max(), freq=RESAMPLE_FREQ)
    )
    aligned[f"depth_{depth_m:.2f}m"] = resampled

aligned = aligned.dropna(how="all").ffill(limit=int(MAX_GAP / pd.Timedelta(RESAMPLE_FREQ)))
```

---

## 4. Descriptive statistics

```python
print(aligned.describe())

# Variance and standard deviation per depth
print(aligned.std())

# Pearson correlation matrix between depths
print(aligned.corr())
```

---

## 5. Statistical hypothesis tests

Install `scipy` if it is not yet in the environment:

```powershell
pip install scipy
```

### 5a. Normality — Shapiro-Wilk test

```python
from scipy import stats

for col in aligned.columns:
    series = aligned[col].dropna()
    stat, p = stats.shapiro(series[:5000])   # Shapiro-Wilk is limited to ~5000 samples
    print(f"{col}: W={stat:.4f}, p={p:.4g}")
# p < 0.05 → reject normality
```

### 5b. Normality — Kolmogorov-Smirnov test (large samples)

```python
for col in aligned.columns:
    series = aligned[col].dropna()
    stat, p = stats.kstest(series, "norm",
                           args=(series.mean(), series.std()))
    print(f"{col}: KS={stat:.4f}, p={p:.4g}")
```

### 5c. Comparing two depths — Mann-Whitney U (non-parametric)

```python
a = aligned["depth_-0.05m"].dropna()
b = aligned["depth_-0.50m"].dropna()
stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"Mann-Whitney U={stat:.1f}, p={p:.4g}")
```

### 5d. Paired t-test (if data are time-aligned)

```python
pair = aligned[["depth_-0.05m", "depth_-0.50m"]].dropna()
stat, p = stats.ttest_rel(pair["depth_-0.05m"], pair["depth_-0.50m"])
print(f"Paired t={stat:.4f}, p={p:.4g}")
```

### 5e. Kruskal-Wallis (compare all depths at once)

```python
groups = [aligned[col].dropna().values for col in aligned.columns]
stat, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H={stat:.4f}, p={p:.4g}")
```

### 5f. Pearson / Spearman correlation between two depths

```python
pair = aligned[["depth_-0.05m", "depth_-0.20m"]].dropna()

r_pearson, p_pearson  = stats.pearsonr( pair.iloc[:, 0], pair.iloc[:, 1])
r_spearman, p_spearman = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])

print(f"Pearson  r={r_pearson:.4f}, p={p_pearson:.4g}")
print(f"Spearman r={r_spearman:.4f}, p={p_spearman:.4g}")
```

---

## 6. Linear regression and least squares

### 6a. Simple linear regression: one depth as a predictor of another

```python
import numpy as np
from scipy import stats

pair = aligned[["depth_-0.05m", "depth_-0.50m"]].dropna()
x = pair["depth_-0.05m"].values
y = pair["depth_-0.50m"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"slope={slope:.4f}, intercept={intercept:.2f}")
print(f"R²={r_value**2:.4f}, p={p_value:.4g}, SE={std_err:.4f}")
```

### 6b. Least squares via NumPy (equivalent, but gives residuals directly)

```python
# Build design matrix [1, x]
A = np.column_stack([np.ones_like(x), x])
coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
intercept_ls, slope_ls = coeffs
print(f"slope={slope_ls:.4f}, intercept={intercept_ls:.2f}")
print(f"Sum of squared residuals: {residuals[0]:.2f}" if residuals.size else "")
```

### 6c. Multiple linear regression: predict depth from several depths

```python
from numpy.linalg import lstsq

target_col = "depth_-0.50m"
predictor_cols = ["depth_-0.05m", "depth_-0.20m", "depth_-0.35m"]

df_reg = aligned[[target_col] + predictor_cols].dropna()
X = np.column_stack([np.ones(len(df_reg))] + [df_reg[c].values for c in predictor_cols])
y = df_reg[target_col].values

coeffs, _, _, _ = lstsq(X, y, rcond=None)
print("intercept:", coeffs[0])
for name, coeff in zip(predictor_cols, coeffs[1:]):
    print(f"  {name}: {coeff:.4f}")

y_hat = X @ coeffs
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")
```

### 6d. Regression with statsmodels (full OLS summary table)

```powershell
pip install statsmodels
```

```python
import statsmodels.api as sm

df_reg = aligned[["depth_-0.50m", "depth_-0.05m", "depth_-0.20m"]].dropna()
X = sm.add_constant(df_reg[["depth_-0.05m", "depth_-0.20m"]])
y = df_reg["depth_-0.50m"]

model = sm.OLS(y, X).fit()
print(model.summary())
```

The `summary()` output includes coefficients, standard errors, t-statistics, p-values,
confidence intervals, R², adjusted R², F-statistic, and information criteria (AIC/BIC).

### 6e. CO2 concentration vs. time (trend analysis)

```python
# Convert timestamps to elapsed minutes since start
series = aligned["depth_-0.05m"].dropna()
t = (series.index - series.index[0]).total_seconds() / 60   # minutes
co2 = series.values

slope_t, intercept_t, r, p, se = stats.linregress(t, co2)
print(f"Trend: {slope_t:.4f} ppm/min  (R²={r**2:.4f}, p={p:.4g})")
```

---

## 7. Quick plotting of regression results

```python
import matplotlib.pyplot as plt

pair = aligned[["depth_-0.05m", "depth_-0.50m"]].dropna()
x = pair["depth_-0.05m"].values
y = pair["depth_-0.50m"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots()
ax.scatter(x, y, s=8, alpha=0.4, label="observations")
ax.plot(x_line, y_line, color="tab:red",
        label=f"OLS  y={slope:.3f}x+{intercept:.1f}\n$R^2$={r_value**2:.3f}, p={p_value:.3g}")
ax.set_xlabel("CO2 at −0.05 m [ppm]")
ax.set_ylabel("CO2 at −0.50 m [ppm]")
ax.legend()
fig.tight_layout()
fig.savefig("co2_regression_-0.05m_vs_-0.50m.png", dpi=150)
plt.close(fig)
```

---

## 8. Required packages

All packages below are available once the standard virtual environment is active
(see [README.md](../README.md)).  
`scipy` and `statsmodels` are **not** in `requirements.txt` yet; install them
separately as needed:

| Package | Purpose |
|---|---|
| `pandas` | time series alignment and descriptive stats |
| `numpy` | array operations and `lstsq` |
| `scipy` | parametric / non-parametric tests and `linregress` |
| `statsmodels` | full OLS summary with confidence intervals |
| `matplotlib` | plotting regression results |

```powershell
pip install scipy statsmodels
```

---

## 9. Where to put analysis scripts

Place standalone analysis scripts in the `scripts/` folder at the project root.
Name them descriptively, for example:

```
scripts/
    co2_regression_basalt_vertical.py
    co2_statistical_tests.py
```

These scripts can import `co2_vertical_profile_viewer` as a module (as shown in
section 2a above) without modifying the viewer files.
