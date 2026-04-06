# CO2 Vertical Profile: Correlation and Time Series Analysis

## Assignment Overview

This assignment involves analyzing CO2 time series data from one vertical profile in the Biosphere 2 LEO (Landscape Evolution Observatory) facility. Each student will work with a specific vertical location consisting of 4 sensors:
- 1 air sensor at 25 cm above surface
- 3 basalt sensors at 5 cm, 20 cm, and 50 cm below surface

## Student Assignments

| Student | Coordinates (x, y) | Location |
|---------|-------------------|----------|
| Matthew Etherington | (1, 24) | LEO West, x=1, y=24 |
| Chris Samuel | (-1, 4) | LEO West, x=-1, y=4 |
| Matias Contreras | (-1, 24) | LEO West, x=-1, y=24 |

---

## Background and Scientific Context

### Why This Analysis Matters

CO2 transport in basalt occurs through diffusion, which is much slower than atmospheric CO2 changes. Understanding the relationship between these processes requires:

1. **Autocorrelation**: How long does each sensor "remember" its past state?
2. **Cross-correlation with lag**: How long does it take for atmospheric changes to reach different depths?
3. **Central Limit Theorem (CLT) interval**: At what time scale do averages become normally distributed?
4. **Principal Component Analysis**: What are the dominant modes of variability in the vertical profile?

This analysis will help:
- Estimate diffusion coefficients at different depths
- Understand time scales for writing differential equations
- Identify which sensors provide the most information
- Enable statistical mechanics approaches (path integrals) for modeling

---

## Data Sources and Workflow

### Step 1: Identify Your Sensors

Use the sensor catalog to find the sensor IDs for your vertical:

**Catalog**: [variables_schema.xlsx](https://github.com/DimitriBolt/CO2Flux/blob/main/Sensors_Description/variables_schema.xlsx)

Open the `CO2` sheet and filter by:
- **Slope**: `LEO West`
- **x_coord_m**: your assigned x coordinate
- **y_coord_m**: your assigned y coordinate

You should find:
- **1 air sensor** with `B = C_CO2,air`, `K = LI-COR`, height = 0.25 m
- **3 basalt sensors** with `B = C_CO2,basalt`, `K = GMM222`, at depths 5 cm, 20 cm, and 50 cm

Record for each sensor:
- `sensorid`
- `table_name` (e.g., `leo_west.datavalues` or `leo_west.datavalueslicor`)
- `variableid`
- Depth/height

### Step 2: Determine the Ideal Period

Check the corresponding ideal period notebook in the repository:

**Ideal Period Notebooks**: [Ideal_period folder](https://github.com/DimitriBolt/CO2Flux/tree/main/Project_description/sensorDB/Ideal_period)

Find the file matching your coordinates:
- Matthew: `co2_ideal_period_LEO_West_x1_y24.ipynb`
- Chris: `co2_ideal_period_LEO_West_x1_y4.ipynb`
- Matias: `co2_ideal_period_LEO_West_x-1_y24.ipynb`

From the notebook, extract:
- **Start timestamp** of the ideal period
- **End timestamp** of the ideal period

### Step 3: Extract Time Series from Oracle Database

Using the sensor IDs and ideal period dates, query the Oracle database to get the 4 time series.

**For air sensor** (from `leo_west.datavalueslicor`):
```sql
SELECT
    localdatetime,
    datavalue
FROM
    leo_west.datavalueslicor
WHERE
    sensorid = <your_air_sensor_id>
    AND variableid = 56
    AND localdatetime >= TO_DATE('<start_date>', 'YYYY-MM-DD HH24:MI')
    AND localdatetime <= TO_DATE('<end_date>', 'YYYY-MM-DD HH24:MI')
ORDER BY
    localdatetime
```

**For basalt sensors** (from `leo_west.datavalues`):
```sql
SELECT
    localdatetime,
    datavalue
FROM
    leo_west.datavalues
WHERE
    sensorid = <your_basalt_sensor_id>
    AND variableid = 9
    AND localdatetime >= TO_DATE('<start_date>', 'YYYY-MM-DD HH24:MI')
    AND localdatetime <= TO_DATE('<end_date>', 'YYYY-MM-DD HH24:MI')
ORDER BY
    localdatetime
```

**Important data preprocessing**:
- Basalt data may have multiple observations per 15-minute slot
- Floor timestamps to 15-minute intervals: `timestamp.floor('15min')`
- Average all values within each 15-minute slot
- Align all 4 time series to the same 15-minute timestamp grid

---

## Analysis Tasks

### Task 1: Basic Statistics

**What to compute:**

For each of the 4 time series:
1. Mean CO2 concentration (ppm)
2. Standard deviation (ppm)
3. Minimum and maximum values
4. Number of data points

**Why this matters:**
- Provides baseline understanding of each sensor
- Air should have lowest mean (~430 ppm)
- Basalt should increase with depth due to CO2 accumulation
- Standard deviation indicates variability at each level

**Deliverables:**
- Table with statistics for all 4 sensors
- Brief interpretation: Does the pattern make physical sense?

---

### Task 2: Autocorrelation Analysis

**What to compute:**

For each of the 4 time series, calculate the autocorrelation function (ACF).

**Procedure:**
1. Center the time series: subtract the mean
2. Compute autocorrelation for lags from 0 to 96 time steps (24 hours at 15-min intervals)
3. Plot ACF vs. time lag (in hours)
4. Determine the **characteristic autocorrelation time**: the lag where ACF drops to ~0.37 (1/e)

**Why this matters:**
- **Physical interpretation**: How long does each sensor "remember" its past state?
- **Air sensor**: Should have short memory (minutes to hours) - responds quickly to atmospheric changes
- **Basalt sensors**: Should have longer memory (hours to days) - slow diffusion processes
- **Depth dependence**: Deeper sensors should have longer autocorrelation times
- **For modeling**: The autocorrelation length defines the time scale for differential equations

**Deliverables:**
1. Plot: 4 autocorrelation functions on one graph (different colors for each sensor)
2. Table: Characteristic autocorrelation time for each sensor (in hours)
3. Interpretation: Does the autocorrelation time increase with depth as expected?

---

### Task 3: Cross-Correlation with Time Lag

**What to compute:**

Calculate time-lagged cross-correlation between pairs of sensors to find how long it takes for changes to propagate through the vertical profile.

**Sensor pairs to analyze:**
1. Air (25 cm) ↔ Basalt 5 cm
2. Air (25 cm) ↔ Basalt 20 cm
3. Air (25 cm) ↔ Basalt 50 cm
4. Basalt 5 cm ↔ Basalt 20 cm
5. Basalt 20 cm ↔ Basalt 50 cm

**Procedure:**
1. Center both time series (subtract their means)
2. For each pair, compute cross-correlation for lags from -96 to +96 time steps (-24 to +24 hours)
   - Positive lag: second sensor lags behind first sensor
   - Negative lag: second sensor leads first sensor
3. Find the lag with **maximum cross-correlation** (peak of the function)
4. Plot cross-correlation vs. lag for each pair

**Why this matters:**
- **Physical interpretation**: Measures the propagation delay of CO2 changes through the profile
- **Expected behavior**:
  - Air changes should propagate downward with increasing delay
  - Delay between air and basalt 5 cm should be shorter than delay to basalt 50 cm
- **Diffusion coefficient estimation**: The time lag between depths is related to the diffusion coefficient
  - If a change takes time Δt to travel distance Δz, then D ~ Δz² / Δt
- **For modeling**: Determines appropriate boundary conditions and coupling terms in differential equations

**Deliverables:**
1. Plots: Cross-correlation function for each of the 5 sensor pairs
2. Table: Optimal time lag (in hours) and maximum correlation value for each pair
3. Interpretation:
   - Do deeper sensors show increasing lag relative to air?
   - What does this tell you about CO2 transport rates?
   - Rough estimate of diffusion coefficient (if possible)

---

### Task 4: Central Limit Theorem (CLT) Interval

**What to compute:**

Find the minimum time interval at which moving averages become normally distributed.

**Procedure:**

For each of the 4 time series:

1. Test multiple averaging intervals: **1 hour, 6 hours, 12 hours, 1 day, 3 days**
   - 1 hour = 4 time steps (15-min intervals)
   - 6 hours = 24 time steps
   - 12 hours = 48 time steps
   - 1 day = 96 time steps
   - 3 days = 288 time steps

2. For each interval length:
   a. Compute **moving average** over that interval (sliding window)
   b. Collect all the moving average values into an array
   c. Test if these averages are normally distributed:
      - **Visual test**: Create Q-Q plot (quantile-quantile plot against normal distribution)
      - **Visual test**: Create histogram with overlay of fitted normal distribution
      - **Statistical test**: Shapiro-Wilk test for normality (p-value > 0.05 indicates normality)

3. Identify the **minimum interval** where normality is achieved (p-value > 0.05)

**Why this matters:**
- **Central Limit Theorem**: States that averages over sufficiently long intervals become normally distributed, regardless of the underlying distribution
- **For statistical modeling**:
  - Once we know the CLT interval, we can apply Gaussian statistics
  - Enables path integral formulation (Feynman path integrals)
  - Parameters in differential equations can be treated as normally distributed random variables
- **Time scale for modeling**: Tells us the appropriate time resolution for writing equations
  - If CLT works at 6 hours, we should model on 6-hour time scales
  - Below this interval, data may not be Gaussian

**Deliverables:**

For each sensor:
1. Table: Shapiro-Wilk test results (p-value) for each interval length
2. Plots: Q-Q plots for each interval (or at least for the minimum working interval)
3. Plots: Histograms with normal fit for each interval
4. Summary: Minimum CLT interval for each sensor (in hours)
5. Interpretation: Why might different sensors have different CLT intervals?

---

### Task 5: Principal Component Analysis (PCA)

**What to compute:**

Perform PCA on the 4 time series to identify dominant modes of variability.

**Procedure:**

1. **Prepare data matrix**:
   - Create a matrix X with shape (N, 4) where:
     - N = number of time points in ideal period
     - 4 columns = [Air_25cm, Basalt_5cm, Basalt_20cm, Basalt_50cm]

2. **Standardize the data**:
   - For each column (sensor):
     - Subtract the mean (centering)
     - Divide by standard deviation (z-score normalization)
   - This ensures all sensors contribute equally regardless of absolute concentration differences

3. **Perform PCA**:
   - Compute the correlation matrix (4×4)
   - Find eigenvalues and eigenvectors
   - Sort by decreasing eigenvalue

4. **Analyze results**:
   - **Eigenvalues**: How much variance each component explains
   - **Eigenvectors**: How each sensor contributes to each component
   - **Principal components**: Project data onto eigenvector directions

**Why this matters:**
- **Dimensionality reduction**: 4 sensors may not provide 4 independent pieces of information
  - If first 2-3 eigenvalues explain >95% variance, system has fewer independent modes
- **Physical interpretation**:
  - PC1 might represent "overall vertical profile level"
  - PC2 might represent "surface vs. deep contrast"
  - PC3 might represent "intermediate depth variations"
- **Sensor importance**: Eigenvectors show which sensors contribute most to each mode
- **For modeling**: Identifies natural coordinates for describing system dynamics

**Deliverables:**

1. **Correlation matrix**: 4×4 heatmap showing correlations between all sensor pairs

2. **Eigenvalue analysis**:
   - Table: Eigenvalues and percentage of variance explained by each component
   - Plot: Scree plot (eigenvalue vs. component number)
   - Answer: How many components explain 95% of variance?

3. **Eigenvector analysis**:
   - Table: Eigenvector components for each PC
   - Plot: Heatmap or bar chart showing sensor loadings for each PC
   - Interpretation: What does each principal component represent physically?

4. **Projection onto principal components**:
   - Plot: Time series of first 2-3 principal components
   - Interpretation: What temporal patterns do the PCs reveal?

5. **Overall interpretation**:
   - How many independent modes of variability exist in your vertical?
   - Which sensors provide the most unique information?
   - Are all 4 sensors necessary, or is there redundancy?

---

## Suggested Tools and Libraries

**Python libraries:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
```

**Key functions:**
- `pandas.Series.autocorr()` or `statsmodels.tsa.stattools.acf()` for autocorrelation
- `scipy.signal.correlate()` or `numpy.correlate()` for cross-correlation
- `scipy.stats.shapiro()` for Shapiro-Wilk test
- `statsmodels.graphics.gofplots.qqplot()` for Q-Q plots
- `sklearn.decomposition.PCA()` for principal component analysis

---

**Final deliverable format:**
- Your choice: Jupyter notebook, Python script + figures, or PDF report
- Must include all plots, tables, and interpretations requested above
- Code should be well-commented and reproducible

---

## Questions?

If you have questions about:
- Data access or SQL queries → Contact Dimitri
- Scientific interpretation → Contact Professor Gabitov or Joel Maldonado
- Statistical methods → Refer to standard time series analysis references

---

**Good luck with your analysis!**

This work will contribute to understanding CO2 diffusion in basalt and developing physics-based models of subsurface carbon dynamics.
