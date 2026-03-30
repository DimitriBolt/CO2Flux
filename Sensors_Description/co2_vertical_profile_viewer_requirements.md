# CO2 Vertical Profile Viewer Requirements

This requirements file is the shared requirements source for:

- `Sensors_Description/co2_vertical_profile_viewer.py`
- `Sensors_Description/co2_viewer_add_surface.py`

The second script is an extension of the first one, but it inherits the same
viewer requirements unless a later document explicitly overrides them.

## Goal

Create a Python program for visualizing the **CO2 concentration profile along one vertical basalt line** for a selected LEO slope and surface point.

The program is for **one vertical profile per run**.

The purpose is **visual inspection of how strongly CO2 concentration changes with depth**.
The program should **not compute the numerical gradient**. It should only visualize CO2 concentration values at the available sensor depths while preserving the true geometric depth positions and the concentration scale.

## Data source

- Use the Excel workbook as the **catalog** for locating the correct vertical profile.
- Use Oracle as the **source of the actual time series values**.

## Scope of one run

Each run should visualize only **one vertical profile** selected by:

- `slope`
- `X`
- `Y`
- `start_date`
- `end_date`

No multi-profile mode is needed.

## Parameter input style

The user will run the script manually from PyCharm.

All parameters should be defined as **plain variables at the top of the Python file**, not via CLI arguments and not via interactive prompts.

The file should contain comments explaining which parameters can be changed manually.

## What to visualize

The program should show only the **current vertical CO2 profile at one time moment**.

It should **not** show:

- numeric gradient values
- a second plot with time series versus time
- multiple subplots

It should use **one plot only**.

## Plot type

The preferred visualization is:

- **horizontal bars**
- the vertical axis represents **depth**
- the bars are positioned at the real sensor depths
- bar length represents **CO2 concentration**
- each horizontal bar starts at **0 ppm** and extends to the right up to the measured concentration value

Depth must be shown in meters as:

- `-0.05 m`
- `-0.20 m`
- `-0.35 m`
- `-0.50 m`

The program must use the **real available depths as they are**, with no interpolation and no forced common depth grid.

The depth axis should always show the fixed reference levels:

- `0 m`
- `-0.05 m`
- `-0.20 m`
- `-0.35 m`
- `-0.50 m`

If a selected vertical profile does not contain one of these depths, that depth should still remain visible as an empty slot with no bar.

The plot must include an explicit **surface line at `z = 0`**.

The vertical axis should be oriented physically:

- surface at the top
- larger negative depth downward

## Animation

The program must support animation over a **user-defined arbitrary interval**:

- `start_date`
- `end_date`

Animation step:

- **every available change / measurement**

No smoothing:

- no moving average
- no resampling
- no filtering

## Axis behavior

The CO2 concentration axis must be **fixed for the whole selected interval**.

It is **strictly forbidden** to auto-rescale the concentration axis from frame to frame.

The animation must not “breathe” because of changing axis limits.

The agreed default concentration axis for all graphs is:

- `0 ... 8000 ppm`

This fixed range is intended to stay the same across all generated plots so that visual comparison between different verticals remains possible.

## Missing data behavior

If some depths are missing at a given moment:

- show only the depths that are available at that moment

Do **not** stop with an error.

Do **not** attempt interpolation.

For time synchronization between depths, the program should use:

- **last known value**

That means each frame time is taken from the sequence of actual measurement timestamps, and each depth uses the most recent available measurement at or before that frame time.

## Output files

The program must save:

- a `GIF` animation
- a `JPEG` image with the **last state / final frame**

Suggested filename style:

- `co2_profile_<slope>_x<X>_y<Y>_<start>_<end>.gif`
- `co2_profile_<slope>_x<X>_y<Y>_<start>_<end>.jpg`

Existing files with the same name should be:

- **overwritten**

No explicit limit on the number of frames is required.

## What must be shown on the frame

Each frame should contain annotations with:

- time
- slope name
- coordinates
- concentration values

The concentration values should be shown:

- as labels at the end of each horizontal bar

The final JPEG should be:

- the exact last frame / last state of the animation

## Validation assumptions

- The user will manually enter only correct parameters.
- The program does not need a “nearest available point” search.
- The program does not need defensive validation for bad user input.

## Date parameter format

The preferred date-time input format is:

- `YYYY-Mon-DD HH:MM`

Example:

- `2025-Mar-26 15:27`

Month should be written with abbreviated letters.

If only the start date is given and the end date is left empty, the program should run:

- from the given start time
- to the last available observation in the selected profile

## Important interpretation rule

This program is for **visualizing concentration profiles**, not for solving the gradient numerically inside the plotting script.

The scientist will estimate the steepness of the concentration change **visually from the profile**.
