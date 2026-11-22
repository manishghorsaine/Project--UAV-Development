# UAV Propulsion System Development

This project evaluates electric propulsion-system combinations for a fixed-wing UAV. It combines propeller performance measurements, motor and battery specifications, and an aircraft drag polar to estimate operating points, screen hardware combinations against design constraints, and visualize propulsion performance.

## Project contents

```text
.
├── Database_CSV/
│   ├── 8x6.csv, 8x8.csv, 9x6.csv, 9x7.5.csv, 9x9.csv  # Propeller coefficients
│   ├── Motor_Datas.csv                                  # Motor electrical limits and constants
│   ├── Battery_Database.csv                             # Battery voltage, capacity, and discharge data
│   ├── drag.csv                                         # Drag source data
│   └── drag_vs_velocity.csv                             # Aircraft drag versus airspeed
├── plots_out/
│   ├── OverallEff_ID1_ID2_ID3_V12_13_20.png
│   ├── TD_vs_V_FullThrottle_ID20.png
│   ├── TorqueMatch_V20_ID20.png
│   └── V12_13_20_rpm_Ibatt_Pbatt_ID20_26_36.csv
├── Initial_Check_All_75_Configurations.csv              # Existing 75-configuration audit
├── Initial_Constraints_Check_All_75_Configurations.py   # Constraint screening calculation
├── Performance_Evaluation_Propulsion_System.py          # Detailed single-configuration analysis
└── plots_all.py                                         # Local-data plotting and comparison workflow
```

## Analysis workflow

The design space contains five propellers, five motors, and three batteries (5 × 5 × 3 = 75 combinations). Each combination is assigned an ID in nested propeller → motor → battery order, starting at 1.

The analysis interpolates propeller thrust and power coefficients (`Ct`, `Cp`) against advance ratio (`J`). For each airspeed and throttle setting, it searches a 1,000–18,000 rpm grid for the motor/propeller torque equilibrium. The electrical model accounts for motor constants and resistance, battery voltage sag, and an approximate throttle-dependent ESC efficiency. A bisection search finds throttle settings when a specified thrust is required.

The screening script checks maximum shaft power, motor current, battery current, maximum speed (at least 25 m/s), level-flight feasibility, and climb thrust. The detailed analysis evaluates a selected configuration over the speed range and estimates static thrust, cruise, climb, and take-off operating points. `plots_all.py` generates torque matching, full-throttle thrust-versus-drag, and efficiency comparison plots, plus a CSV of selected operating-point results.

## Data files

- Propeller CSVs contain `J`, `Ct`, and `Cp` columns. The `8x6`-style filename indicates nominal diameter × pitch in inches; the diameter is converted to metres in the model.
- `Motor_Datas.csv` contains `Name`, `KV`, `I0`, `Ri`, `Power_max`, and `Max_Current`.
- `Battery_Database.csv` contains battery names, nominal voltage, series cell count, capacity, and C-rates.
- `drag_vs_velocity.csv` contains airspeed and total aircraft drag; drag is linearly interpolated at the analysis speed.
- `Initial_Check_All_75_Configurations.csv` is a precomputed screening result. Its component names (for example `APC_8x6`, `A30-22S`, and `SLS_3S_1800_50C`) differ from the generic names used by the scripts and local databases (`8x6`, `MotorA`, `BatteryI`). Treat it as a saved result set from a different naming/data revision, not as output directly reproducible from the current screening script without reconciling its inputs and names.

## Requirements

Python 3 with the following packages:

```text
numpy
pandas
matplotlib
```

Install them in your environment with:

```bash
python -m pip install numpy pandas matplotlib
```

## Running the analysis

Run commands from the project root so relative paths and generated files resolve as expected:

```bash
python plots_all.py
```

`plots_all.py` reads the databases from `Database_CSV/`, writes figures and the operating-point export to `plots_out/`, and opens plot windows through Matplotlib. Change `TARGET_ID` near the top of the script to select a configuration from 1 to 75. The current comparison IDs are set by `COMPARE_IDS` in the same script.

The other two Python scripts currently contain absolute Windows paths pointing to a separate `E:\Aerospace\...` directory structure. Before running either script on another machine (or with this project folder), update its `PROP_DIR`, `MOTOR_FILE`, `BATTERY_FILE`, and, where applicable, `DRAG_FILE` values to point to the local `Database_CSV/` files. The screening script also expects its input component names and column names to match its hard-coded configuration tables. Its output files, `All_Configurations_Audit.csv` and `Working_Configurations.csv`, are written to the current working directory.

## Existing results

The checked-in audit CSV contains 75 configurations: 30 marked `WORK` and 45 marked `NOT WORK`. The three PNGs in `plots_out/` show a torque-matching result at 20 m/s, a full-throttle thrust/drag curve for configuration 20, and an efficiency comparison. The accompanying operating-point CSV records RPM, battery current, and battery power at 12, 13, and 20 m/s for configuration IDs 20, 26, and 36.

In the efficiency figure/export, those configuration IDs are relabeled `ID1`, `ID2`, and `ID3` by `DISPLAY_ID_MAP`. These labels are plot labels and do not refer to configuration IDs 1, 2, and 3 in the 75-combination enumeration.

## Model assumptions and limitations

- Air density is fixed at 1.157 kg/m³ in the analysis scripts.
- The battery internal-resistance relation and ESC efficiency are simplified engineering models, not measurements from a complete powertrain test.
- Propeller coefficient queries outside the measured advance-ratio range are clipped to the available range before interpolation.
- Equilibrium is selected from a finite RPM grid and accepted within a 2% torque-mismatch tolerance.
- The constraint-screening and detailed-evaluation scripts use different mass and stall-speed assumptions; confirm these design inputs before comparing results across scripts.
- The detailed-analysis script contains hard-coded drag arrays and absolute input paths, while `plots_all.py` uses the local drag CSV. For repeatable results, use one consistent data source and align the assumptions across scripts.
- These calculations are preliminary analytical estimates and should be validated with bench and flight testing before hardware selection.
