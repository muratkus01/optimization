# Input Data Documentation

This directory contains the raw data files used across all simulation components (Single Household, Sensitivity, and Germany-Wide Analysis).

---

## 1. Standardized Load Profile (SLP)
**Used in:** `01_techno-economic_analysis`, `02_sensitivity_analysis`, `03_germany_impact` (as base for scaling)

* **Source**: **BDEW - Standardlastprofile Strom** (Bundesverband der Energie- und Wasserwirtschaft).
* **Description**: This folder contains the raw data and processing scripts for standard load profiles used to model representative consumption patterns.
* **Contents**:
    * **Raw Data**: `.csv` files derived from `Repräsentative Profile VDEW.xls` and `Haushalt Lastprofil.xls`, covering household (H0) as well as commercial/agricultural profiles (G0-G6, L0-L2).
    * **Processing Script** (`SLP.py`): A Python script used to process the raw quarterly-hour values into usable hourly profiles.
    * **Visualizations**: A `visualization/` subfolder containing plots of the various load profiles.
* **Key Usage**: The **H0 (Household)** profile serves as the foundational curve for generating the `P_Load` time series in all subsequent simulations, scaled to the specific annual consumption (e.g., 3,221 kWh for the representative household).

## 2. Market Data (`EPEX_2024.csv`)
**Used in:** `01_techno-economic_analysis`, `02_sensitivity_analysis`, `03_germany_impact`

* **Description**: Hourly Day-Ahead electricity prices for the German bidding zone (DE-LU).
* **Source**: EPEX SPOT / SMARD.de (Bundesnetzagentur).
* **Timeframe**: January 1, 2024 – December 31, 2024.
* **Key Columns**:
    * `EP_epex_MWh`: Raw spot market price.
    * `EP_buy_MWh`: Dynamic consumer price (Spot price + grid fees + levies + taxes).
    * `EP_sell_MWh`: Selling price (Market value or Feed-in Tariff).

## 3. Household Profiles (`2024_HSS_Baveria.csv`)
**Used in:** `01_techno-economic_analysis`

* **Description**: Detailed hourly generation and consumption profile for a single representative household located in **Bavaria (Munich)**.
* **Purpose**: This dataset serves as the primary input for the detailed Techno-Economic Analysis, comparing the three main scenarios:
    1.  **Optimized BESS**: Price-driven operation.
    2.  **Rule-Based BESS**: Solar-driven operation.
    3.  **Base Case**: PV system without battery storage.
* **Key Columns**:
    * `P_PV`: PV generation (kW).
    * `P_Load`: Household electricity consumption (kW).

## 4. Sensitivity Base Data (`HSS_Bavaria_PV_1.csv`)
**Used in:** `02_sensitivity_analysis`

* **Description**: A normalized version of the Bavaria household profile with PV generation scaled to **1 kWp**.
* **Purpose**: Serves as the foundational dataset for the Sensitivity Analysis. The simulation scripts (`sensitivity.py`) scale this profile dynamically to test thousands of configuration combinations (varying PV size, Battery capacity, and Inverter power) to find the optimal system design.
* **Key Columns**:
    * `P_PV_1`: Normalized 1 kWp PV generation profile.
    * `P_Load`: Standard household load.

## 5. Federal State PV Profiles (`1kWp_PV_profiles_per_state/`)
**Used in:** `03_germany_impact`

* **Files**: e.g., `pv_profile_Baden-Württemberg.csv`, `pv_profile_Berlin.csv`, etc.
* **Description**: Contains hourly generation profiles for a **1 kWp rooftop PV system** located in the **capital city** of each of the 16 German federal states.
* **Source**: PVGIS (CMSAF) database.
* **Purpose**: These profiles are essential for the Germany-Wide Impact Analysis. In the simulation, these normalized profiles are scaled up by the total installed PV capacity of each respective state to model the aggregate generation accurately.

## 6. Battery Configuration (`FS_Battery.csv`)
**Used in:** `03_germany_impact`

* **Description**: Configuration file defining the total installed capacities for PV and BESS per federal state.
* **Source**: Derived from the **Marktstammdatenregister (MaStR)** (Core Energy Market Data Register).
* **Purpose**: Provides the scaling factors (Number of BESS units, Total Energy Capacity, Total Inverter Power) used to upscale the single-household simulation results to the state level.
* **Key Columns**:
    * `federal_state`: Name of the state.
    * `number_hss`: Total count of registered Home Storage Systems.
    * `total_bess_energy_MWh`: Total installed battery energy capacity.
    * `total_inv_power_MW`: Total installed battery inverter power.

## 7. Federal State Aggregated Profiles (`Federal State Profiles` / `[State].csv`)
**Used in:** `03_germany_impact`

* **Files**: e.g., `Bavaria.csv`, `Hesse.csv`.
* **Description**: These files represent the **state-wide aggregate** generation and load profiles.
* **Scaling Logic**: The values for `P_PV_8_MWh` and `P_Load_MWh` in these files have been **scaled up** based on the `number_hss` (number of households) defined in `FS_Battery.csv`.
* **Key Columns**:
    * `P_PV_8_MWh`: Total PV generation for the state's residential sector (in MWh).
    * `P_Load_MWh`: Total household load for the state (in MWh).