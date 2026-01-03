# Techno-Economic Analysis of PV-BESS under Dynamic Tariffs

## 📌 Project Overview
This project simulates and optimizes the operation of residential Photovoltaic (PV) systems coupled with Battery Energy Storage Systems (BESS) in Germany in 2024 under dynamic tariffs. It specifically evaluates the economic impact of **dynamic electricity tariffs** (based on EPEX Spot 2024 prices) compared to standard rule-based operation strategies that are insensitive to price signals.

The purpose of this documentation is to ensure transparency and to demonstrate the interim steps during the simulation of the thesis **"Techno-Economic Assessment of Residential BESS with PV and Dynamic Tariffs"** written by **Murat Kuş** in 2025. This thesis was written as part of the **"Sustainable Energy Systems"** Master's program at the University of Applied Sciences.

**This project was supported by VERBUND AG.**

The analysis is structured into three core components:
1.  **Base Case Analysis**: Comparing techno-economic results of Optimized BESS, Rule-Based BESS, and the **Base Case (PV without BESS)** scenario for a single representative household in Bavaria, Munich.
2.  **Sensitivity Analysis**: Evaluating system profitability (ROCE) across a wide range of PV, Battery, and Inverter sizing configurations. This includes identifying optimal BESS and Inverter sizes for fixed PV capacities (5, 10, 15 kWp).
3.  **Germany-Wide Impact**: Aggregating results across all 16 federal states to estimate the potential grid impact and economic benefits of widespread BESS adoption throughout the country.

---

## 📂 Repository Structure

### `00_input_data/`
Contains the raw input CSV files required for simulations, including:
* **Standardized Load Profiles**: BDEW standard profiles (H0, etc.) and processing scripts.
* **PV Generation Data**: Generation profile of a 1kWp PV system for the capital city of each federal state in Germany.
* **Load Profiles**: Processed residential load curves (H0).
* **Market Data**: EPEX Day-Ahead spot market prices for 2024.

### `01_techno-economic_analysis/`
Scripts for the detailed analysis of a single representative household.
* **`script/`**:
    * `optimization.py`: MILP model for maximizing revenue under dynamic tariffs.
    * `rule_based.py`: Simulation of standard self-consumption maximization logic.
    * `base_case.py`: Simulation of the reference scenario (PV system without battery storage).
    * `visualization.py`: Generates comparative plots between the three scenarios.
* **`results/`**: Detailed hourly operation logs and financial summaries.

### `02_sensitivity_analysis/`
Scripts and results for evaluating different system sizes.
* **`script/`**:
    * `config_generator.py`: Generates the parameter grid of test configurations (PV/Battery/Inverter sizes) with applied constraints.
    * `sensitivity.py`: The main simulation engine running the optimization loop for each configuration and outputs the results.
    * `sensitivity_visualization.py`: Generates barplots, heatmaps, and bubble plots of the results.
* **`data/`**: Configuration files and input data for a household in Bavaria, Munich.
* **`results/`**: Output CSVs, Excel files for summary, and visualizations.

### `03_germany_impact/`
Scripts for the system-level impact analysis when optimization is deployed throughout the country compared to rule-based operation.
* **`script/`**:
    * `FS_optimized.py`: Optimization model applied to all 16 states with state specifics.
    * `FS_rule_based.py`: Rule-based operation (self-consumption maximization) simulation for all 16 states.
    * `FS_visualization.py`: Generates comparative plots (bar charts, line plots) for grid exchange shifts.
* **`data/`**: State-specific profiles, number of households with BESS, and BESS capacity data (`FS_Battery.csv`).
* **`results/`**: Aggregated CSV results and final impact barplots/heatmaps for each day in 2024.

---

## 🚀 How to Run

### Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn pulp openpyxl
```

### 1. Running Single Household Analysis
Navigate to the techno-economic analysis folder:
```bash
cd 01_techno-economic_analysis/script
```
Run the scenarios:
```bash
python optimization.py
python rule_based.py
python base_case.py
python visualization.py
```

### 2. Running Sensitivity Analysis
Navigate to the sensitivity script folder:
```bash
cd ../../02_sensitivity_analysis/script
```
Run the pipeline in this order:
1.  **Generate Configurations**:
    ```bash
    python config_generator.py
    ```
2.  **Run Simulations** (This may take time depending on your CPU):
    ```bash
    python sensitivity.py
    ```
3.  **Generate Plots**:
    ```bash
    python sensitivity_visualization.py
    ```
*Results will be available in `02_sensitivity_analysis/results/visualizations`.*

### 3. Running Germany-Wide Impact Analysis
Navigate to the impact analysis script folder:
```bash
cd ../../03_germany_impact/script
```
Run the analysis scripts:
1.  **Run Optimized Scenario**:
    ```bash
    python FS_optimized.py
    ```
2.  **Run Rule-Based Scenario**:
    ```bash
    python FS_rule_based.py
    ```
3.  **Generate Comparative Visualizations**:
    ```bash
    python FS_visualization.py
    ```
*Results will be available in `03_germany_impact/results/visualizations`.*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details."# optimization" 
