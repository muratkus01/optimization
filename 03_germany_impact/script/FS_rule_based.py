import pandas as pd
import numpy as np
import os
import logging
import time
import glob


# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)


# === Path Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "data") # input folder
results_dir = os.path.join(parent_dir, "results", "FS_rule_based_results") # output folder

os.makedirs(results_dir, exist_ok=True)

# Inputs
config_path = os.path.join(data_dir, "FS_Battery.csv")
price_path = os.path.join(data_dir, "EPEX_2024.csv")
config_df = pd.read_csv(config_path, delimiter=';')
price_df = pd.read_csv(price_path, delimiter=';')

# Clean column names
price_df.columns = price_df.columns.str.strip()
price_cols = [c for c in price_df.columns if c.lower().startswith('ep')]

states = config_df['federal_state'].tolist()
filename_map = {
    "Baden Wuttenberg": "Baden",
    "Bavaria": "Bavaria",
    "Berlin": "Berlin",
    "Brandenburg": "Brandenburg",
    "Bremen": "Bremen",
    "Hamburg": "Hamburg",
    "Hesse": "Hesse",
    "Lower Saxony": "Lower-Saxony",
    "Mecklenburg": "Mecklenburg",
    "Northrhine": "Northrhine",
    "Rhineland-Palatine": "Rhineland-Palatine",
    "Saarland": "Saarland",
    "Saxony": "Saxony",
    "Saxony-Anhalt": "Saxony-Anhalt",
    "Schleswig-Holstein": "Schleswig-Holstein",
    "Thuringia": "Thuringia"
}

# === Main loop over all states ===
for idx, row in config_df.iterrows():
    start_time = time.time()
    state = row['federal_state']
    state_short = filename_map.get(state, state).replace(" ", "-")
    logging.info(f"--- Starting simulation for state: {state} ---")
    try:
        # --- Load state data ---
        data_file = None
        for name_variant in [state_short, state_short.replace("-", " "), state_short.replace(" ", "-")]:
            fpath = os.path.join(data_dir, f"{name_variant}.csv")
            if os.path.exists(fpath):
                data_file = fpath
                break
        if not data_file:
            raise FileNotFoundError(f"CSV data file for state '{state}' not found in {data_dir}.")

        if data_file:
            logging.info(f"📄 Data file used for {state}: {os.path.basename(data_file)}")

        df = pd.read_csv(data_file, sep=";", encoding="utf-8")
        df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
        df["Day"] = df["DateTime"].dt.dayofyear
        df["Hour"] = df["DateTime"].dt.hour

        # --- Merge price data ---
        for col in price_cols:
            df[col] = price_df[col].values[:len(df)]

        # === Extract Parameters ===
        number_hss = int(row["number_hss"]) # Number of Households
        E_bess = float(row["total_bess_energy_MWh"])  # Now total battery energy for the state
        P_inv = float(row["total_inv_power_MW"])  # Now total inverter power for the state
        P_pv = 8 * number_hss  # Still per-household scaling, change if needed
        T = len(df)

        eta_batt = 0.95
        eta_inv = 0.97
        eta_c = eta_d = eta_batt * (eta_inv * eta_inv)
        SOC_max = E_bess * 0.95
        SOC_min = E_bess * 0.05
        SOC_0 = E_bess * 0.5
        DoD = 0.90
        P_ch_max = P_dc_max = P_inv
        E_max = 2 * (SOC_max - SOC_min)
        fixed_annual_cost = (5 + 4.18285 + 3.3333) * 12 * number_hss
        daily_fixed_costs = fixed_annual_cost / 365
        CAPEX = (P_pv * 130 + E_bess * 350 + P_inv * 190) * number_hss

        P_load = pd.to_numeric(df["P_Load_MWh"], errors='coerce').values
        P_PV = pd.to_numeric(df[df.columns[df.columns.str.contains("P_PV_8_MWh")][0]], errors='coerce').values
        P_prosumer = P_PV - P_load
        EP_epex = pd.to_numeric(df["EP_epex_MWh"], errors='coerce').values
        EP_buy = pd.to_numeric(df.get("EP_buy_MWh", df["EP_epex_MWh"]), errors='coerce').values
        EP_sell = pd.to_numeric(df.get("EP_sell_MWh", df["EP_epex_MWh"]), errors='coerce').values

        # --- Output dirs ---
        state_output_dir = os.path.join(results_dir, f"FS_{state_short}_rule_based")
        os.makedirs(state_output_dir, exist_ok=True)
        os.makedirs(os.path.join(state_output_dir, "csv"), exist_ok=True)
        excel_path = os.path.join(state_output_dir, f"FS_{state_short}_rule_based.xlsx")

        # --- Optimization Model ---
        logging.info(f"Solving Rule-Based for {state} ({T} hours, {number_hss} households)...")

        annual_earning_battery_dc_to_grid = annual_earning_battery_pv_to_grid = 0
        annual_opportunity_cost_ch_from_pv = annual_cost_load_from_grid = annual_cost_ch_from_grid = 0
        annual_saving_from_avoided_grid_import = annual_earnings = 0
        annual_energy_cost = annual_net_bill = annual_optimization_profit = 0
        annual_import = annual_export = annual_cycle = 0

        hours_per_day = 24

        # --- Results Extraction ---
        hourly_results = []
        daily_summary = []
        SOC = [SOC_0]
        for day in sorted(df["Day"].unique()):
            hourly_df = df[df["Day"] == day]
            indices = hourly_df.index.tolist()
            cumulative_energy = 0
            daily_P_from = daily_P_to = daily_P_dc = daily_P_ch = daily_cycle = 0
            daily_earning_battery_dc_to_grid = daily_earning_battery_pv_to_grid = 0
            daily_opportunity_cost_ch_from_pv = daily_saving_from_avoided_grid_import = 0
            daily_cost_ch_from_grid = daily_cost_load_from_grid = daily_earnings = 0
            daily_energy_cost = daily_net_bill = daily_optimization_profit = 0
            daily_pv_used_on_site = daily_load_met_without_grid = daily_pv = daily_load = 0

            for t in indices:
                SOC_prev_t = SOC[-1]
                surplus_t = P_prosumer[t]
                deficit_t = -surplus_t if surplus_t < 0 else 0

                p_ch_t = p_dc_t = p_from_t = p_to_t = 0

                if surplus_t > 0:
                    charge_limit = min(surplus_t, (SOC_max - SOC_prev_t) / eta_c, P_ch_max)
                    p_ch_t = min(charge_limit, max(0, E_max - cumulative_energy))
                    p_to_t = max(0, surplus_t - p_ch_t)
                elif deficit_t > 0:
                    discharge_limit = min(deficit_t, (SOC_prev_t - SOC_min) * eta_d, P_dc_max)
                    p_dc_t = min(discharge_limit, max(0, E_max - cumulative_energy))
                    p_from_t = max(0, deficit_t - p_dc_t)

                cumulative_energy += p_ch_t + p_dc_t
                SOC_t = SOC_prev_t + eta_c * p_ch_t - p_dc_t / eta_d
                SOC.append(SOC_t)

                p_PV_t = P_PV[t]
                p_load_t = P_load[t]
                p_prosumer_t = surplus_t
                ep_buy_t = EP_buy[t]
                ep_sell_t = EP_sell[t]
                ep_epex_t = EP_epex[t]

                cycle_t = p_dc_t / (E_bess * DoD)

                pv_surplus_t = max(0, p_PV_t - p_load_t)
                p_ch_from_pv_t = min(pv_surplus_t, p_ch_t)
                p_ch_from_grid_t = max(0, p_ch_t - p_ch_from_pv_t)

                p_dc_to_load_t = min(p_load_t - p_PV_t, p_dc_t) if p_load_t > p_PV_t else 0
                p_dc_to_grid_t = p_dc_t - p_dc_to_load_t

                p_pv_on_site_t = min(p_PV_t, p_load_t + p_ch_t)
                p_load_met_t = min(p_load_t, p_PV_t + p_dc_t)

                p_pv_to_grid_t = max(0, p_PV_t - p_pv_on_site_t)
                p_load_from_grid_t = max(0, p_load_t - p_load_met_t)

                p_grid_net_t = p_from_t - p_to_t
                p_bess_net_t = p_dc_t - p_ch_t

                ep_epex_t = EP_epex[t]
                ep_buy_t = EP_buy[t]
                ep_sell_t = EP_sell[t]

                saving_from_avoided_grid_import_t = ep_buy_t * p_dc_to_load_t
                earning_battery_dc_to_grid_t = ep_sell_t * p_dc_to_grid_t
                earning_battery_pv_to_grid_t = ep_sell_t * p_pv_to_grid_t

                opportunity_cost_ch_from_pv_t = ep_sell_t * p_ch_from_pv_t
                cost_ch_from_grid_t = ep_buy_t * p_ch_from_grid_t
                cost_load_from_grid_t = ep_buy_t * p_load_from_grid_t

                energy_cost_t = cost_ch_from_grid_t + cost_load_from_grid_t
                earnings_t = earning_battery_dc_to_grid_t + earning_battery_pv_to_grid_t
                net_bill_t = earnings_t - energy_cost_t

                optimization_profit_t = (
                    saving_from_avoided_grid_import_t +
                    earning_battery_dc_to_grid_t +
                    earning_battery_pv_to_grid_t -
                    opportunity_cost_ch_from_pv_t -
                    cost_ch_from_grid_t -
                    cost_load_from_grid_t
                )
                self_consumption_t = (p_pv_on_site_t / p_PV_t) * 100 if p_PV_t > 0 else 0
                self_sufficiency_t = (p_load_met_t / p_load_t) * 100 if p_load_t > 0 else 0

                hourly_results.append({
                    "DateTime": hourly_df["DateTime"].iloc[t - indices[0]],
                    "Day": df["Day"].iloc[t],
                    "Hour": hourly_df["Hour"].iloc[t - indices[0]],
                    "P_Load": p_load_t,
                    "P_PV": p_PV_t,
                    "P_prosumer": p_prosumer_t,
                    "P_ch": p_ch_t,
                    "P_dc": p_dc_t,
                    "P_bess_net": p_bess_net_t,
                    "SOC": SOC_t,
                    "P_from_grid": p_from_t,
                    "P_to_grid": p_to_t,
                    "P_grid_net": p_grid_net_t,
                    "P_ch_grid": p_ch_from_grid_t,
                    "P_ch_pv": p_ch_from_pv_t,
                    "P_dc_to_load": p_dc_to_load_t,
                    "P_dc_to_grid": p_dc_to_grid_t,
                    "P_PV_to_grid": p_pv_to_grid_t,
                    "P_Load_from_grid": p_load_from_grid_t,
                    "P_PV_on_site": p_pv_on_site_t,
                    "P_load_met": p_load_met_t,
                    "Cycle": cycle_t,
                    "EP_epex": ep_epex_t,
                    "EP_buy": ep_buy_t,
                    "EP_sell": ep_sell_t
                })

        hourly_results_df = pd.DataFrame(hourly_results)

        annual_cycle = hourly_results_df["Cycle"].sum()
        annual_import = hourly_results_df["P_from_grid"].sum()
        annual_export = hourly_results_df["P_to_grid"].sum()

        annual_PV = df[df.columns[df.columns.str.contains("P_PV_8_MWh")][0]].sum()
        annual_demand = df["P_Load_MWh"].sum()
        annual_PV_used = hourly_results_df["P_PV_on_site"].sum()
        annual_load_met = hourly_results_df["P_load_met"].sum()
        sc_annual = (annual_PV_used / annual_PV) * 100 if annual_PV > 0 else 0
        ss_annual = (annual_load_met / annual_demand) * 100 if annual_demand > 0 else 0

        yearly_summary_df = pd.DataFrame([{
            "PV [MW]": P_pv,
            "Battery [MWh]": E_bess,
            "Inverter [MW]": P_inv,
            "Self-Consumption [%]": round(sc_annual, 2),
            "Self-Sufficiency [%]": round(ss_annual, 2),
            "Grid Import [MWh]": round(annual_import, 2),
            "Grid Export [MWh]": round(annual_export, 2),
            "Cycles": round(annual_cycle, 2),
        }])

        hourly_csv_path = os.path.join(state_output_dir, "csv", f"{state_short}_rule_based_hourly_results.csv")
        hourly_results_df.to_csv(hourly_csv_path, index=False)

        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            hourly_results_df.to_excel(writer, sheet_name="Hourly Results", index=False)
            yearly_summary_df.to_excel(writer, sheet_name="Yearly Summary", index=False)

        elapsed = time.time() - start_time
        logging.info(f"✅ Results for {state} written to {output_dir} in {elapsed/60:.2f} minutes")

    except Exception as e:
        logging.error(f"❌ Error in simulation for {state}: {e}", exc_info=True)
        print(f"❌ Error in simulation for {state}. See log file.")

logging.info("✅ Rule_based Simulations for all federal-states has finished.✅")

# === Aggregating Country-Wide Hourly Grid Data ===
logging.info("🔄 Aggregating country-wide hourly grid import/export (P_from_grid, P_to_grid)...")

# === Dynamic Results Mapping ===
results_file_map = {}
for state in config_df["federal_state"]:
    state_short = filename_map.get(state, state).replace(" ", "-")
    path = os.path.join(results_dir, f"FS_{state_short}_rule_based", "csv",
                        f"{state_short}_rule_based_hourly_results.csv")
    results_file_map[state_short] = path

state_dfs = {}
for state_name, filepath in results_file_map.items():
    if not os.path.exists(filepath):
        logging.warning(f"⚠️ File not found for {state_name}: {filepath}")
        continue

    df = pd.read_csv(filepath, parse_dates=["DateTime"])
    required = {"DateTime", "Day", "Hour", "P_from_grid", "P_to_grid", "P_grid_net", "P_ch", "P_dc", "P_bess_net"}
    if not required.issubset(df.columns):
        logging.warning(f"⚠️ Missing required columns in {filepath}, skipping {state_name}")
        continue

    df = df[["DateTime", "Day", "Hour", "P_from_grid", "P_to_grid", "P_grid_net", "P_ch", "P_dc", "P_bess_net"]].copy()
    df.rename(columns={
        "P_from_grid": f"P_from_grid_{state_name}",
        "P_to_grid": f"P_to_grid_{state_name}",
        "P_grid_net": f"P_grid_net_{state_name}",
        "P_ch": f"P_ch_{state_name}",
        "P_dc": f"P_dc_{state_name}",
        "P_bess_net": f"P_bess_net_{state_name}"
    }, inplace=True)

    state_dfs[state_name] = df

if not state_dfs:
    logging.warning("❌ No valid state results found; aggregation skipped.")
else:
    states_sorted = sorted(state_dfs.keys())
    merged_df = state_dfs[states_sorted[0]]
    for state in states_sorted[1:]:
        merged_df = pd.merge(
            merged_df, state_dfs[state], on=["DateTime", "Day", "Hour"], how="outer"
        )

    for state in states_sorted:
        merged_df[f"P_from_grid_{state}"] = merged_df[f"P_from_grid_{state}"].fillna(0)
        merged_df[f"P_to_grid_{state}"] = merged_df[f"P_to_grid_{state}"].fillna(0)
        merged_df[f"P_ch_{state}"] = merged_df[f"P_ch_{state}"].fillna(0)
        merged_df[f"P_dc_{state}"] = merged_df[f"P_dc_{state}"].fillna(0)
        merged_df[f"P_bess_net_{state}"] = merged_df[f"P_bess_net_{state}"].fillna(0)

    merged_df["P_from_grid_net"] = merged_df[[f"P_from_grid_{s}" for s in states_sorted]].sum(axis=1)
    merged_df["P_to_grid_net"] = merged_df[[f"P_to_grid_{s}" for s in states_sorted]].sum(axis=1)
    merged_df["P_grid_net"] = merged_df["P_from_grid_net"] - merged_df["P_to_grid_net"]

    merged_df["P_ch_net"] = merged_df[[f"P_ch_{s}" for s in states_sorted]].sum(axis=1)
    merged_df["P_dc_net"] = merged_df[[f"P_dc_{s}" for s in states_sorted]].sum(axis=1)
    merged_df["P_bess_net"] = merged_df[[f"P_bess_net_{s}" for s in states_sorted]].sum(axis=1)

    col_order = (
            ["DateTime", "Day", "Hour",
             "P_from_grid_net", "P_to_grid_net", "P_grid_net",
             "P_ch_net", "P_dc_net", "P_bess_net"] +
            [f"P_from_grid_{s}" for s in states_sorted] +
            [f"P_to_grid_{s}" for s in states_sorted] +
            [f"P_ch_{s}" for s in states_sorted] +
            [f"P_dc_{s}" for s in states_sorted] +
            [f"P_bess_net_{s}" for s in states_sorted]
    )
    merged_df = merged_df[col_order]

    csv_path = os.path.join(results_dir, "rule_based_Germany_aggregated_results_full.csv")
    excel_path = os.path.join(results_dir, "rule_based_Germany_aggregated_results_full.xlsx")

    merged_df.to_csv(csv_path, index=False)
    merged_df.to_excel(excel_path, index=False)

    logging.info(f"✅ Germany-wide rule_based hourly CSV and Excel created:\n📁 {csv_path}\n📁 {excel_path}")