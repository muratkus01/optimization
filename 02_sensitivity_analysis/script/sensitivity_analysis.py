# sensitivity.py

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import pandas as pd
import os
import gc
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import importlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s")

def simulate_config(args):
    row, data_path, output_path = args
    df = pd.read_csv(data_path, sep=";")
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
    df["Day"] = df["DateTime"].dt.dayofyear
    df["Hour"] = df["DateTime"].dt.hour

    # === Configuration ===
    P_pv = row["PV"]
    E_bess = row["Battery"]
    P_inv = row["Inverter"]
    config_name = f"PV{int(P_pv)}_BESS{int(E_bess)}_INV{int(P_inv)}"
    excel_path = f"{output_path}/sensitivity_summary_{config_name}.xlsx"
    logging.info(f"🔍 Running config: PV={P_pv}, BESS={E_bess}, INV={P_inv}")

    # === Scale PV Generation according to config ===
    df["P_PV"] = df["P_PV_1"] * P_pv

    # === Extract Inputs ===
    P_load = df["P_Load"].values
    P_PV = df["P_PV"].values
    P_prosumer = P_PV - P_load
    EP_epex = df["EP_epex"].values
    EP_buy = df["EP_buy"].values
    EP_sell = df["EP_sell"].values

    # === Parameters ===
    eta_batt = 0.95
    eta_inv = 0.97
    eta_c = eta_d = (eta_batt ** 0.5) * eta_inv
    SOC_max = E_bess * 0.95
    SOC_min = E_bess * 0.05
    SOC_0 = E_bess * 0.5
    DoD = 0.90
    P_ch_max = P_dc_max = P_inv
    C_LBTC = 0.05438366526
    C_grid = 0.1936996
    annual_fixed_costs = (5 + 4.18285 + 4.167) * 12
    daily_fixed_costs = annual_fixed_costs / 365
    CAPEX = (P_pv * 130 + E_bess * 350 + P_inv * 190) + 135 + 500 + 800 + 2000 + 2500

    # === Optimization Model ===
    T = len(df)
    model = LpProblem("Yearly_Optimization", LpMaximize)
    P_ch = LpVariable.dicts("P_ch", range(T), 0, P_ch_max)
    P_dc = LpVariable.dicts("P_dc", range(T), 0, P_dc_max)
    SOC = LpVariable.dicts("SOC", range(T), SOC_min, SOC_max)
    P_from_grid = LpVariable.dicts("P_from_grid", range(T), 0)
    P_to_grid = LpVariable.dicts("P_to_grid", range(T), 0)
    S_ch = LpVariable.dicts("S_ch", range(T), cat="Binary")
    S_dc = LpVariable.dicts("S_dc", range(T), cat="Binary")

    model += lpSum(
        EP_epex[t] * (P_dc[t] - P_ch[t])
        - C_grid * (P_from_grid[t] + P_to_grid[t])
        - C_LBTC * P_dc[t]
        for t in range(T)
    )

    for t in range(T):
        if t == 0:
            model += SOC[t] == SOC_0 + eta_c * P_ch[t] - P_dc[t] * (1 / eta_d)
        else:
            model += SOC[t] == SOC[t - 1] + eta_c * P_ch[t] - P_dc[t] * (1 / eta_d)
        model += S_ch[t] + S_dc[t] <= 1
        model += P_ch[t] <= P_ch_max * S_ch[t]
        model += P_dc[t] <= P_dc_max * S_dc[t]
        model += (P_from_grid[t] - P_to_grid[t]) == -P_prosumer[t] + P_ch[t] - P_dc[t]

    model.solve(PULP_CBC_CMD(msg=False))
    if model.status != 1:
        return None

    # === Extract & Aggregate Results ===
    hourly_results = []
    daily_summary = []

    for day in sorted(df["Day"].unique()):
        hourly_data = df[df["Day"] == day]
        indices = hourly_data.index.tolist()

        daily_P_from = daily_P_to = daily_P_dc = daily_P_ch = daily_cycle = 0
        daily_earning_battery_dc_to_grid = daily_earning_battery_pv_to_grid = 0
        daily_opportunity_cost_ch_from_pv = daily_saving_from_avoided_grid_import = 0
        daily_cost_ch_from_grid = daily_cost_load_from_grid = daily_earnings =0
        daily_energy_cost = daily_net_bill = daily_optimization_profit = 0
        daily_pv_used_on_site = daily_load_met_without_grid = daily_pv = daily_load = 0

        for t in indices:
            P_load_t = P_load[t]
            P_PV_t = P_PV[t]
            P_prosumer_t = P_prosumer[t]
            P_ch_t = P_ch[t].varValue
            P_dc_t = P_dc[t].varValue
            SOC_t = SOC[t].varValue
            P_from_t = P_from_grid[t].varValue
            P_to_t = P_to_grid[t].varValue

            cycle_t = P_dc_t / (E_bess * DoD)

            pv_surplus_t = max(0, P_PV_t - P_load_t)
            p_ch_from_pv_t = min(pv_surplus_t, P_ch_t)
            p_ch_from_grid_t = max(0, P_ch_t - p_ch_from_pv_t)

            p_dc_to_load_t = min(P_load_t - P_PV_t, P_dc_t) if P_load_t > P_PV_t else 0
            p_dc_to_grid_t = P_dc_t - p_dc_to_load_t

            p_pv_on_site_t = min(P_PV_t, P_load_t + P_ch_t)
            p_load_met_t = min(P_load_t, P_PV_t + P_dc_t)

            p_pv_to_grid_t = max(0, P_PV_t - p_pv_on_site_t)
            p_load_from_grid_t = max(0, P_load_t - p_load_met_t)

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
                    saving_from_avoided_grid_import_t + earning_battery_dc_to_grid_t +
                    earning_battery_pv_to_grid_t - opportunity_cost_ch_from_pv_t -
                    cost_ch_from_grid_t - cost_load_from_grid_t
            )

            self_consumption_t = (p_pv_on_site_t / P_PV_t) * 100 if P_PV_t > 0 else 0
            self_sufficiency_t = (p_load_met_t / P_load_t) * 100 if P_load_t > 0 else 0

            hourly_results.append({
                "DateTime": df["DateTime"].iloc[t],
                "Day": df["Day"].iloc[t],
                "Hour": df["Hour"].iloc[t],
                "P_Load": P_load_t,
                "P_PV": P_PV_t,
                "P_prosumer": P_prosumer_t,
                "P_ch": P_ch_t,
                "P_dc": P_dc_t,
                "SOC": SOC_t,
                "P_from_grid": P_from_t,
                "P_to_grid": P_to_t,
                "P_ch_grid": p_ch_from_grid_t,
                "P_ch_pv": p_ch_from_pv_t,
                "P_dc_to_load": p_dc_to_load_t,
                "P_dc_to_grid": p_dc_to_grid_t,
                "P_PV_to_grid": p_pv_to_grid_t,
                "P_Load_from_grid": p_load_from_grid_t,
                "P_PV_on_site": p_pv_on_site_t,
                "P_load_met": p_load_met_t,
                "Self_Consumption (%)": self_consumption_t,
                "Self_Sufficiency (%)": self_sufficiency_t,
                "Earning for Discharge to Grid": earning_battery_dc_to_grid_t,
                "Earning for PV Feed-in to Grid": earning_battery_pv_to_grid_t,
                "Opportunity Cost of Not PV Feed-in to Grid": opportunity_cost_ch_from_pv_t,
                "Savings from Avoided Grid Import Cost": saving_from_avoided_grid_import_t,
                "Cost for Covering Load from Grid": cost_load_from_grid_t,
                "Cost for Charging Battery from Grid": cost_ch_from_grid_t,
                "Earnings": earnings_t,
                "Optimization Profit": optimization_profit_t,
                "Net Bill": net_bill_t,
                "Cycle": cycle_t,
                "EP_epex": ep_epex_t,
                "EP_buy": ep_buy_t,
                "EP_sell": ep_sell_t
            })

            daily_P_from += P_from_t
            daily_P_to += P_to_t
            daily_P_dc += P_dc_t
            daily_P_ch += P_ch_t
            daily_cycle += cycle_t
            daily_pv_used_on_site += p_pv_on_site_t
            daily_load_met_without_grid += p_load_met_t
            daily_pv += P_PV_t
            daily_load += P_load_t

            daily_earning_battery_dc_to_grid += earning_battery_dc_to_grid_t
            daily_earning_battery_pv_to_grid += earning_battery_pv_to_grid_t
            daily_opportunity_cost_ch_from_pv += opportunity_cost_ch_from_pv_t
            daily_saving_from_avoided_grid_import += saving_from_avoided_grid_import_t
            daily_cost_ch_from_grid += cost_ch_from_grid_t
            daily_cost_load_from_grid += cost_load_from_grid_t
            daily_energy_cost += cost_ch_from_grid_t + cost_load_from_grid_t

            daily_earnings += earning_battery_dc_to_grid_t + earning_battery_pv_to_grid_t
            daily_net_bill += cost_ch_from_grid_t + cost_load_from_grid_t - (
                    earning_battery_dc_to_grid_t + earning_battery_pv_to_grid_t)
            daily_optimization_profit += optimization_profit_t

        self_consumption = (daily_pv_used_on_site / daily_pv) * 100 if daily_pv > 0 else 0
        self_sufficiency = (daily_load_met_without_grid / daily_load) * 100 if daily_load > 0 else 0
        daily_net_bill -= daily_fixed_costs
        daily_optimization_profit -= daily_fixed_costs

        daily_summary.append({
            "Date": hourly_data["DateTime"].iloc[0].strftime("%d.%m.%Y"),
            "Day": day,
            "P_from_grid": daily_P_from,
            "P_to_grid": daily_P_to,
            "P_prosumer": hourly_data["P_PV"].sum() - hourly_data["P_Load"].sum(),
            "P_dc": daily_P_dc,
            "P_ch": daily_P_ch,
            "D. Cycle": daily_cycle,
            "D. Earning for Discharge to Grid": daily_earning_battery_dc_to_grid,
            "D. Earning for PV Feed-in to Grid": daily_earning_battery_pv_to_grid,
            "D. Opportunity Cost of Not PV Feed-in to Grid ": daily_opportunity_cost_ch_from_pv,
            "D. Savings from Avoided Grid Import Cost": daily_saving_from_avoided_grid_import,
            "D. Cost of Covering Load from Grid": daily_cost_load_from_grid,
            "D. Cost of Charging Battery from Grid": daily_cost_ch_from_grid,
            "D. Earnings": daily_earnings,
            "D. Energy Cost": daily_energy_cost,
            "D. Net Bill": daily_net_bill,
            "D. Optimization Profit": daily_optimization_profit,
            "Self_Consumption (%)": self_consumption,
            "Self_Sufficiency (%)": self_sufficiency,
        })


    hourly_df = pd.DataFrame(hourly_results)
    daily_df = pd.DataFrame(daily_summary)

    annual_earning_battery_dc_to_grid = hourly_df["Earning for Discharge to Grid"].sum()
    annual_earning_battery_pv_to_grid = hourly_df["Earning for PV Feed-in to Grid"].sum()
    annual_opportunity_cost_ch_from_pv = hourly_df["Opportunity Cost of Not PV Feed-in to Grid"].sum()
    annual_saving_from_avoided_grid_import = hourly_df["Savings from Avoided Grid Import Cost"].sum()
    annual_cost_load_from_grid = hourly_df["Cost for Covering Load from Grid"].sum()
    annual_cost_ch_from_grid = hourly_df["Cost for Charging Battery from Grid"].sum()
    annual_earnings = hourly_df["Earnings"].sum()
    annual_energy_cost = hourly_df["Cost for Charging Battery from Grid"].sum() + hourly_df[
        "Cost for Covering Load from Grid"].sum()
    annual_net_bill = hourly_df["Net Bill"].sum() - annual_fixed_costs
    annual_optimization_profit = hourly_df["Optimization Profit"].sum() - annual_fixed_costs
    annual_cycle = hourly_df["Cycle"].sum()
    annual_import = hourly_df["P_from_grid"].sum()
    annual_export = hourly_df["P_to_grid"].sum()

    # === Compute Yearly Summary ===
    annual_PV = df["P_PV"].sum()
    annual_demand = df["P_Load"].sum()

    annual_PV_used = hourly_df["P_PV_on_site"].sum()
    annual_load_met = hourly_df["P_load_met"].sum()

    sc_annual = (annual_PV_used / annual_PV) * 100 if annual_PV > 0 else 0
    ss_annual = (annual_load_met / annual_demand) * 100 if annual_demand > 0 else 0

    logging.info(f"✅ Finished config : NetBill=:{round(annual_net_bill,2)}, OptProfit={round(annual_optimization_profit)}, PV={P_pv}, BESS={E_bess}, INV={P_inv}, SC={round(sc_annual,2)}, SS={round(ss_annual,2)}, Cycle={round(annual_cycle,2)}")    # Save results
    hourly_df.to_csv(f"{output_path}/sensitivity_hourly_{config_name}.csv", index=False)
    daily_df.to_csv(f"{output_path}/sensitivity_daily_{config_name}.csv", index=False)

    gc.collect()
    return {
        "PV [kW]": P_pv,
        "Battery [kWh]": E_bess,
        "Inverter [kW]": P_inv,
        "CAPEX [€]": CAPEX,
        "Self-Consumption [%]": round(sc_annual, 2),
        "Self-Sufficiency [%]": round(ss_annual, 2),
        "Earning for Discharge to Grid": round(annual_earning_battery_dc_to_grid, 2),
        "Earning for PV Feed-in to Grid": round(annual_earning_battery_pv_to_grid, 2),
        "Opportunity Cost of Not PV Feed-in to Grid": round(annual_opportunity_cost_ch_from_pv, 2),
        "Savings from Avoided Grid Import Cost": round(annual_saving_from_avoided_grid_import, 2),
        "Cost for Covering Load from Grid": round(annual_cost_load_from_grid, 2),
        "Cost for Charging Battery from Grid": round(annual_cost_ch_from_grid, 2),
        "Total Earnings": round(annual_earnings, 2),
        "Energy Cost": round(annual_energy_cost, 2),
        "Fixed Annual Cost [€]": round(annual_fixed_costs, 2),
        "Net Bill [€]": round(annual_net_bill, 2),
        "Optimization Profit": round(annual_optimization_profit, 2),
        "Cycles": round(annual_cycle, 2),
        "Grid Import [kWh]": round(annual_import, 2),
        "Grid Export [kWh]": round(annual_export, 2),
        "Opt. Profit / kWh Battery [€/kWh]": round(annual_optimization_profit / E_bess if E_bess else 0, 4),
        "Opt. Profit / kWp PV [€/kW]": round(annual_optimization_profit / P_pv if P_pv else 0, 4),
        "Opt. Profit / kW Inverter [€/kW]": round(annual_optimization_profit / P_inv if P_inv else 0 ,4),
        "Opt. Profit / CAPEX": round(annual_optimization_profit / CAPEX, 6),
        "Cycles / CAPEX [cycle/€]": round(annual_cycle / CAPEX, 6),
    }


if __name__ == "__main__":
    # === Path Configuration ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    # Input folders and paths
    data_dir = os.path.join(parent_dir, "data")
    config_path = os.path.join(data_dir, "config.csv")
    data_path = os.path.join(data_dir, "HSS_Bavaria_PV_1.csv")
    # Output folders and paths
    results_dir = os.path.join(parent_dir, "results")
    output_path = os.path.join(results_dir, "csv")

    os.makedirs(output_path, exist_ok=True)

    config_df = pd.read_csv(config_path)

    args_list = [(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), data_path, output_path)
                 for _, row in config_df.iterrows()]

    with Pool(processes=cpu_count()) as pool:
        yearly_results = pool.map(simulate_config, args_list)

    yearly_results_df = pd.DataFrame([res for res in yearly_results if res])

    # Print configuration statistics
    total_configs = len(config_df)
    valid_configs = len(yearly_results_df)
    print(f"Total configurations attempted: {total_configs}")
    print(f"Valid configurations (solved): {valid_configs}")
    print(f"Fraction valid: {valid_configs}/{total_configs} = {valid_configs / total_configs:.2%}")

    # Print min and max CAPEX values and their configurations
    if not yearly_results_df.empty:
        min_capex = yearly_results_df["CAPEX [€]"].min()
        max_capex = yearly_results_df["CAPEX [€]"].max()
        min_capex_row = yearly_results_df.loc[yearly_results_df["CAPEX [€]"].idxmin()]
        max_capex_row = yearly_results_df.loc[yearly_results_df["CAPEX [€]"].idxmax()]
        print(
            f"Lowest CAPEX: {min_capex} € (PV={min_capex_row['PV [kW]']}, Battery={min_capex_row['Battery [kWh]']}, Inverter={min_capex_row['Inverter [kW]']})")
        print(
            f"Highest CAPEX: {max_capex} € (PV={max_capex_row['PV [kW]']}, Battery={max_capex_row['Battery [kWh]']}, Inverter={max_capex_row['Inverter [kW]']})")
    else:
        print("No valid configurations to report CAPEX.")

    # Save summary results
    yearly_results_df.to_excel(os.path.join(results_dir, "sensitivity_yearly_summary.xlsx"), index=False)
    yearly_results_df.to_csv(os.path.join(output_path, "sensitivity_yearly_summary.csv"), index=False)

    print("✅ Yearly Summary Excel, Hourly and Daily CSVs are prepared.")

