from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import pandas as pd
import os
import logging


# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === Load Data ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"..", "data", "2024_HSS_Baveria.csv")
df = pd.read_csv(file_path, sep=";", encoding="ISO-8859-1")
df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
df["Day"] = df["DateTime"].dt.dayofyear
df["Hour"] = df["DateTime"].dt.hour

# Prepare file and folder
output_dir = os.path.join(script_dir, "..", "results", "optimized_pv_bess")
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, "optimized_pv_bess_hourly_results.xlsx")

os.makedirs(output_dir + "/csv", exist_ok=True)

# === Parameters ===
T = len(df) # Optimization horizon
E_bess = 9.37 # BESS capacity
P_pv = 8.0 # PV capacity
P_inv = 5.63 # Inverter Rated Power
eta_batt = 0.95 # Battery efficiency
eta_inv = 0.97 # Inverter efficiency
eta_c = eta_d = (eta_batt ** 0.5) * eta_inv # Overall efficiency
SOC_max = E_bess * 0.95 # Maximum SOC
SOC_min = E_bess * 0.05 # Minimum SOC
SOC_0 = E_bess * 0.5 # Initial SOC
DoD = 0.90 # Depth of discharge
P_ch_max = P_dc_max = P_inv # Maximum charge/discharge power
C_MBCC = 0.05438366526 # Marginal battery cycling cost
C_grid = 0.1936996 # Grid cost
E_max = 2 * (SOC_max - SOC_min) # Maximum energy throughput
annual_fixed_costs = (5 + 4.18285 + 4.167) * 12 # 11.315 €/month * 12 ~ 160.194 €/year
daily_fixed_costs = annual_fixed_costs / 366 # 0.377 €/day
CAPEX = (P_pv * 130 + E_bess * 350 + P_inv * 190) + 135 + 500 + 800 + 2000 + 2500

# === Extract Inputs ===
P_load = df["P_Load"].values
P_PV = df["PV_Bayern_8"].values
P_prosumer = P_PV - P_load
EP_epex = df["EP_epex"].values

annual_earning_battery_dc_to_grid = annual_earning_battery_pv_to_grid = 0
annual_opportunity_cost_ch_from_pv = annual_cost_load_from_grid = annual_cost_ch_from_grid = 0
annual_saving_from_avoided_grid_import = annual_earnings = 0
annual_energy_cost = annual_net_bill = annual_optimization_profit = 0
annual_import = annual_export = annual_cycle = 0

# === Optimization Model ===
model = LpProblem("Yearly_Optimization", LpMaximize)
P_ch = LpVariable.dicts("P_ch", range(T), 0, P_ch_max)
P_dc = LpVariable.dicts("P_dc", range(T), 0, P_dc_max)
SOC = LpVariable.dicts("SOC", range(T), SOC_min, SOC_max)
P_from_grid = LpVariable.dicts("P_from_grid", range(T), 0)
P_to_grid = LpVariable.dicts("P_to_grid", range(T), 0)
S_ch = LpVariable.dicts("S_ch", range(T), cat="Binary")
S_dc = LpVariable.dicts("S_dc", range(T), cat="Binary")

# === Objective Function ===
model += lpSum(
    EP_epex[t] * (P_dc[t] - P_ch[t])
    - C_grid * (P_from_grid[t] + P_to_grid[t])
    - C_MBCC * P_dc[t]
    for t in range(T)
)

# === Constraints ===
for t in range(T):
    if t == 0:
        model += SOC[t] == SOC_0 + eta_c * P_ch[t] - P_dc[t] * (1 / eta_d)
    else:
        model += SOC[t] == SOC[t - 1] + eta_c * P_ch[t] - P_dc[t] * (1 / eta_d)

    model += S_ch[t] + S_dc[t] <= 1
    model += P_ch[t] <= P_ch_max * S_ch[t]
    model += P_dc[t] <= P_dc_max * S_dc[t]
    model += (P_from_grid[t] - P_to_grid[t]) == -P_prosumer[t] + P_ch[t] - P_dc[t]

for day_start in range(0, T, 24): # total discharged energy per day cannot exceed E_max
    day_end = min(day_start + 24, T)
    model += lpSum(P_dc[t] for t in range(day_start, day_end)) <= E_max

# === Solve ===
model.solve(PULP_CBC_CMD(msg=False))

# === Results ===
hourly_results = []
daily_summary = []

for day in sorted(df["Day"].unique()):
    hourly_df = df[df["Day"] == day]
    indices =hourly_df.index.tolist()

    daily_P_from = daily_P_to = daily_P_dc = daily_P_ch = daily_cycle = 0
    daily_earning_battery_dc_to_grid = daily_earning_battery_pv_to_grid = 0
    daily_opportunity_cost_ch_from_pv = daily_saving_from_avoided_grid_import = 0
    daily_cost_ch_from_grid = daily_cost_load_from_grid = daily_earnings = 0
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
        p_load_from_PV_t = min(P_PV_t, P_load_t) # Portion of load directly met by PV
        p_load_met_t = p_dc_to_load_t + p_load_from_PV_t

        p_pv_to_grid_t = max(0, P_PV_t - p_pv_on_site_t)
        p_load_from_grid_t = max(0, P_load_t - p_load_met_t)

        ep_epex_t = df["EP_epex"].iloc[t]
        ep_buy_t = df["EP_buy"].iloc[t]
        ep_sell_t = df["EP_sell"].iloc[t]

        saving_from_avoided_grid_import_t = ep_buy_t * p_load_met_t
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
        self_consumption_t = (p_pv_on_site_t / P_PV_t) * 100 if P_PV_t > 0 else 0
        self_sufficiency_t = (p_load_met_t / P_load_t) * 100 if P_load_t > 0 else 0

        hourly_results.append({
                "DateTime": df["DateTime"].iloc[t],
                "Day": df["Day"].iloc[t],
                "Hour": df["Hour"].iloc[t],
                "P_Load": round(P_load_t, 6),
                "P_PV": round(P_PV_t, 6),
                "P_prosumer": round(P_prosumer_t, 6),
                "P_ch": round(P_ch_t, 6),
                "P_dc": round(P_dc_t, 6),
                "SOC": round(SOC_t, 6),
                "P_from_grid": round(P_from_t, 6),
                "P_to_grid": round(P_to_t, 6),
                "P_net_grid": round(P_from_t - P_to_t, 6),
                "P_ch_from_grid": round(p_ch_from_grid_t, 6),
                "P_ch_pv": round(p_ch_from_pv_t, 6),
                "P_dc_to_load": round(p_dc_to_load_t, 6),
                "P_dc_to_grid": round(p_dc_to_grid_t, 6),
                "P_PV_to_grid": round(p_pv_to_grid_t, 6),
                "P_Load_from_grid": round(p_load_from_grid_t, 6),
                "P_Load_from_PV": round(p_load_from_PV_t, 6),
                "P_PV_on_site": round(p_pv_on_site_t, 6),
                "P_load_met": round(p_load_met_t, 6),
                "Self_Consumption (%)": self_consumption_t,
                "Self_Sufficiency (%)": self_sufficiency_t,
                "Earning for Discharge to Grid": earning_battery_dc_to_grid_t,
                "Earning for PV Feed-in to Grid": earning_battery_pv_to_grid_t,
                "Opportunity Cost for Not PV Feed-in to Grid": opportunity_cost_ch_from_pv_t,
                "Saving for Avoided Grid Import": saving_from_avoided_grid_import_t,
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
        "Date":hourly_df["DateTime"].iloc[0].strftime("%d.%m.%Y"),
        "Day": day,
        "P_from_grid": daily_P_from,
        "P_to_grid": daily_P_to,
        "P_prosumer": hourly_df["PV_Bayern_8"].sum() -hourly_df["P_Load"].sum(),
        "P_dc": daily_P_dc,
        "P_ch": daily_P_ch,
        "D. Cycle": daily_cycle,
        "D. Earning for Discharge to Grid": daily_earning_battery_dc_to_grid,
        "D. Earning for PV Feed-in to Grid": daily_earning_battery_pv_to_grid,
        "D. Opportunity Cost for Not PV Feed-in to Grid ": daily_opportunity_cost_ch_from_pv,
        "D. Saving for Avoided Grid Import": daily_saving_from_avoided_grid_import,
        "D. Cost for Covering Load from Grid": daily_cost_load_from_grid,
        "D. Cost for Charging Battery from Grid": daily_cost_ch_from_grid,
        "D. Total Earnings": daily_earnings,
        "D. Energy Cost": daily_energy_cost,
        "D. Net Bill": daily_net_bill,
        "D. Optimization Profit": daily_optimization_profit,
        "Self_Consumption (%)": self_consumption,
        "Self_Sufficiency (%)": self_sufficiency,
    })

# Convert lists to DataFrames
hourly_results_df = pd.DataFrame(hourly_results)
daily_summary_df = pd.DataFrame(daily_summary)

annual_earning_battery_dc_to_grid = hourly_results_df["Earning for Discharge to Grid"].sum()
annual_earning_battery_pv_to_grid = hourly_results_df["Earning for PV Feed-in to Grid"].sum()
annual_opportunity_cost_ch_from_pv = hourly_results_df["Opportunity Cost for Not PV Feed-in to Grid"].sum()
annual_saving_from_avoided_grid_import = hourly_results_df["Saving for Avoided Grid Import"].sum()
annual_cost_load_from_grid = hourly_results_df["Cost for Covering Load from Grid"].sum()
annual_cost_ch_from_grid = hourly_results_df["Cost for Charging Battery from Grid"].sum()
annual_earnings = hourly_results_df["Earnings"].sum()
annual_energy_cost = hourly_results_df["Cost for Charging Battery from Grid"].sum() + hourly_results_df[
    "Cost for Covering Load from Grid"].sum()
annual_net_bill = hourly_results_df["Net Bill"].sum() - annual_fixed_costs
annual_optimization_profit = hourly_results_df["Optimization Profit"].sum() - annual_fixed_costs
annual_cycle = hourly_results_df["Cycle"].sum()
annual_import = hourly_results_df["P_from_grid"].sum()
annual_export = hourly_results_df["P_to_grid"].sum()

# === Compute Yearly Summary ===
annual_PV = df["PV_Bayern_8"].sum()
annual_demand = df["P_Load"].sum()

annual_PV_used = hourly_results_df["P_PV_on_site"].sum()
annual_load_met = hourly_results_df["P_load_met"].sum()


sc_annual = (annual_PV_used / annual_PV) * 100 if annual_PV > 0 else 0
ss_annual = (annual_load_met / annual_demand) * 100 if annual_demand > 0 else 0

yearly_summary_df = pd.DataFrame([{
    "PV [kW]": P_pv,
    "Battery [kWh]": E_bess,
    "Inverter [kW]": P_inv,
    "CAPEX [€]": CAPEX,
    "Self-Consumption [%]": round(sc_annual, 2),
    "Self-Sufficiency [%]": round(ss_annual, 2),
    "Earning for Discharge to Grid": round(annual_earning_battery_dc_to_grid, 2),
    "Earning for PV Feed-in to Grid": round(annual_earning_battery_pv_to_grid, 2),
    "Opportunity Cost for Not PV Feed-in to Grid ": round(annual_opportunity_cost_ch_from_pv, 2),
    "Saving for Avoided Grid Import": round(annual_saving_from_avoided_grid_import, 2),
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
    "Opt. Profit / kW Inverter [€/kW]": round(annual_optimization_profit / P_inv if P_inv else 0, 4),
    "Opt. Profit / CAPEX": round(annual_optimization_profit / CAPEX, 6),
    "Cycles / CAPEX [cycle/€]": round(annual_cycle / CAPEX, 6),
}])

# Save results to Excel
hourly_csv_path = f"{output_dir}/csv/optimized_pv_bess_hourly_results.csv"
daily_csv_path = f"{output_dir}/csv/optimized_pv_bess_daily_summary.csv"
hourly_results_df.to_csv(f"{output_dir}/csv/optimized_pv_bess_hourly_results.csv", index=False)
daily_summary_df.to_csv(f"{output_dir}/csv/optimized_pv_bess_daily_summary.csv", index=False)


with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    hourly_results_df.to_excel(writer, sheet_name="Hourly Results", index=False)
    daily_summary_df.to_excel(writer, sheet_name="Daily Summary", index=False)
    yearly_summary_df.to_excel(writer, sheet_name="Yearly Summary", index=False)

# === Logging Summary ===
logging.info(f"✅ Yearly optimized optimization for PV+BESS completed.")
logging.info(f"📁 Hourly results saved to: {hourly_csv_path}")
logging.info(f"📁 Daily summary saved to: {daily_csv_path}")
logging.info(f"📊 Excel results saved to: {excel_path}")
logging.info(f"📈 Self-Sufficiency: %{round(ss_annual, 2)} | Self-Consumption: %{round(sc_annual, 2)}")
logging.info(f"📈 A. Earning for Discharge to Grid: €{round(hourly_results_df['Earning for Discharge to Grid'].sum(), 2)} | "
             f"A. Earning for PV Feed-in to Grid: €{round(hourly_results_df['Earning for PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Saving for Avoided Grid Cost: €{round(hourly_results_df['Saving for Avoided Grid Import'].sum(), 2)}")
logging.info(f"📈 A. Opportunity Cost for Not PV Feed-in to Grid: €{round(hourly_results_df['Opportunity Cost for Not PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Cost for Covering Load from Grid: €{round(hourly_results_df['Cost for Covering Load from Grid'].sum(), 2)} | "
             f"A. Cost for Charging Battery from Grid: €{round(hourly_results_df['Cost for Charging Battery from Grid'].sum(), 2)}")
logging.info(f"📈 A. Cycle: {round(hourly_results_df['Cycle'].sum(), 2)} | "
             f"Annual Grid Import [kWh]: {round(annual_import, 2)} | Annual Grid Export [kWh]: {round(annual_export, 2)}")
logging.info(f"📈 A. Energy Cost: €{round(hourly_results_df['Cost for Covering Load from Grid'].sum() + hourly_results_df['Cost for Charging Battery from Grid'].sum(), 2)} | "
             f"A. Fixed Energy Costs [€]: €{round(annual_fixed_costs, 2)} | "
             f"A. Earnings: €{round(hourly_results_df['Earning for Discharge to Grid'].sum() + hourly_results_df['Earning for PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Net Bill: €{round(hourly_results_df['Net Bill'].sum() - annual_fixed_costs, 2)} | "
             f"A. Optimization Profit: €{round(hourly_results_df['Optimization Profit'].sum(), 2)}")