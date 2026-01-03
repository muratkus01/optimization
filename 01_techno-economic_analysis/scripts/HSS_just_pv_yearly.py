import os
import pandas as pd
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,".." ,"data" , "2024_HSS_Baveria.csv")
output_dir = os.path.join(script_dir, "..", "results", "just_pv")
csv_dir = os.path.join(output_dir, "csv")
excel_path = os.path.join(output_dir, f"just_pv_hourly_results.xlsx")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(file_path, encoding="ISO-8859-1", sep=";")
df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
T = len(df)
hours_per_day = 24

# === Parameters ===
T = len(df) # Optimization horizon
E_bess = 0 # BESS capacity
P_pv = 8.0 # PV capacity
P_inv = 5.63 # Inverter Rated Power
eta_batt = 0.95 # Battery efficiency
eta_inv = 0.97 # Inverter efficiency
eta_c = eta_d =  eta_inv # Overall efficiency
SOC_max = E_bess * 0.95 # Maximum SOC
SOC_min = E_bess * 0.05 # Minimum SOC
SOC_0 = E_bess * 0.5 # Initial SOC
DoD = 0.90 # Depth of discharge
P_ch_max = P_dc_max = P_inv # Maximum charge/discharge power
E_max = 2 * (SOC_max - SOC_min) # Maximum energy throughput
SOC_0 = E_bess * 0.5
annual_fixed_costs = (4.18285 + 1.667 + 3.8568) * 12 # 116.48 €/year
daily_fixed_costs = annual_fixed_costs / 366
CAPEX = (P_pv * 130 + P_inv * 190) + 800 + 2000 + 2500

# === Input Data ===
EP_buy = df["EP_buy"].values
EP_sell = df["EP_sell"].values
EP_epex = df["EP_epex"].values
P_load = df["P_Load"].values
P_PV = df["PV_Bayern_8"].values
P_prosumer = P_PV - P_load

# === Initialization ===
SOC = [SOC_0]
hourly_results = []
daily_summary = []
annual_earning_battery_dc_to_grid = annual_earning_battery_pv_to_grid = 0
annual_opportunity_cost_ch_from_pv = annual_cost_load_from_grid = annual_cost_ch_from_grid = 0
annual_saving_from_avoided_grid_import = annual_earnings = 0
annual_energy_cost = annual_net_bill = annual_optimization_profit = 0
annual_import = annual_export = annual_cycle = 0

# === Battery Dispatch Loop ===
for t in range(T):
    if t % hours_per_day == 0:
        cumulative_energy = 0
        daily_P_from = daily_P_to = daily_P_dc = daily_P_ch = daily_cycle = 0
        daily_earning_battery_dc_to_grid = daily_earning_battery_pv_to_grid = 0
        daily_opportunity_cost_ch_from_pv = daily_saving_from_avoided_grid_import = 0
        daily_cost_ch_from_grid = daily_cost_load_from_grid = daily_earnings = 0
        daily_energy_cost = daily_net_bill = daily_optimization_profit = 0
        daily_pv_used_on_site = daily_load_met_without_grid = daily_pv = daily_load = 0
        daily_P_load = []
        daily_P_PV = []
        daily_P_prosumer = []

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
    p_prosumer_t = P_PV[t] - P_load[t]
    ep_buy_t = EP_buy[t]
    ep_sell_t = EP_sell[t]
    ep_epex_t = EP_epex[t]

    bill_t = ep_buy_t * p_from_t  # Total electricity cost for this hour based on grid import and retail price
    cycle_t = 0 #p_dc_t / (E_bess * DoD)  # Battery cycle depth used in this hour, normalized to usable capacity

    p_pv_surplus_t = max(0,
                     p_PV_t - p_load_t)  # PV surplus after directly covering the household load (available for charging or export)
    p_ch_from_pv_t = min(p_pv_surplus_t, p_ch_t)  # Portion of PV surplus actually used for battery charging
    p_ch_from_grid_t = max(0, p_ch_t - p_ch_from_pv_t)  # Remaining battery charge demand covered by the grid

    p_dc_to_load_t = min(p_load_t - p_PV_t,
                       p_dc_t) if p_load_t > p_PV_t else 0  # Battery discharge used to meet unmet household load
    p_dc_to_grid_t = p_dc_t - p_dc_to_load_t  # Battery discharge exported to the grid after covering local load

    p_pv_on_site_t = min(p_PV_t, p_load_t + p_ch_t)  # PV energy consumed on-site, either for load or for charging the battery
    p_load_met_t = min(p_load_t, p_PV_t + p_dc_t)  # Total load met by PV and battery discharge (i.e., not imported from the grid)

    p_pv_to_grid_t = max(0, p_PV_t - p_pv_on_site_t)  # PV energy exported to the grid (not used on-site)
    p_load_from_grid_t = max(0, p_load_t - p_load_met_t)  # Unmet load that must be covered by grid import

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

    daily_P_from += p_from_t
    daily_P_to += p_to_t
    daily_P_dc += p_dc_t
    daily_P_ch += p_ch_t
    daily_cycle += cycle_t
    daily_pv_used_on_site += p_pv_on_site_t
    daily_load_met_without_grid += p_load_met_t
    daily_pv += p_PV_t
    daily_load += p_load_t
    daily_P_prosumer += p_prosumer_t

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

    hourly_results.append({
        "DateTime": df.loc[t, "DateTime"].strftime("%d.%m.%Y %H:%M"),
        "Day": t // hours_per_day + 1,
        "Hour": t % hours_per_day,
        "P_Load": p_load_t,
        "P_PV": p_PV_t,
        "P_prosumer": p_prosumer_t,
        "P_ch": p_ch_t,
        "P_dc": p_dc_t,
        "SOC": SOC_t,
        "P_from_grid": p_from_t,
        "P_to_grid": p_to_t,
        "P_net_grid": round(p_from_t - p_to_t, 6),
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

    if (t + 1) % hours_per_day == 0:
        day = t // hours_per_day
        self_consumption = (daily_pv_used_on_site / daily_pv) * 100 if daily_pv > 0 else 0
        self_sufficiency = (daily_load_met_without_grid / daily_load) * 100 if daily_load > 0 else 0
        daily_net_bill -= daily_fixed_costs
        daily_optimization_profit -= daily_fixed_costs

        daily_summary.append({
            "Date": df["DateTime"].iloc[t - hours_per_day + 1].strftime("%d.%m.%Y"),
            "Day": day + 1,
            "P_from_grid": daily_P_from,
            "P_to_grid": daily_P_to,
            "P_prosumer": daily_P_prosumer,
            "P_dc": daily_P_dc,
            "P_ch": daily_P_ch,
            "D. Cycle": daily_cycle,
            "D. Earning for Discharge to Grid": daily_earning_battery_dc_to_grid,
            "D. Earning for PV Feed-in to Grid": daily_earning_battery_pv_to_grid,
            "D. Opportunity Cost for Not PV Feed-in to Grid ": daily_opportunity_cost_ch_from_pv,
            "D. Saving for Avoided Grid Import Cost": daily_saving_from_avoided_grid_import,
            "D. Cost for Covering Load from Grid": daily_cost_load_from_grid,
            "D. Cost for Charging Battery from Grid": daily_cost_ch_from_grid,
            "D. Total Earnings": daily_earnings,
            "D. Energy Cost": daily_energy_cost,
            "D. Net Bill": daily_net_bill,
            "D. Optimization Profit": daily_optimization_profit,
            "Self_Consumption (%)": self_consumption,
            "Self_Sufficiency (%)": self_sufficiency,
        })

# === Save Outputs ===
hourly_results_df = pd.DataFrame(hourly_results)
daily_df = pd.DataFrame(daily_summary)

# === Compute Yearly Summary ===
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
annual_PV = df["PV_Bayern_8"].sum()
annual_demand = df["P_Load"].sum()
annual_PV_used = hourly_results_df["P_PV_on_site"].sum()
annual_load_met = hourly_results_df["P_load_met"].sum()


# Energy-based self-consumption and self-sufficiency
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


hourly_csv_path = os.path.join(csv_dir, f"just_pv_hourly_results.csv")
daily_csv_path = os.path.join(csv_dir, f"just_pv_daily_summary.csv")
hourly_results_df.to_csv(hourly_csv_path, index=False)
daily_df.to_csv(daily_csv_path, index=False)

with pd.ExcelWriter(excel_path) as writer:
    hourly_results_df.to_excel(writer, sheet_name="Hourly Results", index=False)
    daily_df.to_excel(writer, sheet_name="Daily Summary", index=False)
    yearly_summary_df.to_excel(writer, sheet_name="Yearly Summary", index=False)

# === Logging Summary ===
logging.info(f"✅ Yearly optimization for Just PV is completed.")
logging.info(f"📁 Hourly results saved to: {hourly_csv_path}")
logging.info(f"📁 Daily summary saved to: {daily_csv_path}")
logging.info(f"📊 Excel results saved to: {excel_path}")
logging.info(f"📈 Self-Sufficiency: %{round(ss_annual, 2)} | Self-Consumption: %{round(sc_annual, 2)}")
logging.info(f"📈 A. Earning for Discharge to Grid: €{round(hourly_results_df['Earning for Discharge to Grid'].sum(), 2)} | "
             f"A. Earning for PV Feed-in to Grid: €{round(hourly_results_df['Earning for PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Saving for Avoided Grid Import: €{round(hourly_results_df['Saving for Avoided Grid Import'].sum(), 2)}")
logging.info(f"📈 A. Opportunity Cost for Not PV Feed-in to Grid: €{round(hourly_results_df['Opportunity Cost for Not PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Cost for Covering Load from Grid: €{round(hourly_results_df['Cost for Covering Load from Grid'].sum(), 2)} | "
             f"A. Cost for Charging Battery from Grid: €{round(hourly_results_df['Cost for Charging Battery from Grid'].sum(), 2)}")
logging.info(f"📈 A. Cycle: {round(hourly_results_df['Cycle'].sum(), 2)} | "
             f"Annual Grid Import [kWh]: {round(annual_import, 2)} | Annual Grid Export [kWh]: {round(annual_export, 2)}")
logging.info(f"📈 A. Energy Cost: €{round(hourly_results_df['Cost for Covering Load from Grid'].sum() + hourly_results_df['Cost for Charging Battery from Grid'].sum(), 2)} | "
             f"A. Fixed Energy Costs [€]: €{round(annual_fixed_costs, 2)} | "
             f"A. Earnings: €{round(hourly_results_df['Earning for Discharge to Grid'].sum() + hourly_results_df['Earning for PV Feed-in to Grid'].sum(), 2)} | "
             f"A. Net Bill: €{round(hourly_results_df['Net Bill'].sum() - annual_fixed_costs, 2)} | "
             f"A. Optimization Profit: €{round(hourly_results_df['Optimization Profit'].sum() - annual_fixed_costs, 2)}")