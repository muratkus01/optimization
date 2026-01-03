import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from matplotlib.colors import PowerNorm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === Path Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
base_results_dir = os.path.join(os.path.dirname(script_dir), "results")
output_dir = os.path.join(base_results_dir, "combined_visualizations")
visualization_dir = output_dir
visualization_dir_combined_flow_profit = os.path.join(output_dir, "combined_flow_profit")
csv_dir = os.path.join(output_dir, "csv")

os.makedirs(visualization_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(visualization_dir_combined_flow_profit, exist_ok=True)

# === Load Data ===
optimized = os.path.join(base_results_dir, "optimized_pv_bess", "csv", "optimized_pv_bess_hourly_results.csv")
rule_based = os.path.join(base_results_dir, "rule_based_pv_bess", "csv", "rule_based_pv_bess_hourly_results.csv")
just_pv = os.path.join(base_results_dir, "just_pv", "csv", "just_pv_hourly_results.csv")

optimized_daily = os.path.join(base_results_dir, "optimized_pv_bess", "csv", "optimized_pv_bess_daily_summary.csv")
rule_based_daily = os.path.join(base_results_dir, "rule_based_pv_bess", "csv", "rule_based_pv_bess_daily_summary.csv")
just_pv_daily = os.path.join(base_results_dir, "just_pv", "csv", "just_pv_daily_summary.csv")

optimized_df = pd.read_csv(optimized, parse_dates=['DateTime'], dayfirst=True)
rule_based_df = pd.read_csv(rule_based, parse_dates=['DateTime'], dayfirst=True)
just_pv_df = pd.read_csv(just_pv, parse_dates=['DateTime'], dayfirst=True)

# === DAILY STACKED COMPONENTS (sum to yearly totals) — 3 base-case scenarios ===

# ===================== 1) Annual inputs (your numbers) =====================
rows = [
    ("Optimized BESS with PV", 178.51, 309.62, 887.43, 46.34, 1.71, 160.20, 128.30, 1039.02, 11324.2, 0.091752,  279.89),
    ("Rule-Based BESS with PV", 0.00, 278.33, 463.65, 46.85, 0.00, 146.48, 159.59,  389.06, 10824.2, 0.035944,   85.00),
    ("PV Without BESS",         0.00, 437.92,   0.00, 510.51, 0.00, 116.48,   0.00, -189.07,  7409.7,-0.025516, -189.07),
]
cols = ["Scenario",
        "Earning for Discharge to Grid",
        "Earning for PV Feed-in to Grid",
        "Savings from Avoided Grid Import",
        "Cost for Covering Load from Grid",
        "Cost for Charging Battery from Grid",
        "Annual Fixed Cost",
        "Opportunity Cost of Not PV Feed-in to Grid",
        "Optimization Profit", "CAPEX", "Opt. Profit / CAPEX", "Net Bill"]
df = pd.DataFrame(rows, columns=cols)

# ===================== 2) Styling / palettes =====================
colors = {
    "Earning for Discharge to Grid":               "#1f78b4",
    "Earning for PV Feed-in to Grid":              "#a6cee3",
    "Savings from Avoided Grid Import":            "#33a02c",
    "Cost for Covering Load from Grid":            "#fb9a99",
    "Cost for Charging Battery from Grid":         "#e31a1c",
    "Annual Fixed Cost":                           "#ff7f00",
    "Opportunity Cost of Not PV Feed-in to Grid":  "#6a3d9a",
}
pos_comps = ["Earning for Discharge to Grid",
             "Earning for PV Feed-in to Grid",
             "Savings from Avoided Grid Import"]
neg_comps = ["Cost for Covering Load from Grid",
             "Cost for Charging Battery from Grid",
             "Annual Fixed Cost",
             "Opportunity Cost of Not PV Feed-in to Grid"]

# Enforce sign conventions (revenues >= 0, costs <= 0)
for c in pos_comps:
    df[c] = df[c].abs()
for c in neg_comps:
    df[c] = -df[c].abs()

# ===================== 3) Plot =====================
fig, ax = plt.subplots(figsize=(10.5, 6.5))
x = np.arange(len(df))
bar_width = 0.62

# Diverging stacking (two baselines)
pos_bottom = np.zeros(len(df))
neg_bottom = np.zeros(len(df))

# stack positives upward
for c in pos_comps:
    ax.bar(x, df[c], bar_width, bottom=pos_bottom, color=colors[c], label=c)
    pos_bottom += df[c].values

# stack negatives downward
for c in neg_comps:
    ax.bar(x, df[c], bar_width, bottom=neg_bottom, color=colors[c], label=c)
    neg_bottom += df[c].values

# ===================== 4) Annotations =====================
def euro_int(v):  # "1 234 €"
    return f"{v:,.0f} €".replace(",", " ")

total_span = (pos_bottom.max() - neg_bottom.min())
top_buffer = max(30, total_span * 0.05)

for i, row in df.iterrows():
    top = pos_bottom[i]
    mid = (top + neg_bottom[i]) / 2.0

    # (A) Optimization Profit on top
    ax.text(i, top + 12,
            f"Opt. Profit of {euro_int(row['Optimization Profit'])}",
            ha='center', va='bottom', fontsize=10,
            color='darkgreen', fontweight='bold')

    # (B) Net Bill (left, vertical) — **bold**
    ax.text(i - bar_width / 2 - 0.02, mid,
            f"Net Bill of {row['Net Bill']:,.0f} €".replace(",", " "),
            ha='right', va='center', fontsize=9, fontweight='bold',
            color='dimgray', rotation=90)

    # (C) Opt. Profit/CAPEX (right, vertical) — *italic*
    ax.text(i + bar_width / 2 + 0.02, mid,
            f"ROCE of {row['Opt. Profit / CAPEX']:.3f} €/€",
            ha='left', va='center', fontsize=9, fontstyle='italic',
            color='teal', rotation=90)

# (D) Inside-segment labels if |value| > 50 €
value_threshold = 50  # €
for i, row in df.iterrows():
    y_pos = 0.0
    y_neg = 0.0

    # positives
    for c in pos_comps:
        v = row[c]
        if v != 0:
            y = y_pos + v/2.0
            if abs(v) > value_threshold:
                r,g,b = mcolors.to_rgb(colors[c])
                bright = 0.299*r + 0.587*g + 0.114*b
                txt = f"{int(round(v)):d} €".replace(",", " ")
                ax.text(i, y, txt, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color=('white' if bright < 0.5 else 'black'))
            y_pos += v

    # negatives
    for c in neg_comps:
        v = row[c]  # negative
        if v != 0:
            y = y_neg + v/2.0
            if abs(v) > value_threshold:
                r,g,b = mcolors.to_rgb(colors[c])
                bright = 0.299*r + 0.587*g + 0.114*b
                txt = f"–{int(round(abs(v))):d} €".replace(",", " ")
                ax.text(i, y, txt, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color=('white' if bright < 0.5 else 'black'))
            y_neg += v

# ===================== 5) Axes / legend =====================
# Custom x-tick labels with CAPEX in parentheses
xtick_labels = []
for _, row in df.iterrows():
    scenario_bold = row['Scenario'].replace(' ', r'\ ')
    label = rf"$\bf{{{scenario_bold}}}$" + f"\n(CAPEX of {row['CAPEX']:,.0f} €)".replace(",", " ")
    xtick_labels.append(label)

ax.set_xticks(x)
ax.set_xticklabels(xtick_labels, rotation=0)

ax.set_title("Annual Optimization Profit and Net Bill Components Across All Scenarios in 2024", fontsize=12)
ax.set_ylabel("Annual Value [€]", fontsize=10, color='darkgreen', fontweight='bold')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", " ")))

# Legend (deduplicated + bottom)
handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(),
          loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)

# headroom so labels don't clip
ax.set_ylim(min(neg_bottom) - top_buffer*0.4, max(pos_bottom) + top_buffer)

plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "annual_netbill_base_cases.png.png"), dpi=300, bbox_inches='tight')

logging.info("✅ Annual Net Bill components (with CAPEX under labels) saved.")

# === 1. Setup Paths ===
base_results_dir = os.path.join(os.path.dirname(script_dir), "results")
daily_opt_path = os.path.join(base_results_dir, "optimized_pv_bess", "csv", "optimized_pv_bess_daily_summary.csv")
daily_rb_path = os.path.join(base_results_dir, "rule_based_pv_bess", "csv", "rule_based_pv_bess_daily_summary.csv")
daily_jpv_path = os.path.join(base_results_dir, "just_pv", "csv", "just_pv_daily_summary.csv")

# === Files and Config ===
files = {
    "(a) Optimized BESS with PV": daily_opt_path,
    "(b) Rule-Based BESS with PV": daily_rb_path,
    "(c) PV Without BESS": daily_jpv_path
}

colors = {
    "(a) Optimized BESS with PV": "#1f77b4",
    "(b) Rule-Based BESS with PV": "#ff7f0e",
    "(c) PV Without BESS": "#2ca02c"
}


# === Plotting Function ===
def plot_metric(metric_column, title, filename):
    fig, ax = plt.subplots(figsize=(16, 6))

    all_dates = []

    for label, file_path in files.items():
        df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        all_dates.extend(df["Date"].tolist())

        ax.plot(df["Date"], df[metric_column].rolling(window=7).mean(), label=f"{label}",
                color=colors[label])

    # Define xticks
    min_date = min(all_dates)
    max_date = max(all_dates)
    xticks = pd.date_range(start=min_date, end=max_date, freq="14D")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=14)

    ax.legend(loc="lower left", fontsize=15, framealpha=0.1, facecolor='white')

    ax.set_ylabel(metric_column, fontsize=14)
    ax.set_xlabel("Week Start Date", fontsize=14)
    ax.set_title(f"{title} with 7-Day Moving Average Across All Scenarios in 2024", fontsize=18)
    month_starts = list(pd.date_range(start=min_date, end=max_date, freq='MS'))
    month_starts.append(pd.Timestamp("2025-01-01"))

    for month_start in month_starts:
        ax.axvline(month_start, color='lightgrey', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.grid(axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, filename), dpi=300)
    plt.close()


# === Plot Both ===
plot_metric("Self_Sufficiency (%)", "Daily Self-Sufficiency (%)", "daily_self_sufficiency_moving_average.png")
plot_metric("Self_Consumption (%)", "Daily Self-Consumption (%)", "daily_self_consumption_moving_average.png")

logging.info("✅ Self-Sufficiency and Self-Consumption Lien Plots are  saved successfully.")


# === Function: preprocess and aggregate cycles ===
def prepare_daily_cycles(df, scenario_name):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df = df[df["DateTime"].notna()]
    df["SOC_percent"] = 100 * df["SOC"] / 9.37
    df["Cycle"] = df["Cycle"].fillna(0)
    df["Date"] = df["DateTime"].dt.date
    daily = df.groupby("Date")["Cycle"].sum().reset_index()
    df["Date"] = df["DateTime"].dt.floor("D")
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Scenario"] = scenario_name
    return daily

# === Prepare Data for the Two BESS Scenarios ===
daily_optimized = prepare_daily_cycles(optimized_df, "(a) Optimized BESS with PV")
daily_rule_based = prepare_daily_cycles(rule_based_df, "(b) Rule-Based BESS with PV")

# === Combine Only BESS Cases ===
combined_df = pd.concat([daily_optimized, daily_rule_based], ignore_index=True)

# === Date Buffering ===
left_buffer = pd.Timedelta(days=7)
right_buffer = pd.Timedelta(days=7)
min_date = combined_df["Date"].min() - left_buffer
max_date = combined_df["Date"].max() + right_buffer

# === Plotting: Daily Battery Cycles ===
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8), sharex=True)

scenario_list = ["(a) Optimized BESS with PV", "(b) Rule-Based BESS with PV"]
colors = ["#1b9e77", "#d95f02"]

for ax, scenario, color in zip(axes, scenario_list, colors):
    df = combined_df[combined_df["Scenario"] == scenario]
    ax.bar(df["Date"], df["Cycle"], edgecolor='black', color=color, width=1)
    ax.set_ylabel("Cycles", fontsize=14)
    ax.set_title(f"{scenario}", fontsize=16, pad=6)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xlim([min_date, max_date])
    ax.label_outer()

# Add suptitle
fig.suptitle("Daily Battery Cycling Behavior Across Optimized and Rule-Based Scenarios in 2024", fontsize=16, y=0.94)

# Shared X-axis formatting
axes[-1].set_xlabel("Date", fontsize=14)
xtick_start = pd.to_datetime("2024-01-01")
xticks = pd.date_range(start=xtick_start, end=max_date, freq="14D")
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=14)

# Add vertical lines at month starts
month_starts = pd.date_range(start=min_date, end=max_date, freq="MS")
for ax in axes:
    for month_start in month_starts:
        ax.axvline(month_start, color='lightgrey', linestyle='--', linewidth=1.2, alpha=0.8)
# Align layout
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_daily_battery_cycles_BESS_only.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Combined Daily Battery Cycle Distribution (BESS only) completed.")


# === Function: Create Pivot Tables for Net Grid Exchange Heatmaps ===
def get_net_grid_pivot(df):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df = df[df["DateTime"].notna()]
    df["Date"] = df["DateTime"].dt.date
    df["Hour"] = df["DateTime"].dt.hour
    return df.pivot(index="Hour", columns="Date", values="P_net_grid")

# === Generate Pivot Tables ===
net_pivot_optimized = get_net_grid_pivot(optimized_df)
net_pivot_rule_based = get_net_grid_pivot(rule_based_df)
net_pivot_just_pv = get_net_grid_pivot(just_pv_df)

# === Plot Combined Net Grid Exchange Heatmap (+ Import, - Export) ===
fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(22, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1]},
    constrained_layout=False
)

fig.suptitle("Hourly Net Grid Exchange Across All Scenarios in 2024 (+ Import, - Export) ", fontsize=20, y=0.94)

scenario_titles = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]
net_pivots = [net_pivot_optimized, net_pivot_rule_based, net_pivot_just_pv]
colormaps = ["seismic", "seismic", "seismic"]

for i, (ax, data, title, cmap) in enumerate(zip(axes, net_pivots, scenario_titles, colormaps)):
    data_min = np.nanmin(data.to_numpy())
    data_max = np.nanmax(data.to_numpy())
    vmin = np.floor(data_min)
    vmax = np.ceil(data_max)

    sns.heatmap(
        data,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        center=0,
        cbar_kws={
            'pad': 0.01  # reduce distance between plot and colorbar
        },
        ax=ax
    )
    # Force whole-number ticks on colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.arange(vmin, vmax + 1, 1))  # adjust step if needed
    colorbar.set_label('Net Grid Exchange [kW]', size=16)
    colorbar.ax.tick_params(labelsize=14)

    ax.set_title(title, fontsize=18, pad=6)
    ax.set_ylabel("Hour of Day", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    if i < 2:
        ax.set_xlabel("")
    ax.label_outer()

    # Add vertical lines at start of each month
    start_date = pd.to_datetime(data.columns.min())
    end_date = pd.to_datetime(data.columns.max())
    for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
        month_start_date = month_start.date()
        if month_start_date in data.columns:
            ax.axvline(x=data.columns.get_loc(month_start_date), color='lightgray', linestyle='--', linewidth=1.2, alpha=0.8)

# X-axis formatting
xticks = pd.date_range(start=net_pivot_optimized.columns.min(), end=net_pivot_optimized.columns.max(), freq="14D")
axes[-1].set_xticks(np.arange(0, len(net_pivot_optimized.columns), 14))
axes[-1].set_xticklabels(
    [pd.to_datetime(d).strftime('%b-%d') for d in net_pivot_optimized.columns[::14]],
    rotation=45, ha='right', fontsize=14
)
axes[-1].set_xlabel("Date", fontsize=16)

fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_hourly_net_grid_exchange_heatmaps.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Combined Net Grid Exchange Heatmap with scenario-specific colorbars saved.")


# === Function: Create Pivot Tables for Heatmaps ===
def get_hourly_heatmap_pivots(df):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
    df = df[df["DateTime"].notna()]
    df["Date"] = df["DateTime"].dt.date
    df["Hour"] = df["DateTime"].dt.hour
    import_pivot = df.pivot(index="Hour", columns="Date", values="P_from_grid")
    export_pivot = df.pivot(index="Hour", columns="Date", values="P_to_grid")
    return import_pivot, export_pivot

# === Generate Pivot Tables ===
import_optimized, export_optimized = get_hourly_heatmap_pivots(optimized_df)
import_rule_based, export_rule_based = get_hourly_heatmap_pivots(rule_based_df)
import_just_pv, export_just_pv = get_hourly_heatmap_pivots(just_pv_df)

# === Combined Grid Import Heatmap ===
fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(22, 14),  # Make it wider and taller
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1]},  # Equal heights
    constrained_layout=False  # We'll control spacing manually
)
fig.suptitle("Hourly Grid Import Heatmap Across All Scenarios in 2024", fontsize=22, y=0.94)

scenario_titles = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]
import_pivots = [import_optimized, import_rule_based, import_just_pv]
colors = ["Blues", "Blues", "Blues"]

vmin = 0
vmax = max(np.nanmax(p.to_numpy()) for p in import_pivots)

for i, (ax, data, title, cmap) in enumerate(zip(axes, import_pivots, scenario_titles, colors)):
    sns.heatmap(
        data,
        cmap=cmap,
        norm=PowerNorm(gamma=0.5),  # sqrt scale
        cbar_kws={'label': 'Grid Import [kW]'},
        ax=ax
    )
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_ylabel("Hour of Day")
    if i < 2:
        ax.set_xlabel("")
    ax.label_outer()

    start_date = pd.to_datetime(data.columns.min())
    end_date = pd.to_datetime(data.columns.max())

    for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
        month_start_date = month_start.date()
        if month_start_date in data.columns:
            ax.axvline(x=data.columns.get_loc(month_start_date), color='lightgray', linestyle='--', linewidth=1.2, alpha=0.8)

# Shared X-axis formatting
xticks = pd.date_range(start=import_optimized.columns.min(), end=import_optimized.columns.max(), freq="14D")
axes[-1].set_xticks(np.arange(0, len(import_optimized.columns), 14))
axes[-1].set_xticklabels(
    [pd.to_datetime(d).strftime('%b-%d') for d in import_optimized.columns[::14]],
    rotation=45, ha='right', fontsize=9
)
axes[-1].set_xlabel("Date")
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_hourly_grid_import_heatmaps.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Combined Grid Export Heatmap ===
fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(22, 14),  # Make it wider and taller
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1]},  # Equal heights
    constrained_layout=False  # We'll control spacing manually
)

fig.suptitle("Hourly Grid Export Heatmap Across All Scenarios in 2024", fontsize=22, y=0.94)

export_pivots = [export_optimized, export_rule_based, export_just_pv]
scenario_titles = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]
colors = ["Oranges", "Oranges", "Oranges"]


for i, (ax, data, title, cmap) in enumerate(zip(axes, export_pivots, scenario_titles, colors)):
    sns.heatmap(
        data,
        cmap=cmap,
        cbar_kws={'label': 'Grid Export [kW]'},
        ax=ax
    )
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_ylabel("Hour of Day")
    if i < 2:
        ax.set_xlabel("")  # Remove xlabel for top two plots
    ax.label_outer()

    start_date = pd.to_datetime(data.columns.min())
    end_date = pd.to_datetime(data.columns.max())

    for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
        month_start_date = month_start.date()
        if month_start_date in data.columns:
            ax.axvline(x=data.columns.get_loc(month_start_date), color='lightgray', linestyle='--', linewidth=1.2, alpha=0.8)

# X-axis formatting
xticks = pd.date_range(start=export_optimized.columns.min(), end=export_optimized.columns.max(), freq="14D")
axes[-1].set_xticks(np.arange(0, len(export_optimized.columns), 14))
axes[-1].set_xticklabels(
    [pd.to_datetime(d).strftime('%b-%d') for d in export_optimized.columns[::14]],
    rotation=45, ha='right', fontsize=9
)
axes[-1].set_xlabel("Date")

# Fix suptitle placement
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_hourly_grid_export_heatmaps.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Combined Grid Import and Export Heatmaps with subplot titles generated.")

# === Daily Grid Import/Export: Stacked Bar Chart Across All Scenarios ===

def prepare_daily_ie(df, label):
    daily = df.copy()
    daily["DateTime"] = pd.to_datetime(daily["DateTime"])
    daily["Date"] = daily["DateTime"].dt.date
    grouped = daily.groupby("Date")[["P_from_grid", "P_to_grid"]].sum().reset_index()
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    grouped["Scenario"] = label
    return grouped

# Prepare data
daily_ie_optimized = prepare_daily_ie(optimized_df, "Optimized BESS with PV")
daily_ie_rule_based = prepare_daily_ie(rule_based_df, "Rule-Based BESS with PV")
daily_ie_just_pv = prepare_daily_ie(just_pv_df, "PV Without BESS")

all_daily_ie = [daily_ie_optimized, daily_ie_rule_based, daily_ie_just_pv]
titles = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]

# Get global x limits
all_dates = pd.concat(all_daily_ie)["Date"]
left_buffer = pd.Timedelta(days=7)
right_buffer = pd.Timedelta(days=7)
xlim_left = all_dates.min() - left_buffer
xlim_right = all_dates.max() + right_buffer

# === Plotting ===
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(22, 12), sharex=True)
fig.suptitle("Daily Grid Import and Export Over 2024 Across All Scenarios in 2024", fontsize=20, y=0.94)

for ax, df, title in zip(axes, all_daily_ie, titles):
    ax.bar(df["Date"], df["P_from_grid"], label="Grid Import", color="steelblue", width=1.0)
    ax.bar(df["Date"], -df["P_to_grid"], label="Grid Export", color="orange", width=1.0)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontsize=18, pad=6)
    ax.set_ylabel("Energy [kWh]", fontsize=16)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xlim([xlim_left, xlim_right])
    if title == "(a) Optimized BESS with PV":
        ax.legend(loc="upper left", fontsize=20)
    ax.label_outer()

# Custom X-axis ticks
xtick_start = pd.to_datetime("2024-01-01")
xticks = pd.date_range(start=xtick_start, end=xlim_right, freq="14D")
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=16)
axes[-1].set_xlabel("Date")

# Add vertical lines at month starts
month_starts = pd.date_range(start=xlim_left, end=xlim_right, freq="MS")
for ax in axes:
    for month_start in month_starts:
        ax.axvline(month_start, color='lightgrey', linestyle='--', linewidth=1.2, alpha=0.8)

# Save
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_daily_grid_import_export_all_scenarios.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Daily Import/Export stacked plot completed for all scenarios.")

# === Monthly Average Export for Each Scenario ===

def compute_monthly_avg(df, scenario):
    df["Month"] = df["Date"].dt.month
    monthly_avg = df.groupby("Month")[["P_from_grid", "P_to_grid"]].mean().reset_index()
    monthly_avg["Scenario"] = scenario
    return monthly_avg

monthly_avg_dfs = [
    compute_monthly_avg(daily_ie_optimized, "Optimized BESS with PV"),
    compute_monthly_avg(daily_ie_rule_based, "Rule-Based BESS with PV"),
    compute_monthly_avg(daily_ie_just_pv, "PV Without BESS")
]

# Combine and reorder columns
monthly_avg_all = pd.concat(monthly_avg_dfs, ignore_index=True)
monthly_avg_all = monthly_avg_all[["Scenario", "Month", "P_from_grid", "P_to_grid"]]

# Export to CSV
monthly_avg_all.to_csv(os.path.join(csv_dir, "monthly_avg_grid_import_export.csv"), index=False)

logging.info("✅ Monthly average import/export CSV exported to csv_dir.")

# === Load Data Again if Needed ===
optimized_df["Scenario"] = "(a) Optimized BESS with PV"
rule_based_df["Scenario"] = "(b) Rule-Based BESS with PV"
just_pv_df["Scenario"] = "(c) PV Without BESS"

combined_df = pd.concat([optimized_df, rule_based_df, just_pv_df], ignore_index=True)

# === Plot ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False, constrained_layout=False)

scenarios = ["(a) Optimized BESS with PV", "(b) Rule-based BESS with PV", "(c) PV Without BESS"]
colors = [("steelblue", "orange")] * 3  # Import, Export colors

for ax, scenario, (import_color, export_color) in zip(axes, scenarios, colors):
    df = combined_df[combined_df["Scenario"] == scenario]

    import_counts = df.groupby("Hour")["P_from_grid"].apply(lambda x: (x > 0).sum())
    export_counts = df.groupby("Hour")["P_to_grid"].apply(lambda x: (x > 0).sum())

    width = 0.4
    ax.bar(import_counts.index - width / 2, import_counts.values, width=width, color=import_color,
           label="Grid Import Events")
    ax.bar(export_counts.index + width / 2, export_counts.values, width=width, color=export_color,
           label="Grid Export Events")

    ax.set_ylabel("Freq. (hours/year)")
    ax.set_title(scenario, fontsize=13)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=9)
    ax.legend(loc="upper right", fontsize=13)


axes[-1].set_xlabel("Hour of Day")
plt.tight_layout(pad=0.1)
plt.subplots_adjust(
    top=0.90,     # slightly reduced to fit suptitle tighter
    bottom=0.07,
    left=0.055,
    right=0.985,
    hspace=0.25   # reduced subplot gap
)

fig.suptitle("Hourly Frequency of Grid Import and Export Events Across All Scenarios in 2024", fontsize=16, y=0.96)
# Save
output_path = os.path.join(visualization_dir, "combined_hourly_import_export_event_frequency.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
logging.info("✅ Combined Hourly Import/Export Frequency Histogram completed.")

hourly_data = []

# === Tag Scenarios ===
optimized_df["Scenario"] = "(a) Optimized BESS with PV"
rule_based_df["Scenario"] = "(b) Rule-Based BESS with PV"
combined_df = pd.concat([optimized_df, rule_based_df], ignore_index=True)

# === Plot Setup ===
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

scenarios = ["(a) Optimized BESS with PV", "(b) Rule-Based BESS with PV"]
colors = [("#1f77b4", "#ff7f0e")] * 2  # Charge, Discharge

for ax, scenario, (ch_color, dc_color) in zip(axes, scenarios, colors):
    df = combined_df[combined_df["Scenario"] == scenario]
    ch = df[df["P_ch"] > 0].groupby("Hour").size().reindex(range(24), fill_value=0)
    dc = df[df["P_dc"] > 0].groupby("Hour").size().reindex(range(24), fill_value=0)

    width = 0.4
    ax.bar(ch.index - width/2, ch.values, width=width, color=ch_color, label="Charging Events")
    ax.bar(dc.index + width/2, dc.values, width=width, color=dc_color, label="Discharging Events")

    ax.set_ylabel("Freq. (hours/year)")
    ax.set_title(scenario, fontsize=13)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xticks(range(24))
    ax.tick_params(labelbottom=True)
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=9)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.5)

for scenario in scenarios:
    df = combined_df[combined_df["Scenario"] == scenario]
    ch = df[df["P_ch"] > 0].groupby("Hour").size().reindex(range(24), fill_value=0)
    dc = df[df["P_dc"] > 0].groupby("Hour").size().reindex(range(24), fill_value=0)

    for hour in range(24):
        hourly_data.append({
            "Scenario": scenario,
            "Hour": hour,
            "Charging Events": ch[hour],
            "Discharging Events": dc[hour]
        })

hourly_df = pd.DataFrame(hourly_data)
csv_output_path = os.path.join(csv_dir, "hourly_charging_discharging_events_by_scenario.csv")
hourly_df.to_csv(csv_output_path, index=False)
logging.info("✅ Hourly Charging/Discharging Event Data saved to CSV.")

axes[-1].set_xlabel("Hour of Day")
fig.suptitle("Hourly Frequency of Battery Charging and Discharging Events Across Optimized and Rule-Based Scenarios in 2024", fontsize=15, y=0.98)

# Layout adjustments
plt.tight_layout(pad=0.05)
plt.subplots_adjust(
    top=0.88,      # lower than before to pull the top plot down
    bottom=0.08,
    left=0.06,
    right=0.985,
    hspace=0.3
)

# Save
output_path = os.path.join(visualization_dir, "combined_hourly_ch_dc_event_frequency_BESS_only.png")
plt.savefig(output_path, dpi=300)
plt.close()

logging.info("✅ Hourly Charging/Discharging Frequency Comparison Chart completed. (Optimized vs Rule-Based PV+BESS)")


# === Combined Self-Consumption and Self-Sufficiency Heatmaps ===
# Tag scenarios
optimized_df["Scenario"] = "(a) Optimized BESS with PV"
rule_based_df["Scenario"] = "(b) Rule-Based BESS with PV"
just_pv_df["Scenario"] = "(c) PV Without BESS"

# Merge data
combined_df = pd.concat([optimized_df, rule_based_df, just_pv_df], ignore_index=True)
combined_df["DateTime"] = pd.to_datetime(combined_df["DateTime"], errors="coerce")
combined_df = combined_df.dropna(subset=["DateTime"])
combined_df["Date"] = combined_df["DateTime"].dt.date
combined_df["Hour"] = combined_df["DateTime"].dt.hour

# Scenario list and colormap
scenarios = ["(a) Optimized BESS with PV", "(b) Rule-Based BESS with PV", "(c) PV Without BESS"]
cmaps = ["YlGnBu"] * 3

# === Self-Consumption Heatmap (All Scenarios) ===
fig, axes = plt.subplots(3, 1, figsize=(25, 12), sharex=True)
fig.suptitle("Hourly Self-Consumption Heatmap [%] Across All Scenarios", fontsize=18, y=0.94)

for ax, scenario, cmap in zip(axes, scenarios, cmaps):
    df = combined_df[combined_df["Scenario"] == scenario]
    sc_pivot = df.pivot(index="Hour", columns="Date", values="Self_Consumption (%)")
    sc_pivot = sc_pivot[sorted(sc_pivot.columns)]
    sc_pivot.columns = pd.to_datetime(sc_pivot.columns)

    tick_positions = np.arange(0, len(sc_pivot.columns), 14)
    tick_labels = [sc_pivot.columns[i].strftime('%b-%d') for i in tick_positions]

    sns.heatmap(sc_pivot, cmap=cmap, vmin=0, vmax=100, cbar_kws={'label': 'Self-Consumption [%]'}, ax=ax)
    ax.set_title(scenario, fontsize=14, pad=6)
    ax.set_ylabel("Hour of Day")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    if scenario != scenarios[-1]:
        ax.set_xlabel("")

    start_date = combined_df["DateTime"].min()
    end_date = combined_df["DateTime"].max()

    for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
        ax.axvline(x=sc_pivot.columns.get_loc(month_start), color='grey', linestyle='--', linewidth=1.2, alpha=0.8)

axes[-1].set_xlabel("Date (MMM-DD)")
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_self_consumption_heatmaps.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Self-Sufficiency Heatmap (All Scenarios) ===
fig, axes = plt.subplots(3, 1, figsize=(25, 12), sharex=True)
fig.suptitle("Hourly Self-Sufficiency Heatmap [%] Across All Scenarios in 2024", fontsize=18, y=0.94)

for ax, scenario, cmap in zip(axes, scenarios, cmaps):
    df = combined_df[combined_df["Scenario"] == scenario]
    ss_pivot = df.pivot(index="Hour", columns="Date", values="Self_Sufficiency (%)")
    ss_pivot = ss_pivot[sorted(ss_pivot.columns)]
    ss_pivot.columns = pd.to_datetime(ss_pivot.columns)

    tick_positions = np.arange(0, len(ss_pivot.columns), 14)
    tick_labels = [ss_pivot.columns[i].strftime('%b-%d') for i in tick_positions]

    sns.heatmap(ss_pivot, cmap=cmap, vmin=0, vmax=100, cbar_kws={'label': 'Self-Sufficiency [%]'}, ax=ax)
    ax.set_title(scenario, fontsize=14, pad=6)
    ax.set_ylabel("Hour of Day")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    if scenario != scenarios[-1]:
        ax.set_xlabel("")

    start_date = combined_df["DateTime"].min()
    end_date = combined_df["DateTime"].max()

    for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
        ax.axvline(x=sc_pivot.columns.get_loc(month_start), color='grey', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(month_start, color='grey', linestyle='--', linewidth=1.2, alpha=0.8)

axes[-1].set_xlabel("Date (MMM-DD)")
fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_self_sufficiency_heatmaps.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Combined Self-Consumption & Self-Sufficiency Heatmaps generated across all scenarios.")

# === Prepare Weekly Aggregation per Scenario ===
def prepare_weekly_df(df, scenario_label):
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Week_Start"] = df["DateTime"].dt.to_period("W").apply(lambda r: r.start_time)
    grouped = df.groupby("Week_Start").agg({
        "Net Bill": "sum",
        "Optimization Profit": "sum",
        "EP_epex": "mean"
    }).reset_index()
    grouped["Scenario"] = scenario_label
    return grouped

weekly_opt = prepare_weekly_df(optimized_df, "(a) Optimized BESS with PV")
weekly_rule_based = prepare_weekly_df(rule_based_df, "(b) Rule-Based BESS with PV")
weekly_justpv = prepare_weekly_df(just_pv_df, "(c) PV Without BESS")

# === Merge Weekly Results Across Scenarios ===
weekly_opt_summary = weekly_opt[["Week_Start", "Optimization Profit", "Net Bill"]].rename(columns={
    "Optimization Profit": "Weekly_Opt_Profit_Optimized",
    "Net Bill": "Weekly_Net_Bill_Optimized"
})

weekly_rule_based_summary = weekly_rule_based[["Week_Start", "Optimization Profit", "Net Bill"]].rename(columns={
    "Optimization Profit": "Weekly_Opt_Profit_rule_based",
    "Net Bill": "Weekly_Net_Bill_rule_based"
})

weekly_pv_summary = weekly_justpv[["Week_Start", "Optimization Profit", "Net Bill"]].rename(columns={
    "Optimization Profit": "Weekly_Opt_Profit_PV_without_BESS",
    "Net Bill": "Weekly_Net_Bill_PV_without_BESS"
})

# Merge on Week_Start
weekly_summary = weekly_opt_summary.merge(weekly_rule_based_summary, on="Week_Start", how="outer")
weekly_summary = weekly_summary.merge(weekly_pv_summary, on="Week_Start", how="outer")

# Add ISO Week Number column
weekly_summary["Week Number"] = weekly_summary["Week_Start"].dt.isocalendar().week

# Final column order
weekly_summary = weekly_summary[[
    "Week Number",
    "Weekly_Opt_Profit_Optimized", "Weekly_Net_Bill_Optimized",
    "Weekly_Opt_Profit_rule_based", "Weekly_Net_Bill_rule_based",
    "Weekly_Opt_Profit_PV_without_BESS", "Weekly_Net_Bill_PV_without_BESS"
]]

# Save to CSV
csv_path = os.path.join(csv_dir, "weekly_results_all_scenarios.csv")
weekly_summary.to_csv(csv_path, index=False)
logging.info("✅ Weekly optimization profit and net bill for all scenarios written to CSV.")


# === Combine all scenarios ===
weekly_combined_df = pd.concat([weekly_opt, weekly_rule_based, weekly_justpv], ignore_index=True)

# === Common Formatting ===
tick_dates = pd.date_range(start=weekly_combined_df["Week_Start"].min(),
                           end=weekly_combined_df["Week_Start"].max(),
                           freq='2W-MON')
tick_labels = [d.strftime('%b-%d') for d in tick_dates]

scenario_order = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]

# === FIGURE 1: Optimization Profit ===
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Weekly Optimization Profit with Avg. EPEX Price Across All Scenarios in 2024", fontsize=16, y=0.95)

for ax, scenario in zip(axes, scenario_order):
    df = weekly_combined_df[weekly_combined_df["Scenario"] == scenario]
    weeks = df["Week_Start"]

    ax.bar(weeks, df["Optimization Profit"], width=5, color="#9467bd", label="Optimization Profit [€]")
    twin_ax = ax.twinx()
    twin_ax.plot(weeks, df["EP_epex"], color="blue", marker='o', linestyle='--', label="Weekly Avg. EPEX [€/kWh]")

    ax.set_ylabel("Opt. Profit [€]", color="#9467bd", fontsize=14)
    twin_ax.set_ylabel("Weekly Avg. EPEX [€/kWh]", color="blue", fontsize=12)
    ax.set_title(scenario, fontsize=16)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6)


    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = twin_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=13)

axes[-1].set_xlabel("Week Start Date", fontsize=14)
axes[-1].set_xticks(tick_dates)
axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(visualization_dir, "weekly_optimization_profit_all_scenarios.png"), dpi=300)
plt.close()
logging.info("✅ Weekly Optimization Profit for all scenarios saved.")


# === FIGURE 2: Net Bill ===
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Weekly Net Bill with Avg. EPEX Price Across All Scenarios in 2024", fontsize=16, y=0.95)

for ax, scenario in zip(axes, scenario_order):
    df = weekly_combined_df[weekly_combined_df["Scenario"] == scenario]
    weeks = df["Week_Start"]

    ax.bar(weeks, df["Net Bill"], width=5, color="#1b9e77", label="Net Bill [€]")
    twin_ax = ax.twinx()
    twin_ax.plot(weeks, df["EP_epex"], color="blue", marker='o', linestyle='--', label="Weekly Avg. EPEX [€/kWh]")

    ax.set_ylabel("Net Bill [€]", color="#1b9e77", fontsize=14)
    twin_ax.set_ylabel("Weekly Avg. EPEX [€/kWh]", color="blue", fontsize=12)
    ax.set_title(scenario, fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4)


    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = twin_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=13)

axes[-1].set_xlabel("Week Start Date", fontsize=14)
axes[-1].set_xticks(tick_dates)
axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(visualization_dir, "weekly_net_bill_all_scenarios.png"), dpi=300)
plt.close()
logging.info("✅ Weekly Net Bill for all scenarios saved.")


# === Daily PV Allocation Composition Across Scenarios ===
# === Prepare Data Function ===
def prepare_pv_allocation_df(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["DateTime"]).dt.date
    daily = df.groupby("Date").agg({
        "P_PV": "sum",
        "P_ch_pv": "sum",
        "P_PV_to_grid": "sum"
    }).reset_index()

    daily["PV to Load"] = daily["P_PV"] - daily["P_ch_pv"] - daily["P_PV_to_grid"]
    daily["Date"] = pd.to_datetime(daily["Date"])
    pv_total = daily["P_PV"].replace(0, np.nan)

    return pd.DataFrame({
        "Date": daily["Date"],
        "PV to Load": 100 * daily["PV to Load"] / pv_total,
        "PV to Battery": 100 * daily["P_ch_pv"] / pv_total,
        "PV to Grid": 100 * daily["P_PV_to_grid"] / pv_total
    })

# === Prepare Data for All Scenarios ===
pv_plot_optimized = prepare_pv_allocation_df(optimized_df)
pv_plot_rule_based = prepare_pv_allocation_df(rule_based_df)
pv_plot_just_pv = prepare_pv_allocation_df(just_pv_df)

pv_plot_dfs = [pv_plot_optimized, pv_plot_rule_based, pv_plot_just_pv]
scenario_titles = [
    "(a) Optimized BESS with PV",
    "(b) Rule-Based BESS with PV",
    "(c) PV Without BESS"
]

# === Plotting ===
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(22, 12), sharex=True)
fig.suptitle("Daily PV Allocation Composition by Flow Direction [%] Across All Scenarios", fontsize=18, y=0.94)

for ax, df, title in zip(axes, pv_plot_dfs, scenario_titles):
    bottom = np.zeros(len(df))
    for label in ["PV to Load", "PV to Battery", "PV to Grid"]:
        ax.bar(df["Date"], df[label], bottom=bottom, label=label, width=1)
        bottom += df[label]

    ax.set_ylabel("PV Allocation [%]")
    ax.set_title(title, fontsize=14, pad=6)
    ax.grid(visible=True, axis='y', linestyle='--', linewidth=0.4, alpha=0.8)
    ax.grid(visible=False, axis='x')
    ax.label_outer()

# X-axis formatting
all_dates = pv_plot_optimized["Date"]
xticks = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq="14D")
axes[-1].set_xlim([all_dates.min() - pd.Timedelta(days=7), all_dates.max() + pd.Timedelta(days=7)])
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=9)
axes[-1].set_xlabel("Date (MMM-DD)")

month_starts = pd.date_range(
    start=all_dates.min() - pd.Timedelta(days=7),
    end=all_dates.max() + pd.Timedelta(days=7),
    freq="MS"
)

for ax in axes:
    for month_start in month_starts:
        ax.axvline(month_start, color='grey', linestyle='--', linewidth=1.2, alpha=0.8)

# Legend (shared)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="lower left", fontsize=13)

# Save and finalize
fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_daily_pv_allocation_composition.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

logging.info("✅ Combined Daily PV Allocation Composition Chart is saved.")

# === Daily Load Supply Composition Across Scenarios ===
# === Set up scenario DataFrames ===
scenario_dfs = {
    "(a) Optimized BESS with PV": optimized_df,
    "(b) Rule-Based BESS with PV": rule_based_df,
    "(c) PV Without BESS": just_pv_df
}

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(22, 12), sharex=True)
fig.suptitle("Daily Load Supply Composition by Source (%) Across All Scenarios in 2024", fontsize=18, y=0.94)

# To collect all dates after processing
all_date_list = []

for ax, (scenario_name, df) in zip(axes, scenario_dfs.items()):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.floor("D")

    daily = df.groupby("Date").agg({
        "P_PV": "sum",
        "P_ch_pv": "sum",
        "P_Load_from_grid": "sum",
        "P_dc_to_load": "sum",
        "P_PV_to_grid": "sum"
    }).reset_index()

    daily["PV_to_Load"] = daily["P_PV"] - daily["P_ch_pv"] - daily["P_PV_to_grid"]
    daily["Date"] = pd.to_datetime(daily["Date"])  # Convert to datetime

    all_date_list.append(daily["Date"])  # Collect dates

    supply_df = pd.DataFrame({
        "Date": daily["Date"],
        "PV to Load": daily["PV_to_Load"],
        "Battery to Load": daily["P_dc_to_load"],
        "Grid to Load": daily["P_Load_from_grid"]
    })

    total = supply_df[["PV to Load", "Battery to Load", "Grid to Load"]].sum(axis=1).replace(0, np.nan)
    for col in ["PV to Load", "Battery to Load", "Grid to Load"]:
        supply_df[col] = 100 * supply_df[col] / total

    bottom = np.zeros(len(supply_df))
    for source in ["PV to Load", "Battery to Load", "Grid to Load"]:
        ax.bar(supply_df["Date"], supply_df[source], bottom=bottom, label=source, width=1.0)
        bottom += supply_df[source]

    ax.set_title(scenario_name, fontsize=14, pad=6)
    ax.set_ylabel("Load Supply [%]")
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)

for ax in axes:
    ax.set_ylim([0, 100])


# === Now safely generate common X-axis ticks ===
all_dates = pd.concat(all_date_list)
# Recalculate common start and end with buffers
true_min = all_dates.min()
true_max = all_dates.max()
start = true_min - pd.Timedelta(days=7)
end = true_max + pd.Timedelta(days=7)

axes[-1].set_xlim([start, end])

xticks = pd.date_range(start=true_min, end=true_max, freq="14D")

# Apply to bottom subplot
axes[-1].set_xlim([start, end])
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=9)
axes[-1].set_xlabel("Date")

# Vertical month lines
month_starts = pd.date_range(start=start, end=end, freq="MS")
for ax in axes:
    for month_start in month_starts:
        ax.axvline(month_start, color='grey', linestyle='--', linewidth=1.2, alpha=1)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="lower left", fontsize=13)

fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_daily_load_supply_composition.png"), dpi=300, bbox_inches='tight')
plt.close()

logging.info("✅ Combined Daily Load Composition Chart is saved.")

# === Daily Optimization Profit Composition Comparison Across Scenarios ===
# === Scenario Setup ===
scenarios = {
    "(a) Optimized BESS with PV": {
        "df": optimized_df,
        "fixed_cost": 0.4376
    },
    "(b) Rule-Based BESS with PV": {
        "df": rule_based_df,
        "fixed_cost": 0.4376
    },
    "(c) PV Without BESS": {
        "df": just_pv_df,
        "fixed_cost": 0.377
    }
}
# === Components to Plot ===
components = [
    "Saving for Avoided Grid Import",
    "Earning for Discharge to Grid",
    "Earning for PV Feed-in to Grid",
    "Cost for Covering Load from Grid",
    "Cost for Charging Battery from Grid",
    "Opportunity Cost for Not PV Feed-in to Grid"
]
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#bcbd22", "#e377c2",
    "#7f7f7f"
]

# === Plotting ===
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(22, 12), sharex=True)
fig.suptitle("Daily Optimization Profit Compositon by Percentage [%] Across All Scenarios in 2024", fontsize=20, y=0.94)

# Set date limits for all scenarios
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-12-31")

combined_dfs = []

for ax, (scenario_name, scenario_info) in zip(axes, scenarios.items()):
    df = scenario_info["df"].copy()
    fixed_cost_per_day = scenario_info["fixed_cost"]

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.floor("D")

    # Filter strictly to year 2024
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Create full daily date range for 2024
    full_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Aggregate daily values
    daily_df = df.groupby("Date")[components].sum()
    daily_df["Fixed Daily Cost"] = fixed_cost_per_day

    # Reindex to full 2024 range (missing days become NaN)
    daily_df = daily_df.reindex(full_dates).fillna(0)
    daily_df.index.name = "Date"
    daily_df = daily_df.reset_index()

    # Add dummy rows for the buffer zone
    buffer_before = pd.date_range(start=start_date - pd.Timedelta(days=7), end=start_date - pd.Timedelta(days=1),
                                  freq="D")
    buffer_after = pd.date_range(start=end_date + pd.Timedelta(days=1), end=end_date + pd.Timedelta(days=7), freq="D")
    buffer_df = pd.DataFrame({
        "Date": list(buffer_before) + list(buffer_after),
        **{col: 0 for col in components + ["Fixed Daily Cost"]}
    })
    share_df = pd.concat([daily_df, buffer_df], ignore_index=True).sort_values("Date")

    # Normalize by total absolute value
    cols_to_use = components + ["Fixed Daily Cost"]
    total_abs = daily_df[cols_to_use].abs().sum(axis=1)
    share_df = daily_df.copy()
    for col in cols_to_use:
        share_df[col] = 100 * share_df[col].abs() / total_abs

    # Plot
    bottom = np.zeros(len(share_df))
    for col, color in zip(cols_to_use, colors):
        ax.bar(share_df["Date"], share_df[col], bottom=bottom, label=col, color=color, width=1.0)
        bottom += share_df[col]

    ax.set_title(scenario_name, fontsize=18, pad=6)
    ax.set_ylabel("Share [%]", fontsize=16)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.label_outer()

    # === Save Daily Optimization Profit Composition (Percentage) as CSV ===
    composition_to_save = share_df.copy()
    composition_to_save.insert(1, "Scenario", scenario_name)
    composition_to_save = composition_to_save[["Date", "Scenario"] + cols_to_use]
    filename = f"daily_optimization_profit_composition_{scenario_name[4:].lower().replace(' ', '_')}.csv"
    composition_to_save.to_csv(os.path.join(csv_dir, filename), index=False)

    # === Add to combined DataFrame list ===
    combined_dfs.append(composition_to_save)

# === Save Combined CSV for All Scenarios ===
combined_df = pd.concat(combined_dfs, ignore_index=True)
combined_df.to_csv(os.path.join(csv_dir, "daily_optimization_profit_composition_all_scenarios.csv"), index=False)
logging.info("✅ Combined Daily Optimization Profit Composition CSV saved.")

# === X-axis Shared Formatting ===
xlim_left = pd.to_datetime("2023-12-25")  # one week before Jan 1
xlim_right = pd.to_datetime("2025-01-07")  # one week after Dec 31

# Generate ticks starting strictly from Jan 1, 2024 to Dec 31, 2024 with 14-day steps
xticks = pd.date_range(start=pd.to_datetime("2024-01-01"), end=pd.to_datetime("2024-12-31"), freq="14D")

axes[-1].set_xlim([xlim_left, xlim_right])
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=16)
axes[-1].set_xlabel("Date (MMM-DD)", fontsize=16)


# Vertical month lines
month_starts = pd.date_range(start=xlim_left, end=xlim_right, freq="MS")
for ax in axes:
    for month_start in month_starts:
        ax.axvline(month_start, color='lightgrey', linestyle='--', linewidth=1.0, alpha=1)

# Shared Legend
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles, labels, loc="lower left", fontsize=16)

# Save
fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.95])
output_path = os.path.join(visualization_dir, "combined_daily_profit_component_percentages.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

logging.info("✅ Combined Daily Optimization Profit Component Share Chart is saved.")

# === Daily Net Bill Composition Comparison Across Scenarios ===
# Define fixed daily cost per scenario
fixed_costs = {
    "(a) Optimized BESS with PV": 0.4376,
    "(b) Rule-Based BESS with PV": 0.4376,
    "(c) PV Without BESS": 0.377
}

# Relevant components
components = [
    "Earning for Discharge to Grid",
    "Earning for PV Feed-in to Grid",
    "Cost for Covering Load from Grid",
    "Cost for Charging Battery from Grid"
]

# Prepare function
def prepare_daily_net_bill_components(df, scenario_label, fixed_cost):
    df = df.copy()
    # Ensure DateTime exists
    if "DateTime" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "DateTime" not in df.columns:
            df = df.rename(columns={"index": "DateTime"})

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.date
    daily = df.groupby("Date")[components].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Fixed Daily Cost"] = fixed_cost

    # Compute total absolute for percent normalization
    daily["Total_Abs"] = daily[components + ["Fixed Daily Cost"]].abs().sum(axis=1).replace(0, 1e-6)

    return pd.DataFrame({
        "Date": daily["Date"],
        "Earning for Discharge to Grid (%)": 100 * daily["Earning for Discharge to Grid"].abs() / daily["Total_Abs"],
        "Earning for PV Feed-in to Grid (%)": 100 * daily["Earning for PV Feed-in to Grid"].abs() / daily["Total_Abs"],
        "Cost for Covering Load from Grid (%)": 100 * daily["Cost for Covering Load from Grid"].abs() / daily["Total_Abs"],
        "Cost for Charging Battery from Grid (%)": 100 * daily["Cost for Charging Battery from Grid"].abs() / daily["Total_Abs"],
        "Fixed Daily Cost (%)": 100 * daily["Fixed Daily Cost"].abs() / daily["Total_Abs"],
        "Scenario": scenario_label
    })

# Prepare all three
nb_opt_df = prepare_daily_net_bill_components(optimized_df, "(a) Optimized BESS with PV", fixed_costs["(a) Optimized BESS with PV"])
nb_rb_df = prepare_daily_net_bill_components(rule_based_df, "(b) Rule-Based BESS with PV", fixed_costs["(b) Rule-Based BESS with PV"])
nb_jpv_df = prepare_daily_net_bill_components(just_pv_df, "(c) PV Without BESS", fixed_costs["(c) PV Without BESS"])

# === Save Daily Net Bill Composition (Percentage) as CSV ===
net_bill_combined = pd.concat([nb_opt_df, nb_rb_df, nb_jpv_df], ignore_index=True)
net_bill_combined.to_csv(os.path.join(csv_dir, "daily_net_bill_composition_all_scenarios.csv"), index=False)
logging.info("✅ Daily Net Bill Composition data saved for all scenarios.")


all_dfs = [nb_opt_df, nb_rb_df, nb_jpv_df]
titles = list(fixed_costs.keys())

# Get global x limits
all_dates = pd.concat([df["Date"] for df in all_dfs])
xlim_left = all_dates.min() - pd.Timedelta(days=7)
xlim_right = all_dates.max() + pd.Timedelta(days=7)
xticks = pd.date_range(start=xlim_left, end=xlim_right, freq="14D")
month_starts = pd.date_range(start=xlim_left, end=xlim_right, freq="MS")

# === Plotting ===
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(22, 12), sharex=True)
fig.suptitle("Daily Net Bill Composition by Percentage Across All Scenarios in 2024", fontsize=20, y=0.94)

color_map = {
    "Earning for Discharge to Grid (%)": "#ff7f0e",
    "Earning for PV Feed-in to Grid (%)": "#2ca02c",
    "Cost for Covering Load from Grid (%)": "#d62728",
    "Cost for Charging Battery from Grid (%)": "#bcbd22",
    "Fixed Daily Cost (%)": "#7f7f7f"
}

for ax, df, title in zip(axes, all_dfs, titles):
    bottom = np.zeros(len(df))
    for component, color in color_map.items():
        ax.bar(df["Date"], df[component], bottom=bottom, label=component.replace(" (%)", ""), color=color, width=1.0)
        bottom += df[component].values

    ax.set_title(title, fontsize=18, pad=6)
    ax.set_ylabel("Share [%]", fontsize=16)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Add month vertical lines
    for m in month_starts:
        ax.axvline(m, color='lightgrey', linestyle='--', linewidth=1.2, alpha=0.8)

# === X-axis Shared Formatting ===
xlim_left = pd.to_datetime("2023-12-25")  # one week before Jan 1
xlim_right = pd.to_datetime("2025-01-07")  # one week after Dec 31

# Generate ticks starting strictly from Jan 1, 2024 to Dec 31, 2024 with 14-day steps
xticks = pd.date_range(start=pd.to_datetime("2024-01-01"), end=pd.to_datetime("2024-12-31"), freq="14D")

axes[-1].set_xlim([xlim_left, xlim_right])
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xticks.strftime('%b-%d'), rotation=45, ha='right', fontsize=16)
axes[-1].set_xlabel("Date (MMM-DD)", fontsize=14)

# Shared Legend
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles, labels, loc="lower left", fontsize=18)

# Save and finalize
fig.tight_layout(rect=[0.03, 0.03, 0.985, 0.95])
plt.savefig(os.path.join(visualization_dir, "combined_daily_net_bill_component_percentages.png"), dpi=300, bbox_inches="tight")
plt.close()
logging.info("✅ Daily Net Bill Component Share Chart saved for all scenarios.")


# === Daily Energy Flow & Financials ===

SOC_ORANGE = "#d95f02"

for df in [optimized_df, rule_based_df]:
    # 1. Ensure DateTime is actually a datetime object (handle potential string format issues)
    if not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors='coerce')

    # 2. Create 'Day' (1-366) and 'Hour' (0-23) if they don't exist
    if "Day" not in df.columns:
        df["Day"] = df["DateTime"].dt.dayofyear

    if "Hour" not in df.columns:
        df["Hour"] = df["DateTime"].dt.hour

for day in range(1,367):
    df_opt_day = optimized_df[optimized_df["Day"] == day]
    df_rule_based_day = rule_based_df[rule_based_df["Day"] == day]

    if df_opt_day.empty and df_rule_based_day.empty:
        continue

    # Correct way to get actual date string
    if not df_opt_day.empty:
        date_str = df_opt_day["DateTime"].iloc[0].strftime("%d-%m-%Y")
    else:
        date_str = df_rule_based_day["DateTime"].iloc[0].strftime("%d-%m-%Y")

    # === FIGURE SETUP ===
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12),
                            gridspec_kw={'height_ratios': [2, 1.3]}, sharex=False)

    scenarios = [
        {"title_energy": "(a) Optimized BESS with PV – Daily Energy Flow",
         "title_fin": "(b) Optimized BESS with PV – Financial Metrics", "data": df_opt_day},
        {"title_energy": "(c) Rule-Based BESS with PV – Daily Energy Flow",
         "title_fin": "(d) Rule-Based BESS with PV – Financial Metrics", "data": df_rule_based_day}
    ]

    for col_idx, scenario in enumerate(scenarios):
        df_day = scenario["data"]
        if df_day.empty:
            continue

        hours = df_day["Hour"]
        battery_net = df_day["P_dc"] - df_day["P_ch"]
        grid_net = df_day["P_from_grid"] - df_day["P_to_grid"]
        SOC_percent = 100 * df_day["SOC"] / 9.37

        # ==========================================
        # 1. TOP ROW: Energy Flow Plot
        # ==========================================
        ax1 = axs[0, col_idx]
        ax1.bar(hours, battery_net, label="Battery", color="#1b9e77", alpha=0.9, edgecolor='black', hatch='//')
        ax1.bar(hours, grid_net, label="Grid", color="#e7298a", alpha=0.9, edgecolor='black', hatch='xx')
        ax1.bar(hours, -df_day["P_Load"], label="Load", color="#7570b3", alpha=0.6, edgecolor='black')
        ax1.bar(hours, df_day["P_PV"], label="PV", color="#e6ab02", alpha=0.6, edgecolor='black')

        ax1.set_title(scenario["title_energy"], fontsize=18, pad=15)

        # FIX: Reduced labelpad to 6 (keeps it close but not overlapping)
        ax1.set_ylabel("Power [kW]", fontsize=16, labelpad=1)

        # FIX: labelright=False prevents the ghost numbers on the right
        ax1.tick_params(axis='y', labelsize=16, labelleft=True, labelright=False)

        ax1.set_xticks(range(0, 24))
        ax1.set_xticklabels(range(0, 24), fontsize=14)
        ax1.grid(axis='x', linestyle=':', linewidth=0.5)
        ax1.grid(axis='y', linestyle='--', linewidth=0.5)

        # --- Twin Axis (SOC) ---
        ax1b = ax1.twinx()
        ax1b.plot(hours, SOC_percent, label="SOC", color=SOC_ORANGE, linestyle="--", linewidth=2, alpha=0.8)

        # FIX: Reduced labelpad to 10
        ax1b.set_ylabel("State of Charge [%]", fontsize=16, labelpad=1, color=SOC_ORANGE)

        ax1b.set_ylim(0, 100)
        ax1b.set_yticks(list(range(0, 101, 20)))
        ax1b.tick_params(axis='y', labelsize=16, colors=SOC_ORANGE)
        ax1b.axhline(5, color=SOC_ORANGE, linestyle=':', linewidth=1.3)
        ax1b.axhline(95, color=SOC_ORANGE, linestyle=':', linewidth=1.3)
        ax1b.text(0.98, 0.06, '5% (SoC Min)', color=SOC_ORANGE, va='bottom', ha='right', fontsize=12,
                  transform=ax1b.transAxes)
        ax1b.text(0.98, 0.98, '95% (SoC Max)', color=SOC_ORANGE, va='top', ha='right', fontsize=12,
                  transform=ax1b.transAxes)
        ax1b.yaxis.label.set_color(SOC_ORANGE)
        ax1b.grid(False)  # Turn off grid for 2nd axis to avoid clutter

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=15, framealpha=0.9)

        # ==========================================
        # 2. BOTTOM ROW: Financial Plot
        # ==========================================
        ax2 = axs[1, col_idx]
        ax2.plot(hours, df_day["EP_epex"], label="EPEX Price [€/kWh]", color="#1f77b4", marker="o", linewidth=2)
        ax2.plot(hours, df_day["Net Bill"], label="Net Bill [€]", color="#d62728", marker="o", linewidth=2)
        ax2.plot(hours, df_day["Optimization Profit"], label="Optimization Profit [€]",
                 color="#9467bd", marker="o", linewidth=2)

        ax2.set_title(scenario["title_fin"], fontsize=18, pad=10)
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1.2)
        ax2.set_xlabel("Hour", fontsize=16)

        # FIX: Reduced labelpad to 6
        ax2.set_ylabel("€/kWh", fontsize=16, labelpad=6)

        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_xticks(range(0, 24))
        ax2.set_xticklabels(range(0, 24), fontsize=14)
        ax2.grid(True)
        ax2.legend(loc="upper left", fontsize=15)

    # === LAYOUT FINALIZATION ===
    plt.suptitle(f"Daily Energy Flow & Financials for BESS-equipped Scenarios on {date_str}", fontsize=20, y=0.97)

    # FIX: Increased hspace (vertical gap) and wspace (horizontal gap)
    # rect=[left, bottom, right, top]
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    plt.subplots_adjust(hspace=0.15, wspace=0.20)

    output_path = os.path.join(
        visualization_dir_combined_flow_profit,
        f"{date_str}_combined_flow_profit_both_scenarios.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"✅ Daily Combined Energy Flow & Financials chart saved for {date_str}.")
