import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm

# === Path Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

results_dir = os.path.join(parent_dir, "results")
input_csv = os.path.join(results_dir, "csv", "sensitivity_yearly_summary.csv")
base_output_dir = os.path.join(results_dir, "visualizations")

os.makedirs(base_output_dir, exist_ok=True)

# === Load Data ===
if not os.path.exists(input_csv):
    print(f"❌ Error: Input file not found at {input_csv}")
    print("   Please run 'sensitivity.py' first to generate the data.")
    exit(1)

# Robust CSV loading (tries comma first, then semicolon)
try:
    df = pd.read_csv(input_csv, sep=',')
    if len(df.columns) < 2:  # If comma failed, try semicolon
        df = pd.read_csv(input_csv, sep=';')
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit(1)

df.columns = df.columns.str.strip()  # Clean whitespace from column names

# === Handle Column Names ===
# Ensure we use the correct CAPEX column name
capex_col = "CAPEX [€]" if "CAPEX [€]" in df.columns else "CAPEX"

# Ensure numeric columns
numeric_cols = ["PV [kW]", "Battery [kWh]", "Inverter [kW]", "Opt. Profit / CAPEX", capex_col, "Optimization Profit"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Add Labels
df["Simple_Label"] = df.apply(
    lambda row: f"{int(row['PV [kW]'])}_{int(row['Battery [kWh]'])}_{int(row['Inverter [kW]'])}",
    axis=1
)


# === Helpers ===
def luminance(rgba):
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def scale_sizes(sizes, scale=150, min_size=50):
    if sizes.max() == sizes.min():
        return np.full(sizes.shape, min_size + scale / 2)
    scaled = (sizes - sizes.min()) / (sizes.max() - sizes.min())
    return scaled * scale + min_size


print(f"📊 Processing {len(df)} configurations...")

# ==============================================================================
# 1. TOP 25 Bar Plot (Ranked by Profit / CAPEX)
# ==============================================================================
df_sorted = df.sort_values("Opt. Profit / CAPEX", ascending=False)
top25 = df_sorted.head(25).copy()

norm = mcolors.Normalize(vmin=top25["Opt. Profit / CAPEX"].min(), vmax=top25["Opt. Profit / CAPEX"].max())
cmap = plt.get_cmap("viridis")
colors = [cmap(norm(val)) for val in top25["Opt. Profit / CAPEX"]]
edgecolors = ['black' if i < 5 else 'none' for i in range(len(top25))]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(top25["Simple_Label"], top25["Opt. Profit / CAPEX"], color=colors, edgecolor=edgecolors)

# Annotate CAPEX inside bars
for i, (bar, capex_val, color) in enumerate(zip(bars, top25[capex_col], colors)):
    height = bar.get_height()
    bar_center = bar.get_x() + bar.get_width() / 2
    text_color = 'white' if luminance(color) < 0.5 else 'black'

    ax.annotate(f"{int(capex_val)} €",
                xy=(bar_center, height / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='center',
                fontsize=8,
                color=text_color,
                fontweight='bold',
                rotation=90)

ax.set_ylabel("Return On Capital Expenditure [€/€]", fontsize=10)
ax.set_xlabel("Configuration (PV_BESS_INVERTER)", fontsize=10)
ax.set_title("Top 25 Configurations by Return On Capital Expenditure (ROCE)", fontsize=12)
ax.set_xticks(range(len(top25)))
ax.set_xticklabels(top25["Simple_Label"], rotation=45, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="ROCE [€/€]")

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "top25_roce.png"), dpi=300)
plt.close()

# ==============================================================================
# 2. TOP 5 Stacked Component Bar Plot
# ==============================================================================
top5 = df_sorted.head(5).copy()

# Make costs negative for visual separation
cost_cols = [
    "Cost for Covering Load from Grid",
    "Cost for Charging Battery from Grid",
    "Annual Fixed Cost",
    "Opportunity Cost of Not PV Feed-in to Grid"
]

for col in cost_cols:
    if col in top5.columns:
        top5.loc[:, col] *= -1

# Map your CSV columns to colors
component_colors = {
    "Earning for Discharge to Grid": "#1f78b4",
    "Earning for PV Feed-in to Grid": "#a6cee3",
    "Savings from Avoided Grid Import Cost": "#33a02c",
    "Cost for Covering Load from Grid": "#fb9a99",
    "Cost for Charging Battery from Grid": "#e31a1c",
    "Annual Fixed Cost": "#ff7f00",
    "Opportunity Cost of Not PV Feed-in to Grid": "#6a3d9a",
}

stack_components = [c for c in component_colors.keys() if c in top5.columns]
stack_data = top5[stack_components]
stack_data.index = top5["Simple_Label"]

# === PLOTTING ===
# Increased figure size for better spacing
ax = stack_data.plot(
    kind='bar',
    stacked=True,
    figsize=(18, 10),
    color=[component_colors[col] for col in stack_components],
    width=0.75  # Slightly wider bars
)

# --- Annotations (External Labels) ---
for i, label in enumerate(stack_data.index):
    # Get metrics
    profit_value = top5.loc[top5["Simple_Label"] == label, "Optimization Profit"].values[0]
    capex_value = top5.loc[top5["Simple_Label"] == label, capex_col].values[0]
    ratio_value = top5.loc[top5["Simple_Label"] == label, "Opt. Profit / CAPEX"].values[0]

    # Calculate Top of positive bars
    pos_cols = [c for c in stack_components if top5.loc[top5["Simple_Label"] == label, c].values[0] > 0]
    bar_top = stack_data.loc[label][pos_cols].sum()

    # 1. Top: Optimization Profit Label
    ax.text(
        i,
        bar_top + 40,  # Higher offset
        f"{int(profit_value):,} €".replace(",", " "),
        ha='center',
        va='bottom',
        fontsize=16,  # Large
        color='darkgreen',
        fontweight='bold',
    )

    # Vertical center for side labels
    bar_center_y = stack_data.loc[label].sum() / 2

    # 2. Left: CAPEX
    ax.text(
        i - 0.42,  # Shifted further left
        bar_center_y,
        f"CAPEX {int(capex_value):,} €".replace(",", " "),
        ha='right',
        va='center',
        fontsize=13,  # Large
        color='dimgray',
        fontweight='bold',
        rotation=90,
        clip_on=False
    )

    # 3. Right: ROCE
    ax.text(
        i + 0.42,  # Shifted further right
        bar_center_y,
        f"ROCE {ratio_value:.3f}",
        ha='left',
        va='center',
        fontsize=13,  # Large
        color='teal',
        fontstyle='italic',
        fontweight='bold',
        rotation=90,
        clip_on=False
    )

# --- Inner Bar Annotations (The Numbers Inside Components) ---
value_threshold = 50
for i, label in enumerate(stack_data.index):
    y_offset_pos = 0.0
    y_offset_neg = 0.0

    for component in stack_components:
        value = stack_data.loc[label][component]

        if value >= 0:
            y = y_offset_pos + value / 2.0
            y_offset_pos += value
        else:
            y = y_offset_neg + value / 2.0
            y_offset_neg += value

        if abs(value) > value_threshold:
            val_str = f"{int(abs(value)):,} €".replace(",", " ")
            label_text = val_str if value >= 0 else f"–{val_str}"

            bg_color = component_colors[component]
            r, g, b = mcolors.to_rgb(bg_color)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'white' if brightness < 0.5 else 'black'

            ax.text(
                i,
                y,
                label_text,
                ha='center',
                va='center',
                fontsize=16,  # INCREASED TO 12 (Very Readable)
                color=text_color
            )

# === AXIS & LEGEND STYLING ===
ax.set_ylabel("Annual Optimization Profit [€]", fontsize=18, color='darkgreen')
ax.set_xlabel("Configuration (PV_BESS_INVERTER)", fontsize=18, labelpad=15)
ax.set_title("Top 5 System Configuration\n(Ranked by Return On Capital Expenditure)", fontsize=20,
             pad=25)

# Larger tick labels
ax.set_xticklabels(stack_data.index, rotation=0, fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# Grid
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

# Legend (Larger & Centered)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "top5_breakdown.png"), dpi=300)
plt.close()

# ==============================================================================
# 3. Bubble Plot (All Data)
# ==============================================================================
plt.figure(figsize=(11, 9))
df_bubble = df.dropna(subset=["Opt. Profit / CAPEX", "Battery [kWh]", "Inverter [kW]", "PV [kW]"])
sizes = scale_sizes(df_bubble["PV [kW]"], scale=200)

norm_bubble = TwoSlopeNorm(vmin=df_bubble["Opt. Profit / CAPEX"].min(),
                           vcenter=df_bubble["Opt. Profit / CAPEX"].median(),
                           vmax=df_bubble["Opt. Profit / CAPEX"].max())

scatter = plt.scatter(
    df_bubble["Inverter [kW]"], df_bubble["Battery [kWh]"],
    s=sizes, c=df_bubble["Opt. Profit / CAPEX"],
    cmap="viridis", edgecolor='k', alpha=0.8, norm=norm_bubble
)

plt.colorbar(scatter, label="ROCE [€/€]")
plt.xlabel("Inverter Size [kW]")
plt.ylabel("Battery Capacity [kWh]")
plt.title("Configuration Space (Size = PV Capacity)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Legend
pvs = sorted(df_bubble["PV [kW]"].unique())
if len(pvs) > 5:
    legend_pvs = np.unique(np.linspace(min(pvs), max(pvs), 5).astype(int))
else:
    legend_pvs = pvs

for pv in legend_pvs:
    plt.scatter([], [], s=scale_sizes(np.array([pv]), scale=200)[0], c='gray', alpha=0.6, label=f"{pv} kW")
plt.legend(title="PV Size", loc="upper left", bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, "bubble_plot_all.png"), dpi=300)
plt.close()

# ==============================================================================
# 4. Heatmaps (Dynamic for each PV size)
# ==============================================================================
heatmap_dir = os.path.join(base_output_dir, "fixed_pv_heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)
cmap_hm = plt.get_cmap("turbo").copy()
cmap_hm.set_bad("#f0f0f0")

for pv in sorted(df["PV [kW]"].unique()):
    sub = df[df["PV [kW]"] == pv].copy()
    if sub.empty: continue

    pivot = sub.pivot_table(index="Battery [kWh]", columns="Inverter [kW]", values="Opt. Profit / CAPEX")

    # Skip if only one row/col (can't render nice heatmap)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        continue

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap=cmap_hm, ax=ax, cbar_kws={'label': 'ROCE [€/€]'}, annot=True, fmt=".2f",
                annot_kws={"size": 8})

    ax.set_title(f"ROCE Heatmap (PV = {pv} kWp)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"heatmap_pv_{int(pv)}.png"), dpi=300)
    plt.close()

print(f"✅ Visualizations generated in: {base_output_dir}")