import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import calendar

# ======================= PATH CONFIGURATION =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

data_dir = os.path.join(parent_dir, "data")
results_base_dir = os.path.join(parent_dir, "results")

# Input Files (Aggregated Results from previous steps)
FILE_OPT = os.path.join(results_base_dir, "FS_optimized_results", "Optimized_Germany_aggregated_results_full.csv")
FILE_UNOPT = os.path.join(results_base_dir, "FS_rule_based_results", "rule_based_Germany_aggregated_results_full.csv")
PRICE_FILE = os.path.join(data_dir, "EPEX_2024.csv")
VIS_BASE = os.path.join(results_base_dir, "visualizations")

# ======================= LOGGING SETUP =========================
log_file = os.path.join(script_dir, 'visualization_generation.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

VIS_FOLDERS = {
    # Heatmaps
    'heatmap': os.path.join(VIS_BASE, 'heatmap'),

    # Barplots
    'daily_combined': os.path.join(VIS_BASE, 'barplots', 'daily_grid_interaction'),

    # Lineplots & events
    'lines': os.path.join(VIS_BASE, 'lines'),
    'hourly_events': os.path.join(VIS_BASE, 'lines', 'hourly_events'),
    'weekly_combined_avg_profile': os.path.join(VIS_BASE, 'lines', 'weekly_combined_avg_profile'),
    'monthly_combined_avg_profile': os.path.join(VIS_BASE, 'lines', 'monthly_combined_avg_profile'),
    'yearly_combined_avg_profile': os.path.join(VIS_BASE, 'lines', 'yearly_combined_avg_profile'),
}

# Ensure all folders exist
for folder in VIS_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

FEDERAL_STATES = [
    "Baden", "Bavaria", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hesse",
    "Lower-Saxony", "Mecklenburg", "Northrhine", "Rhineland-Palatine", "Saarland",
    "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"
]

# ======================= DATA LOADING =========================
def preprocess_df(df):
    df = df.copy()
    df['P_grid_net'] = (df['P_from_grid_net'] - df['P_to_grid_net'])
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    return df

df_opt = pd.read_csv(FILE_OPT, parse_dates=['DateTime'])
df_unopt = pd.read_csv(FILE_UNOPT, parse_dates=['DateTime'])
df_opt = preprocess_df(df_opt)
df_unopt = preprocess_df(df_unopt)

df_price = pd.read_csv(PRICE_FILE, sep=';', parse_dates=['DateTime'], dayfirst=True)
df_opt = pd.merge(df_opt, df_price[['DateTime', 'EP_epex_MWh']], on='DateTime', how='left').rename(columns={'EP_epex_MWh': 'EP_epex'})
df_unopt = pd.merge(df_unopt, df_price[['DateTime', 'EP_epex_MWh']], on='DateTime', how='left').rename(columns={'EP_epex_MWh': 'EP_epex'})

# ======================= UTILITY FUNCTIONS =========================
def reindex_24(series):
    # Utility to always plot full 0-23 x-axis, even for missing hours
    return pd.Series(series, index=np.arange(len(series))).reindex(np.arange(24), fill_value=0)

def hourly_state_event_frequency(df, event_type='import'):
    freq = np.zeros(24)
    for hour in range(24):
        df_hour = df[df['Hour'] == hour]
        count = 0
        for state in FEDERAL_STATES:
            if event_type == 'import':
                count += (df_hour.get(f'P_from_grid_{state}', pd.Series(0)) > 0).sum()
            elif event_type == 'export':
                count += (df_hour.get(f'P_to_grid_{state}', pd.Series(0)) > 0).sum()
        freq[hour] = count
    return reindex_24(freq)

def hourly_state_ch_dc_frequency(df, event_type='charge'):
    freq = np.zeros(24)
    for hour in range(24):
        df_hour = df[df['Hour'] == hour]
        count = 0
        for state in FEDERAL_STATES:
            if event_type == 'charge':
                count += (df_hour.get(f'P_ch_{state}', pd.Series(0)) > 0).sum()
            elif event_type == 'discharge':
                count += (df_hour.get(f'P_dc_{state}', pd.Series(0)) > 0).sum()
        freq[hour] = count
    return reindex_24(freq)

# ======================= FREQUENCY CALCULATION =========================
freq_opt_import = hourly_state_event_frequency(df_opt, 'import')
freq_opt_export = hourly_state_event_frequency(df_opt, 'export')
freq_unopt_import = hourly_state_event_frequency(df_unopt, 'import')
freq_unopt_export = hourly_state_event_frequency(df_unopt, 'export')

freq_opt_charge = hourly_state_ch_dc_frequency(df_opt, 'charge')
freq_opt_discharge = hourly_state_ch_dc_frequency(df_opt, 'discharge')
freq_unopt_charge = hourly_state_ch_dc_frequency(df_unopt, 'charge')
freq_unopt_discharge = hourly_state_ch_dc_frequency(df_unopt, 'discharge')

hours = np.arange(24)

# ======================= PLOTTING & FILE OUTPUT =========================
# 1. Hourly Import/Export Events (Optimized)
plt.figure(figsize=(14,6))
plt.bar(hours - 0.2, freq_opt_import, width=0.4, label='Import Events (Optimized)', color='C0')
plt.bar(hours + 0.2, freq_opt_export, width=0.4, label='Export Events (Optimized)', color='orange')
plt.xlabel('Hour of Day')
plt.xticks(hours)
plt.ylabel('Event Count (federal state-hours)')
plt.title('Hourly Frequency of Import/Export Events (Optimized)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "hourly_grid_import_export_events_optimized.png"))
plt.close()

# 2. Hourly Import/Export Events (Rule-Based)
plt.figure(figsize=(14,6))
plt.bar(hours - 0.2, freq_unopt_import, width=0.4, label='Import Events (Rule-Based)', color='C0')
plt.bar(hours + 0.2, freq_unopt_export, width=0.4, label='Export Events (Rule-Based)', color='orange')
plt.xlabel('Hour of Day')
plt.xticks(hours)
plt.ylabel('Event Count (federal state-hours)')
plt.title('Hourly Frequency of Import/Export Events (Rule-Based)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "hourly_grid_import_export_events_Rule-Based.png"))
plt.close()

# 3. Hourly Charging/Discharging Events (Optimized)
plt.figure(figsize=(18,6))
plt.bar(hours - 0.2, freq_opt_charge, width=0.4, label='Charging Events (Optimized)', color='C0')
plt.bar(hours + 0.2, freq_opt_discharge, width=0.4, label='Discharging Events (Optimized)', color='orange')
plt.xlabel('Hour of Day')
plt.ylabel('Event Count (federal state-hours)')
plt.title('Hourly Frequency of Charging/Discharging Events (Optimized)')
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "hourly_charging_discharging_events_optimized.png"))
plt.close()

# 4. Hourly Charging/Discharging Events (Rule-Based)
plt.figure(figsize=(18,6))
plt.bar(hours - 0.2, freq_unopt_charge, width=0.4, label='Charging Events (Rule-Based)', color='C0')
plt.bar(hours + 0.2, freq_unopt_discharge, width=0.4, label='Discharging Events (Rule-Based)', color='orange')
plt.xlabel('Hour of Day')
plt.ylabel('Event Count (federal state-hours)')
plt.title('Hourly Frequency of Charging/Discharging Events (Rule-Based)')
plt.xticks(hours)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "hourly_charging_discharging_events_Rule-Based.png"))
plt.close()

# ======================= COMBINED EVENTS PLOTS =========================
hours = np.arange(24)

# --- Combined Import/Export Events Subplots ---
fig, axs = plt.subplots(2, 1, figsize=(22, 14))
plt.rcParams.update({'font.size': 18})  # e.g. 18 for general text

# Rule-Based
axs[0].bar(hours - 0.2, freq_unopt_import, width=0.4, label="Import Events", color='#2171b5')
axs[0].bar(hours + 0.2, freq_unopt_export, width=0.4, label="Export Events", color='#ffb300')
axs[0].set_title("(a) Rule-Based BESS Operation", fontsize=24)
axs[0].set_ylabel("Event Count", fontsize=18)
axs[0].tick_params(axis='y', labelsize=16)
axs[0].set_xticks(hours)
axs[0].tick_params(axis='x', labelsize=16)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# Optimized
axs[1].bar(hours - 0.2, freq_opt_import, width=0.4, label="Import Events", color='#2171b5')
axs[1].bar(hours + 0.2, freq_opt_export, width=0.4, label="Export Events", color='#ffb300')
axs[1].set_title("(b) Optimized BESS Operation", fontsize=24)
axs[1].set_ylabel("Event Count", fontsize=18)
axs[1].tick_params(axis='y', labelsize=16)
axs[1].set_xticks(hours)
axs[1].tick_params(axis='x', labelsize=16)
axs[1].set_xlabel("Hour of Day", fontsize=18)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

fig.suptitle(
    "Germany-Wide Hourly Frequency of Grid Import/Export Events\nfor Households with BESS in 2024",
    fontsize=28, y=0.98
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "subplots_hourly_import_export_events.png"))
plt.close(fig)

# --- Combined Charging/Discharging Events Subplots ---
plt.rcParams.update({'font.size': 18})  # e.g. 18 for general text
fig, axs = plt.subplots(2, 1, figsize=(22, 14))

# Rule-Based
axs[0].bar(hours - 0.2, freq_unopt_charge, width=0.4, label="Charging Events", color='#2171b5')
axs[0].bar(hours + 0.2, freq_unopt_discharge, width=0.4, label="Discharging Events", color='#ffb300')
axs[0].set_title("(a) Rule-Based BESS Operation", fontsize=24)
axs[0].set_ylabel("Event Count", fontsize=20)
axs[0].tick_params(axis='y', labelsize=18)
axs[0].tick_params(axis='x', labelsize=18)
axs[0].set_xticks(hours)
axs[0].legend(fontsize=18)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Optimized
axs[1].bar(hours - 0.2, freq_opt_charge, width=0.4, label="Charging Events", color='#2171b5')
axs[1].bar(hours + 0.2, freq_opt_discharge, width=0.4, label="Discharging Events", color='#ffb300')
axs[1].set_title("(b) Optimized BESS Operation", fontsize=24)
axs[1].set_ylabel("Event Count", fontsize=20)
axs[1].tick_params(axis='y', labelsize=18)
axs[1].tick_params(axis='x', labelsize=18)
axs[1].set_xlabel("Hour of Day", fontsize=20)
axs[1].set_xticks(hours)
axs[1].legend(fontsize=18)
axs[1].grid(True, linestyle='--', alpha=0.7)

fig.suptitle("Germany-Wide Hourly Frequency of Charging/Discharging Events \nfor Households with BESS in 2024", fontsize=28)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(VIS_FOLDERS['hourly_events'], "subplots_hourly_charging_discharging_events.png"))
plt.close(fig)


# === VISUALIZATION FUNCTIONS ===
def plot_grid_interaction_heatmap(
    df,
    value_col,
    title,
    fname,
    colormap='coolwarm',
    multiply_by=-1,
    center_zero=True,
    cbar_label=None,
    vmin=None,
    vmax=None
):
    heatmap_data = df.pivot_table(
        index='Hour',
        columns='DayOfYear',
        values=value_col,
        aggfunc='mean'
    ).sort_index().sort_index(axis=1) * multiply_by

    # === Compute vmin and vmax ===
    data_min = heatmap_data.min().min()
    data_max = heatmap_data.max().max()

    # Round limits outward to nearest integer
    vmin = vmin if vmin is not None else np.floor(data_min)
    vmax = vmax if vmax is not None else np.ceil(data_max)

    plt.figure(figsize=(18, 8))

    # === Plot heatmap ===
    ax = sns.heatmap(
        heatmap_data,
        cmap=colormap,
        center=0 if center_zero else None,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            'label': cbar_label if cbar_label else f'{value_col} [MWh]',
            'shrink': 0.9
        }
    )

    # === Customize colorbar ===
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label if cbar_label else f'{value_col} [MWh]', fontsize=13)

    # Force integer ticks within range
    tick_vals = list(range(int(vmin), int(vmax) + 1))
    cbar.set_ticks(tick_vals)

    # === Axis labels and ticks ===
    plt.title(title, fontsize=16)
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('Hour of Day', fontsize=12)
    plt.yticks(ticks=np.arange(24), labels=np.arange(24), fontsize=10)

    # === Format X-axis ticks ===
    tick_every_n = 14
    tick_indices = heatmap_data.columns[::tick_every_n]
    tick_labels = pd.to_datetime(tick_indices, format='%j').strftime('%b-%d')
    plt.xticks(
        ticks=range(0, len(tick_indices) * tick_every_n, tick_every_n),
        labels=tick_labels,
        fontsize=10,
        rotation=45
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    logging.info(f"Heatmap saved to {fname}")

# === Combined Barplots (all hours in both axes) ===
def plot_bar_daily_net_grid_with_epex(df_unopt, df_opt, day, output_folder):
    # Enforce 0-23 for both
    day_unopt = df_unopt[df_unopt['DayOfYear'] == day].set_index('Hour').reindex(range(24), fill_value=0).reset_index()
    day_opt = df_opt[df_opt['DayOfYear'] == day].set_index('Hour').reindex(range(24), fill_value=0).reset_index()
    date = (day_unopt['DateTime'].iloc[0] if 'DateTime' in day_unopt.columns and not day_unopt['DateTime'].isnull().all()
            else f"Day {int(day)}")
    date_str = date.strftime('%d.%m.%Y') if hasattr(date, "strftime") else str(date)
    prices = day_opt['EP_epex'] if 'EP_epex' in day_opt.columns else None

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(
        f"Germany-Wide Net Grid Exchange for Households with BESS \n and Market Prices on {date_str} (+ Import, – Export)",
        fontsize=18, y=0.98
    )

    # Top: Combined Barplot
    axs[0].bar(day_unopt['Hour'] - 0.15, day_unopt['P_grid_net'], width=0.3, color='gray', alpha=0.7, label='Rule-Based BESS Operation')
    axs[0].bar(day_opt['Hour'] + 0.15, day_opt['P_grid_net'], width=0.3, color='#22b573', alpha=0.7, label='Optimized BESS Operation')
    axs[0].axhline(0, color='black', linewidth=1)
    axs[0].grid(axis='y', linestyle='--', alpha=0.6)
    axs[0].set_ylabel('Net Grid Exchange [MWh]', fontsize=14)
    axs[0].set_title("(a) Daily Grid Exchange", fontsize=14)
    axs[0].legend(fontsize=14)
    axs[0].set_xticks(np.arange(24))
    axs[0].tick_params(axis='both', labelsize=11)

    # Bottom: EPEX Spot Price
    if prices is not None:
        axs[1].plot(day_opt['Hour'], prices, marker='o', color='#0066cc', label='EPEX Spot Price [€/MWh]')
        axs[1].set_ylabel('EPEX Spot Price [€/MWh]', fontsize=12)
        axs[1].set_xlabel('Hour of Day', fontsize=14)
        axs[1].set_title("(b) Corresponding EPEX Spot Market Prices", fontsize=14)
        axs[1].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        axs[1].legend(fontsize=14)
        axs[1].set_xticks(np.arange(24))
        axs[1].tick_params(axis='both', labelsize=11)
    else:
        axs[1].text(0.5, 0.5, "EPEX price data not found", ha='center', va='center', fontsize=12)
        axs[1].set_axis_off()

    plt.tight_layout()
    fname = os.path.join(output_folder, f"combined_day_{int(day):03d}_with_epex.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Combined daily barplot with EPEX (Day {int(day)}) saved to {fname}")


def plot_combined_weekly_avg_profiles(df_opt, df_unopt, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for week in sorted(df_opt['DateTime'].dt.isocalendar().week.unique()):
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

        # --- Export subplot ---
        df_opt_week = df_opt[df_opt['DateTime'].dt.isocalendar().week == week]
        df_unopt_week = df_unopt[df_unopt['DateTime'].dt.isocalendar().week == week]
        avg_export_opt = -reindex_24(df_opt_week.groupby('Hour')['P_to_grid_net'].mean())
        avg_export_unopt = -reindex_24(df_unopt_week.groupby('Hour')['P_to_grid_net'].mean())

        axs[0].plot(avg_export_unopt.index, avg_export_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
        axs[0].plot(avg_export_opt.index, avg_export_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
        axs[0].set_ylabel('Average Export [MWh]', fontsize=12)
        axs[0].set_title(f'(a) Export Profile', fontsize=14)
        axs[0].legend(fontsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_xticks(np.arange(24))
        axs[0].tick_params(axis='both', labelsize=11)

        # --- Import subplot ---
        avg_import_opt = reindex_24(df_opt_week.groupby('Hour')['P_from_grid_net'].mean())
        avg_import_unopt = reindex_24(df_unopt_week.groupby('Hour')['P_from_grid_net'].mean())

        axs[1].plot(avg_import_unopt.index, avg_import_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
        axs[1].plot(avg_import_opt.index, avg_import_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
        axs[1].set_ylabel('Average Import [MWh]', fontsize=12)
        axs[1].set_title(f'(b) Import Profile', fontsize=14)
        axs[1].set_xlabel('Hour of Day', fontsize=11)
        axs[1].legend(fontsize=12)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_xticks(np.arange(24))
        axs[1].tick_params(axis='both', labelsize=11)
        week_dates = df_opt[df_opt['DateTime'].dt.isocalendar().week == week]['DateTime']
        if not week_dates.empty:
            representative_date = week_dates.min()
            month_name = calendar.month_name[representative_date.month]
        else:
            month_name = "Unknown"

        fig.suptitle(
            f"Germany-Wide Weekly Averaged Hourly Grid Profiles \nfor Households with BESS \n (Week {week} - {month_name} 2024)",
            fontsize=18, y=0.97)

        plt.tight_layout()
        fname = os.path.join(out_folder, f'week_{week:02d}_combined_avg_profile.png')
        plt.savefig(fname)
        plt.close()
        logging.info(f"Weekly combined avg profile saved to {fname}")


def plot_combined_monthly_avg_profiles(df_opt, df_unopt, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for month in sorted(df_opt['DateTime'].dt.month.unique()):
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

        # --- Export subplot ---
        df_opt_month = df_opt[df_opt['DateTime'].dt.month == month]
        df_unopt_month = df_unopt[df_unopt['DateTime'].dt.month == month]
        avg_export_opt = -reindex_24(df_opt_month.groupby('Hour')['P_to_grid_net'].mean())
        avg_export_unopt = -reindex_24(df_unopt_month.groupby('Hour')['P_to_grid_net'].mean())

        axs[0].plot(avg_export_unopt.index, avg_export_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
        axs[0].plot(avg_export_opt.index, avg_export_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
        axs[0].set_ylabel('Average Export [MWh]', fontsize=13)
        axs[0].set_title(f'(a) Export Profile', fontsize=15)
        axs[0].legend(fontsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_xticks(np.arange(24))
        axs[0].tick_params(axis='both', labelsize=11)

        # --- Import subplot ---
        avg_import_opt = reindex_24(df_opt_month.groupby('Hour')['P_from_grid_net'].mean())
        avg_import_unopt = reindex_24(df_unopt_month.groupby('Hour')['P_from_grid_net'].mean())

        axs[1].plot(avg_import_unopt.index, avg_import_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
        axs[1].plot(avg_import_opt.index, avg_import_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
        axs[1].set_ylabel('Average Import [MWh]', fontsize=13)
        axs[1].set_title(f'(b) Import Profile', fontsize=15)
        axs[1].set_xlabel('Hour of Day', fontsize=13)
        axs[1].legend(fontsize=12)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_xticks(np.arange(24))
        axs[1].tick_params(axis='both', labelsize=11)
        fig.suptitle(
            f"Germany-Wide Monthly Averaged Hourly Grid Profiles \n for Households with BESS \n ({calendar.month_name[month]} 2024)",
            fontsize=18, y=0.97)

        plt.tight_layout()
        fname = os.path.join(out_folder, f'month_{month:02d}_combined_avg_profile.png')
        plt.savefig(fname)
        plt.close()
        logging.info(f"Monthly combined avg profile saved to {fname}")

def plot_combined_yearly_avg_profiles(df_opt, df_unopt, out_folder):
    """
    Plots combined average hourly import and export for the whole year,
    comparing optimized and Rule-Based.
    """
    import os
    os.makedirs(out_folder, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

    # --- Export ---
    avg_export_opt = -reindex_24(df_opt.groupby('Hour')['P_to_grid_net'].mean())
    avg_export_unopt = -reindex_24(df_unopt.groupby('Hour')['P_to_grid_net'].mean())
    axs[0].plot(avg_export_unopt.index, avg_export_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
    axs[0].plot(avg_export_opt.index, avg_export_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
    axs[0].set_ylabel('Average Export [MWh]', fontsize=11)
    axs[0].set_title('(a) Export Profile', fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_xticks(np.arange(24))
    axs[0].tick_params(axis='both', labelsize=11)

    # --- Import ---
    avg_import_opt = reindex_24(df_opt.groupby('Hour')['P_from_grid_net'].mean())
    avg_import_unopt = reindex_24(df_unopt.groupby('Hour')['P_from_grid_net'].mean())
    axs[1].plot(avg_import_unopt.index, avg_import_unopt.values, color='orange', marker='o', label='Rule-Based BESS Operation')
    axs[1].plot(avg_import_opt.index, avg_import_opt.values, color='#22b573', marker='o', label='Optimized BESS Operation')
    axs[1].set_ylabel('Average Import [MWh]', fontsize=11)
    axs[1].set_title('(b) Import Profile', fontsize=14)
    axs[1].set_xlabel('Hour of Day', fontsize=13)
    axs[1].legend(fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_xticks(np.arange(24))
    axs[1].tick_params(axis='both', labelsize=11)

    # Add suptitle
    fig.suptitle("Germany-Wide Yearly Averaged Hourly Grid Exchange Profiles in 2024\n"
                 "for Households with BESS", fontsize=18, y=0.97)

    plt.tight_layout()
    fname = os.path.join(out_folder, 'yearly_combined_avg_profile.png')
    plt.savefig(fname)
    plt.close()
    logging.info(f"Yearly combined avg profile saved to {fname}")


# === MAIN VISUALIZATION EXECUTION ===

# 1. High-level overview

# Net Grid Exchange (Optimized)
plot_grid_interaction_heatmap(
    df_opt, 'P_grid_net',
    "Germany-Wide Hourly Net Grid Exchange \nOptimized BESS Operation (2024) \n(- Export, + Import)",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_P_grid_net_optimized.png"),
    colormap='seismic',
    multiply_by=0.001,
    center_zero=True,
    cbar_label="Net Grid Exchange [GWh]"
)

# Net Grid Exchange (Rule-Based)
plot_grid_interaction_heatmap(
    df_unopt, 'P_grid_net',
    "Germany-Wide Hourly Net Grid Exchange \nRule-Based BESS Operation (2024) \n(- Export, + Import)",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_P_grid_net_Rule-Based.png"),
    colormap='seismic',
    multiply_by=0.001,
    center_zero=True,
    cbar_label="Net Grid Exchange [GWh]"
)

# Grid Import (Optimized)
plot_grid_interaction_heatmap(
    df_opt, 'P_from_grid_net',
    "Optimized: Grid Import",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_import_optimized.png"),
    colormap='YlGnBu',
    multiply_by=0.001,
    center_zero=False,
    cbar_label="Imported Energy [GWh]"
)

# Grid Export (Optimized, flipped sign)
plot_grid_interaction_heatmap(
    df_opt, 'P_to_grid_net',
    "Optimized: Grid Export",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_export_optimized.png"),
    colormap='OrRd',
    multiply_by=-0.001,
    center_zero=True,
    cbar_label="Exported Energy [GWh]"
)

# Grid Import (Rule-Based)
plot_grid_interaction_heatmap(
    df_unopt, 'P_from_grid_net',
    "Rule-Based: Grid Import",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_import_Rule-Based.png"),
    colormap='YlGnBu',
    multiply_by=0.001,
    center_zero=False,
    cbar_label="Imported Energy [GWh]"
)

# Grid Export (Rule-Based, flipped sign)
plot_grid_interaction_heatmap(
    df_unopt, 'P_to_grid_net',
    "Rule-Based: Grid Export",
    os.path.join(VIS_FOLDERS['heatmap'], "heatmap_export_Rule-Based.png"),
    colormap='OrRd',
    multiply_by=-0.001,
    center_zero=True,
    cbar_label="Exported Energy [GWh]"
)

# 2. Average Profiles
plot_combined_monthly_avg_profiles(df_opt, df_unopt, VIS_FOLDERS['monthly_combined_avg_profile'])
plot_combined_weekly_avg_profiles(df_opt, df_unopt, VIS_FOLDERS['weekly_combined_avg_profile'])
plot_combined_yearly_avg_profiles(df_opt, df_unopt, VIS_FOLDERS['yearly_combined_avg_profile'])

for day in sorted(df_opt['DayOfYear'].unique()):
        plot_bar_daily_net_grid_with_epex(df_unopt, df_opt, day, VIS_FOLDERS['daily_combined'])

print("✅ All visualizations saved in their respective subfolders in ./visualizations")
