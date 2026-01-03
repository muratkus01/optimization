import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.dates as mdates
import calendar


# Create folder for visualizations
os.makedirs("visualizations", exist_ok=True)

# Load Excel file (single sheet with all profiles)
df_raw = pd.read_excel("Haushalt Lastprofil.xls", engine="xlrd")

# Rename columns for clarity
df_raw.columns = [
    "time",
    "winter_saturday", "winter_sunday", "winter_weekday",
    "summer_saturday", "summer_sunday", "summer_weekday",
    "transition_saturday", "transition_sunday", "transition_weekday"
]

# Ensure time column is parsed
if not pd.api.types.is_datetime64_any_dtype(df_raw["time"]):
    df_raw["time"] = pd.date_range("00:00", periods=96, freq="15min")

# Create a mapping to access the correct column by season and day type
season_day_map = {
    ("winter", "weekday"): "winter_weekday",
    ("winter", "saturday"): "winter_saturday",
    ("winter", "sunday"): "winter_sunday",
    ("summer", "weekday"): "summer_weekday",
    ("summer", "saturday"): "summer_saturday",
    ("summer", "sunday"): "summer_sunday",
    ("transition", "weekday"): "transition_weekday",
    ("transition", "saturday"): "transition_saturday",
    ("transition", "sunday"): "transition_sunday"
}

# Define seasons
def get_season(date):
    y = date.year
    if datetime.date(y, 11, 1) <= date <= datetime.date(y, 12, 31) or datetime.date(y, 1, 1) <= date <= datetime.date(y, 3, 20):
        return "winter"
    elif datetime.date(y, 5, 15) <= date <= datetime.date(y, 9, 14):
        return "summer"
    else:
        return "transition"

# Generate calendar and load values
year = 2024
profile_data = []
for day in range(1, 367):  # Leap year
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=day - 1)
    weekday = date.weekday()
    day_type = "weekday" if weekday < 5 else "saturday" if weekday == 5 else "sunday"
    season = get_season(date)
    col_name = season_day_map[(season, day_type)]
    profile_row = df_raw[col_name].values  # 96 quarter-hour values
    profile_data.append({
        "date": date,
        "season": season,
        "day_type": day_type,
        "values": profile_row
    })

# Convert to DataFrame of all quarter-hours
flat_profile = []
for day in profile_data:
    for i, val in enumerate(day["values"]):
        flat_profile.append({
            "datetime": datetime.datetime.combine(day["date"], datetime.time(0)) + datetime.timedelta(minutes=15 * i),
            "value_kW": val / 1000,  # Convert from W to kW
            "season": day["season"],
            "day_type": day["day_type"]
        })

df_profile = pd.DataFrame(flat_profile)

# Add day number column
df_profile["day_of_year"] = df_profile["datetime"].dt.dayofyear

# Dynamization polynomial (4th degree)
def dynamization_factor(t):
    return -3.92e-10 * t**4 + 3.2e-7 * t**3 - 7.02e-5 * t**2 + 2.1e-3 * t + 1.24

# Apply the factor
df_profile["factor"] = df_profile["day_of_year"].apply(dynamization_factor)
df_profile["value_kW_dyn"] = df_profile["value_kW"] * df_profile["factor"]

# Normalize to 1000 kWh/year
energy_kWh = df_profile["value_kW_dyn"].sum() * 0.25  # 15 min intervals = 0.25h
to_1000_kWh = 1000 / energy_kWh
df_profile["value_kW_dyn_norm"] = df_profile["value_kW_dyn"] * to_1000_kWh

# Scale to 3221 kWh/year
df_profile["value_kW_3221"] = df_profile["value_kW_dyn_norm"] * 3.221

# Save results
df_profile.to_csv("profile_3221kWh_quarter_hourly.csv", index=False)
print("Dynamized and scaled profile saved.")

# ========== Aggregate to Hourly Profile ==========
df_profile["hourly_label"] = df_profile["datetime"] - pd.to_timedelta(df_profile["datetime"].dt.minute % 60, unit='m') - pd.to_timedelta(df_profile["datetime"].dt.second, unit='s') - pd.to_timedelta(df_profile["datetime"].dt.microsecond, unit='us')
df_profile["energy_kWh"] = df_profile["value_kW_3221"] * 0.25
df_hourly = df_profile.groupby("hourly_label")["energy_kWh"].sum().reset_index()
df_hourly.rename(columns={"hourly_label": "datetime", "energy_kWh": "value_kWh_3221"}, inplace=True)
df_hourly.to_csv("profile_3221kWh_hourly.csv", index=False)

# ========== Visualization Section ==========

# Plot hourly load profile for entire year colored by season
df_profile["hour"] = df_profile["datetime"].dt.floor("h")
df_hourly_season = df_profile.groupby(["hour", "season"])["energy_kWh"].sum().reset_index()
df_hourly_season.sort_values("hour", inplace=True)

plt.figure(figsize=(20, 6))
for season, color in zip(["winter", "transition", "summer"], ["blue", "orange", "green"]):
    season_data = df_hourly_season[df_hourly_season["season"] == season]

    # Split into continuous segments based on gaps > 1 hour
    time_diffs = season_data["hour"].diff().dt.total_seconds().fillna(0)
    breaks = time_diffs > 3600  # 1 hour
    segment_ids = breaks.cumsum()

    for _, segment in season_data.groupby(segment_ids):
        plt.plot(segment["hour"], segment["energy_kWh"], color=color, label=season.capitalize() if _ == 0 else "")

plt.title("Hourly Load of a German household with 3221 kWh/a in 2024 by Season")
plt.xlabel("DateTime")
plt.ylabel("Energy [kWh]")
plt.legend(title="Season", fontsize=12, title_fontsize=14)
# Set fixed tick locations every 14 days from 01-Jan to 31-Dec
start = pd.Timestamp("2024-01-01")
end = pd.Timestamp("2024-12-31")
biweekly_ticks = pd.date_range(start, end, freq="14D")

plt.gca().set_xticks(biweekly_ticks)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/hourly_load_by_season.png")
plt.close()

# 9 Subplots: Daily load profiles by season and day_type
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=True)
season_list = ["winter", "transition", "summer"]
day_type_list = ["weekday", "saturday", "sunday"]

for i, season in enumerate(season_list):
    for j, day_type in enumerate(day_type_list):
        ax = axes[i, j]
        mask = (df_profile["season"] == season) & (df_profile["day_type"] == day_type)
        df_filtered = df_profile[mask].copy()
        df_filtered["hour_of_day"] = df_filtered["datetime"].dt.hour
        avg_profile = df_filtered.groupby("hour_of_day")["value_kW_3221"].mean()
        ax.plot(avg_profile.index, avg_profile.values)
        ax.set_title(f"{season.capitalize()} - {day_type.capitalize()}")
        ax.set_xticks(range(0, 24))
        ax.set_ylabel("Power (kW)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.yaxis.set_tick_params(labelleft=True)
        if i == 2:
            ax.set_xlabel("Hour")

fig.suptitle("Average Daily Load Profiles by Season and Day Type", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # reserve top 5% of the figure for suptitle
plt.savefig("visualizations/daily_load_profiles.png")
plt.close()


##### HEATMAP #####

# Generate biweekly date ticks
from datetime import datetime

# Create pivot table
pivot = df_profile.pivot_table(
    index=df_profile["datetime"].dt.hour,
    columns=df_profile["datetime"].dt.normalize(),  # normalize to midnight
    values="value_kW_3221",
    aggfunc="mean"
)

# Convert columns to datetime (should be already, but safe)
pivot.columns = pd.to_datetime(pivot.columns)

# Define tick range every 14 days from Jan 1 to Dec 31
date_range = pd.date_range("2024-01-01", "2024-12-31", freq="14D")

# Get tick positions and labels
tick_positions = [i for i, c in enumerate(pivot.columns) if c in date_range]
tick_labels = [c.strftime('%d-%b') for c in pivot.columns if c in date_range]

# Plot heatmap
plt.figure(figsize=(18, 8))
sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={'label': 'Power (kW)'})
plt.title("Hourly Load Heatmap of a German household with 3221 kWh/a in 2024")
plt.ylabel("Hour of Day")
plt.xlabel("Date (dd-mmm)")
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
plt.tight_layout()
plt.savefig("visualizations/hourly_heatmap_year_.png")
plt.close()


# ========== Monthly Total Energy Consumption Plot ==========

# ========== Aggregate to Hourly Profile ==========
df_profile["hourly_label"] = df_profile["datetime"] - pd.to_timedelta(df_profile["datetime"].dt.minute % 60, unit='m') - pd.to_timedelta(df_profile["datetime"].dt.second, unit='s') - pd.to_timedelta(df_profile["datetime"].dt.microsecond, unit='us')
df_profile["energy_kWh"] = df_profile["value_kW_3221"] * 0.25
df_hourly = df_profile.groupby("hourly_label")["energy_kWh"].sum().reset_index()
df_hourly.rename(columns={"hourly_label": "datetime", "energy_kWh": "value_kWh_3221"}, inplace=True)
df_hourly.to_csv("profile_3221kWh_hourly.csv", index=False)

# ========== Monthly Total Energy Consumption Plot ==========
df_profile["month"] = df_profile["datetime"].dt.month
monthly_totals = df_profile.groupby("month")["energy_kWh"].sum()
plt.figure(figsize=(10, 5))
plt.bar(monthly_totals.index, monthly_totals.values, color='skyblue', edgecolor='black')
plt.title("Monthly Total Energy Consumption of a German household with 3221 kWh/a")
plt.xlabel("Month")
plt.ylabel("Energy [kWh]")
plt.grid(axis='y', alpha=0.5)
plt.xticks(ticks=range(1, 13), labels=[calendar.month_abbr[m] for m in range(1, 13)])
plt.tight_layout()
plt.savefig("visualizations/monthly_total_energy.png")
plt.close()
