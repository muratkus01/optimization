import pandas as pd
from itertools import product
import os

# === Path Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "data")

os.makedirs(data_dir, exist_ok=True)

# Step 1: Create full grid
pv_range = range(2, 16, 1)
battery_range = range(2, 16, 1)
inverter_range = range(2, 16, 1)
grid = list(product(pv_range, battery_range, inverter_range))
df = pd.DataFrame(grid, columns=['PV', 'Battery', 'Inverter'])

# Step 2: Calculate ratios
df["PV_to_Battery"] = df["PV"] / df["Battery"]
df["Battery_to_Inverter"] = df["Battery"] / df["Inverter"]
df["PV_to_Inverter"] = df["PV"] / df["Inverter"]

# Step 3: Apply **stricter constraints** (edit if needed)
df_filtered = df[
    (df["PV_to_Battery"] >= 0.5) & (df["PV_to_Battery"] <= 2.0) &
    (df["Battery_to_Inverter"] >= 0.5) & (df["Battery_to_Inverter"] <= 2.0) &
    (df["PV_to_Inverter"] >= 0.5) & (df["PV_to_Inverter"] <= 2.0)
].copy()


# Step 4: Round floats to 2 decimals
float_cols = ["PV_to_Battery", "Battery_to_Inverter", "PV_to_Inverter"]
df_filtered[float_cols] = df_filtered[float_cols].round(2)

# Step 5: Save to CSV
output_path = os.path.join(data_dir, "config.csv")
df_filtered.to_csv(output_path, index=False)

print(f"✅ Config file successfully saved to: {output_path}")
print(f"   Total configurations generated: {len(df_filtered)}")
