from pathlib import Path
from data_utils import (read_wells, read_layers, read_pumping_test, DatasetInfo)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

wells_path = ROOT / "data" / "raw" / "hw01" / "wells.csv"
layers_path = ROOT / "data" / "raw" / "hw01" / "layers.xlsx"
pumping_test_path = ROOT / "data" / "raw" / "hw01" / "pumping_test.txt"

wells = read_wells(wells_path)
layers = read_layers(layers_path)
pumping_test = read_pumping_test(pumping_test_path)

wells_info = DatasetInfo("wells", wells)
layers_info = DatasetInfo("layers", layers)
pumping_test_info = DatasetInfo("pumping_test", pumping_test)

print(wells_info.describe())
print(layers_info.describe())
print(pumping_test_info.describe())

print(wells.isna().sum())
wells_clean = wells.dropna(subset=["pressure_mpa"]).copy()

assert (wells_clean["radius_m"] > 0).all()
assert (layers["thickness_m"] > 0).all()
assert layers["porosity_fraction"].between(0, 1).all()

pressure_pa = wells_clean["pressure_mpa"] * 1_000_000
pressure_difference_mpa = 12 - wells_clean["pressure_mpa"]
relative_change_percent = pressure_difference_mpa / 12 * 100

theta_rad = wells_clean["azimuth_deg"] * np.pi / 180
x_check = wells_clean["radius_m"] * np.cos(theta_rad)
y_check = wells_clean["radius_m"] * np.sin(theta_rad)

print(np.allclose(x_check, wells_clean["x_m"], atol=0.02))
print(np.allclose(y_check, wells_clean["y_m"], atol=0.02))

log_radius = np.log(wells_clean["radius_m"])
decay = np.exp(-pumping_test["time_h"] / 36)

print("pressure_mpa:", wells_clean["pressure_mpa"].min(),
      wells_clean["pressure_mpa"].max(),
      wells_clean["pressure_mpa"].mean())

print("radius_m:", wells_clean["radius_m"].min(),
      wells_clean["radius_m"].max(),
      wells_clean["radius_m"].mean())

print("thickness_m:", layers["thickness_m"].min(),
      layers["thickness_m"].max(),
      layers["thickness_m"].mean())

print("porosity_fraction:", layers["porosity_fraction"].min(),
      layers["porosity_fraction"].max(),
      layers["porosity_fraction"].mean())

pressure = wells_clean["pressure_mpa"].to_numpy()

print("first:", pressure[0])
print("last:", pressure[-1])
print("first 3:", pressure[:3])
print("elements 3-5:", pressure[2:5])
print("every second:", pressure[::2])

mask = pressure < pressure.mean()
selected = pressure[mask]

print("below mean:", selected)

center_x = wells_clean["x_m"].iloc[0]
center_y = wells_clean["y_m"].iloc[0]

distance = np.sqrt(
    (wells_clean["x_m"] - center_x) ** 2 + (wells_clean["y_m"] - center_y) ** 2
)

print(wells_clean[distance > 100])

pressure_matrix = pumping_test[["boundary_pressure_mpa", "well_pressure_mpa"]].to_numpy()

print("matrix ndim:", pressure_matrix.ndim)
print("matrix shape:", pressure_matrix.shape)
print("matrix size:", pressure_matrix.size)
print("matrix dtype:", pressure_matrix.dtype)

experiment_1 = pressure_matrix
experiment_2 = pressure_matrix + 0.05

pressure_cube = np.stack([experiment_1, experiment_2], axis=0)

print("cube shape:", pressure_cube.shape)
print("cube:", pressure_cube)
print("first experiment:", pressure_cube[0])
print("second experiment:", pressure_cube[1])

pressure_transposed = pressure_matrix.T

print("transposed shape:", pressure_transposed.shape)
print(pressure_transposed)

flat = pressure_cube.reshape(-1)
restored = flat.reshape(pressure_cube.shape)

print("flat shape:", flat.shape)
print("restored shape:", restored.shape)

assert np.allclose(restored, pressure_cube)

processed_dir = ROOT / "data" / "processed" / "hw01"
exports_dir = ROOT / "exports" / "hw01"

processed_dir.mkdir(parents=True, exist_ok=True)
exports_dir.mkdir(parents=True, exist_ok=True)

wells_clean.to_csv(processed_dir / "wells_clean.csv", index=False)

table_summary = pd.DataFrame({
    "dataset": ["pressure_mpa", "radius_m", "thickness_m", "porosity_fraction"],
    "min": [
        wells_clean["pressure_mpa"].min(),
        wells_clean["radius_m"].min(),
        layers["thickness_m"].min(),
        layers["porosity_fraction"].min()
    ],
    "max": [
        wells_clean["pressure_mpa"].max(),
        wells_clean["radius_m"].max(),
        layers["thickness_m"].max(),
        layers["porosity_fraction"].max()
    ],
    "mean": [
        wells_clean["pressure_mpa"].mean(),
        wells_clean["radius_m"].mean(),
        layers["thickness_m"].mean(),
        layers["porosity_fraction"].mean()
    ]
})

table_summary.to_excel(processed_dir / "table_summary.xlsx", index=False)

np.save(processed_dir / "pressure_matrix.npy", pressure_matrix)

np.savez(processed_dir / "pressure_cube.npz", pressure_cube=pressure_cube)

with open(exports_dir / "results.txt", "w", encoding="utf-8") as file:
    file.write(f"pressure_matrix shape: {pressure_matrix.shape}\n")
    file.write(f"pressure_cube shape: {pressure_cube.shape}\n")
    file.write(f"pressure min: {pressure.min()}\n")
    file.write(f"pressure max: {pressure.max()}\n")
    file.write(f"pressure mean: {pressure.mean()}\n")

loaded_matrix = np.load(processed_dir / "pressure_matrix.npy")

loaded_cube_data = np.load(processed_dir / "pressure_cube.npz")

loaded_cube = loaded_cube_data["pressure_cube"]

assert np.allclose(loaded_matrix, pressure_matrix)
assert np.allclose(loaded_cube, pressure_cube)