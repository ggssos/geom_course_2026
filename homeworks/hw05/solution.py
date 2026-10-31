from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "homeworks" / "hw04" / "field_data.npz"
SVG_PATH = ROOT / "figures" / "hw04" / "pressure_maps.svg"

CSV_PATH = ROOT / "homeworks" / "hw05" / "pressure_long.csv"
GIF_PATH = ROOT / "figures" / "hw05" / "pressure_animation.gif"


with np.load(DATA_PATH) as data:
    required_keys = {
        "time_h",
        "radius_m",
        "well_id",
        "pressure_field",
        "contour_levels",
    }

    assert required_keys.issubset(data.files)

    time_h = data["time_h"]
    radius_m = data["radius_m"]
    well_id = data["well_id"]
    pressure_field = data["pressure_field"]
    contour_levels = data["contour_levels"]


assert time_h.shape == (13,)
assert radius_m.shape == (7,)
assert well_id.shape == (7,)
assert pressure_field.shape == (13, 7)
assert contour_levels.shape == (9,)

assert np.all(np.diff(time_h) > 0)
assert np.all(np.diff(radius_m) > 0)

assert SVG_PATH.exists()
assert SVG_PATH.stat().st_size > 0


time_grid, radius_grid = np.meshgrid(
    time_h,
    radius_m,
    indexing="ij",
)

time_index, radius_index = np.indices(
    pressure_field.shape
)

well_names_long = well_id[
    radius_index.reshape(-1)
]

long_table = pd.DataFrame(
    {
        "time_h": time_grid.reshape(-1),
        "radius_m": radius_grid.reshape(-1),
        "well_id": well_names_long,
        "pressure_mpa": pressure_field.reshape(-1),
    }
)

assert long_table.shape == (91, 4)
assert long_table["time_h"].nunique() == 13
assert long_table["radius_m"].nunique() == 7
assert not long_table.isna().any().any()


long_table.to_csv(CSV_PATH, index=False)

loaded_table = pd.read_csv(CSV_PATH)

assert loaded_table.shape == (91, 4)
assert loaded_table["time_h"].nunique() == 13
assert loaded_table["radius_m"].nunique() == 7
assert not loaded_table.isna().any().any()


fig, ax = plt.subplots()

line, = ax.plot(
    radius_m,
    pressure_field[0],
    marker="o",
)

ax.set_xscale("log")

ax.set_xlabel("Расстояние, м")
ax.set_ylabel("Давление, МПа")

margin = 0.1
y_min = pressure_field.min() - margin
y_max = pressure_field.max() + margin

ax.set_ylim(y_min, y_max)


def update(frame_index):
    current_pressure = pressure_field[frame_index]

    line.set_data(radius_m, current_pressure)

    title = ax.set_title(
        f"Давление, время = {time_h[frame_index]} ч"
    )

    return line, title


animation = FuncAnimation(
    fig,
    update,
    frames=13,
    interval=500,
    blit=False,
)

animation.save(
    GIF_PATH,
    writer=PillowWriter(fps=2),
)

plt.close(fig)