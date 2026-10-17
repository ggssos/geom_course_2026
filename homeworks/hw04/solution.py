from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "homeworks" / "hw03" / "viz_data.npz"
output_path = project_root / "homeworks" / "hw04" / "field_data.npz"
figures_path = project_root / "figures" / "hw04"
figures_path.mkdir(parents=True, exist_ok=True)

with np.load(input_path) as data:
    required_keys = {
        "time_h",
        "boundary_pressure_mpa",
        "well_pressure_mpa",
        "pressure_drop_mpa",
        "well_id",
        "radius_m",
        "observation_pressure_mpa",
        "observation_drop_mpa",
    }

    assert required_keys.issubset(data.files)

    time_h = data["time_h"]
    boundary_pressure_mpa = data["boundary_pressure_mpa"]
    well_pressure_mpa = data["well_pressure_mpa"]
    pressure_drop_mpa = data["pressure_drop_mpa"]
    well_id = data["well_id"]
    radius_m = data["radius_m"]
    observation_pressure_mpa = data["observation_pressure_mpa"]
    observation_drop_mpa = data["observation_drop_mpa"]

    assert time_h.shape == (13,)
    assert well_pressure_mpa.shape == (13,)
    assert well_id.shape == (7,)
    assert radius_m.shape == (7,)
    assert observation_pressure_mpa.shape == (7,)
    assert np.all(np.diff(time_h) > 0)
    assert np.all(np.diff(radius_m) > 0)


time_component = well_pressure_mpa - well_pressure_mpa[0]

radius_component = (
    observation_pressure_mpa
    - observation_pressure_mpa[0]
)

pressure_field = (
    well_pressure_mpa[0]
    + time_component[:, np.newaxis]
    + radius_component[np.newaxis, :]
)

assert pressure_field.shape == (13, 7)


assert (radius_m > 0).all()

field_for_plot = pressure_field.T
assert field_for_plot.shape == (7, 13)


vmin = pressure_field.min()
vmax = pressure_field.max()
levels = np.linspace(vmin, vmax, 9)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mesh = axes[0].pcolormesh(
    time_h,
    radius_m,
    field_for_plot,
    shading="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    edgecolors="black",
    linewidth=0.25,
)

axes[0].set_yscale("log")
axes[0].set_xlabel("Время, ч")
axes[0].set_ylabel("Расстояние, м")
axes[0].set_title("A — Тепловая карта")
fig.colorbar(mesh, ax=axes[0], label="Давление, МПа")


filled = axes[1].contourf(
    time_h,
    radius_m,
    field_for_plot,
    levels=levels,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)

lines = axes[1].contour(
    time_h,
    radius_m,
    field_for_plot,
    levels=levels,
    colors="black",
    linewidths=0.6,
)


axes[1].clabel(lines, fmt="%.2f")
axes[1].set_yscale("log")
axes[1].set_xlabel("Время, ч")
axes[1].set_ylabel("Расстояние, м")
axes[1].set_title("B — Заливка и изолинии")
fig.colorbar(filled, ax=axes[1], label="Давление, МПа")

fig.tight_layout()
fig.savefig(figures_path / "pressure_maps.png", dpi=200)
fig.savefig(figures_path / "pressure_maps.svg")
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

lines = ax.contour(
    time_h,
    radius_m,
    field_for_plot,
    levels=levels,
    colors="black",
    linewidths=0.6,
)

ax.clabel(lines, fmt="%.2f")
ax.set_yscale("log")
ax.set_xlabel("Время, ч")
ax.set_ylabel("Расстояние, м")
ax.set_title("Изолинии давления")

fig.tight_layout()
fig.savefig(figures_path / "pressure_isolines.png", dpi=200)
fig.savefig(figures_path / "pressure_isolines.svg")
plt.close(fig)


np.savez(
    output_path,
    time_h=time_h,
    radius_m=radius_m,
    well_id=well_id,
    pressure_field=pressure_field,
    contour_levels=levels,
)

with np.load(output_path) as saved:
    assert {
        "time_h",
        "radius_m",
        "well_id",
        "pressure_field",
        "contour_levels",
    }.issubset(saved.files)

    assert saved["time_h"].shape == (13,)
    assert saved["radius_m"].shape == (7,)
    assert saved["well_id"].shape == (7,)
    assert saved["pressure_field"].shape == (13, 7)
    assert saved["contour_levels"].shape == (9,)

    np.testing.assert_allclose(saved["time_h"], time_h)
    np.testing.assert_allclose(saved["radius_m"], radius_m)
    np.testing.assert_array_equal(saved["well_id"], well_id)
    np.testing.assert_allclose(saved["pressure_field"], pressure_field)
    np.testing.assert_allclose(saved["contour_levels"], levels)