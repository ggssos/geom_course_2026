from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pumping_path = PROJECT_ROOT / "data" / "raw" / "hw01" / "pumping_test.txt"
wells_path = PROJECT_ROOT / "data" / "processed" / "hw01" / "wells_clean.csv"

hw02_figures = PROJECT_ROOT / "figures" / "hw02"
output_dir = PROJECT_ROOT / "figures" / "hw03"
npz_path = PROJECT_ROOT / "homeworks" / "hw03" / "viz_data.npz"

required_files = [
    hw02_figures / "pressure_over_time.svg",
    hw02_figures / "pressure_by_radius.svg",
    pumping_path,
    wells_path,
]

for path in required_files:
    assert path.exists(), f"Файл не найден: {path}"
    assert path.stat().st_size > 0, f"Файл пустой: {path}"

output_dir.mkdir(parents=True, exist_ok=True)

pumping_test = pd.read_csv(pumping_path, sep="\t")
wells = pd.read_csv(wells_path)

assert pumping_test.shape == (13, 4), "Неверный размер pumping_test"
assert len(wells) == 7, "Ожидалось 7 скважин"
assert not pumping_test.isna().any().any(), "В pumping_test есть пропуски"
assert not wells.isna().any().any(), "В wells есть пропуски"


time_h = pumping_test["time_h"].to_numpy()
boundary_pressure_mpa = pumping_test["boundary_pressure_mpa"].to_numpy()
well_pressure_mpa = pumping_test["well_pressure_mpa"].to_numpy()

well_id = wells["well_id"].to_numpy()
radius_m = wells["radius_m"].to_numpy()
observation_pressure_mpa = wells["pressure_mpa"].to_numpy()

order = np.argsort(radius_m)
well_id = well_id[order]
radius_m = radius_m[order]
observation_pressure_mpa = observation_pressure_mpa [order]

pressure_drop_mpa = boundary_pressure_mpa - well_pressure_mpa
observation_drop_mpa = 12.0 - observation_pressure_mpa


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]


ax_a.plot(
    time_h,
    boundary_pressure_mpa,
    marker="o",
    label="Давление на границе",
)

ax_a.plot(
    time_h,
    well_pressure_mpa,
    marker="o",
    label="Давление в скважине",
)

ax_a.set_title("A. Давление во времени")
ax_a.set_xlabel("Время, ч")
ax_a.set_ylabel("Давление, МПа")
ax_a.legend()
ax_a.grid(alpha=0.3)


scatter = ax_b.scatter(
    radius_m,
    observation_pressure_mpa,
    c=observation_pressure_mpa,
    cmap="viridis",
)

ax_b.set_title("B. Давление по радиусу")
ax_b.set_xlabel("Радиус, м")
ax_b.set_ylabel("Давление, МПа")

colorbar = fig.colorbar(scatter, ax=ax_b)
colorbar.set_label("Давление, МПа")

ax_c.bar(
    well_id.astype(str),
    observation_drop_mpa,
)

ax_c.set_title("C. Падение давления по скважинам")
ax_c.set_xlabel("Скважина")
ax_c.set_ylabel("Падение давления, МПа")

pressure_labels = ["На границе", "В скважине"]

ax_d.boxplot(
    [boundary_pressure_mpa, well_pressure_mpa],
    tick_labels=pressure_labels,
)

ax_d.set_title("D. Распределение давления")
ax_d.set_xlabel("Набор данных")
ax_d.set_ylabel("Давление, МПа")


fig.savefig(output_dir / "pressure_overview.png", dpi=300)
fig.savefig(output_dir / "pressure_overview.svg")
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 5),
    constrained_layout=True,)

ax_box, ax_violin = axes

pressure_data = [
    boundary_pressure_mpa,
    well_pressure_mpa,
]

pressure_labels = ["На границе", "В скважине"]

ax_box.boxplot(
    pressure_data,
    tick_labels=pressure_labels,
)

ax_box.set_title("Boxplot — диаграмма размаха")
ax_box.set_xlabel("Набор данных")
ax_box.set_ylabel("Давление, МПа")

ax_violin.violinplot(
    pressure_data,
    showmeans=True,
    showmedians=True,
)

ax_violin.set_xticks([1, 2])
ax_violin.set_xticklabels(pressure_labels)
ax_violin.set_title("Violin plot — распределение")
ax_violin.set_xlabel("Набор данных")
ax_violin.set_ylabel("Давление, МПа")

fig.savefig(output_dir / "distributions.png", dpi=300)
fig.savefig(output_dir / "distributions.svg")
plt.close(fig)


np.savez(
    npz_path,
    time_h=time_h,
    boundary_pressure_mpa=boundary_pressure_mpa,
    well_pressure_mpa=well_pressure_mpa,
    pressure_drop_mpa=pressure_drop_mpa,
    well_id=well_id.astype(str),
    radius_m=radius_m,
    observation_pressure_mpa=observation_pressure_mpa,
    observation_drop_mpa=observation_drop_mpa,
)


expected_arrays = {
    "time_h": time_h,
    "boundary_pressure_mpa": boundary_pressure_mpa,
    "well_pressure_mpa": well_pressure_mpa,
    "pressure_drop_mpa": pressure_drop_mpa,
    "well_id": well_id.astype(str),
    "radius_m": radius_m,
    "observation_pressure_mpa": observation_pressure_mpa,
    "observation_drop_mpa": observation_drop_mpa,
}

with np.load(npz_path) as saved:
    assert set(saved.files) == set(expected_arrays)

    for name, expected in expected_arrays.items():
        assert saved[name].shape == expected.shape
        np.testing.assert_array_equal(saved[name], expected)