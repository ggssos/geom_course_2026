from pathlib import Path

import matplotlib.pyplot as plt

def plot_pressure_over_time(table, output_dir):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        table["time_h"],
        table["boundary_pressure_mpa"],
        marker="o",
        color="#C2185B",
        label="Давление на границе"
    )

    ax.plot(
        table["time_h"],
        table["well_pressure_mpa"],
        marker="o",
        color="#E6A0B5",
        label="Давление в скважине"
    )

    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Изменение давления во времени")

    ax.grid(alpha=0.3)
    ax.legend()

    min_index = table["well_pressure_mpa"].idxmin()
    min_time = table.loc[min_index, "time_h"]
    min_pressure = table.loc[min_index, "well_pressure_mpa"]

    ax.annotate(
        f"Минимум: {min_pressure:.2f} МПа",
        xy=(min_time, min_pressure),
        xytext=(-15, 35),
        textcoords="offset points",
        ha="right",
        va="bottom",
        arrowprops=dict(arrowstyle="->")
    )

    fig.tight_layout()

    png_path = output_dir / "pressure_over_time.png"
    svg_path = output_dir / "pressure_over_time.svg"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)


def plot_pressure_by_radius(table, output_dir):

    table = table.sort_values("radius_m")

    assert (table["radius_m"] > 0).all()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        table["radius_m"],
        table["pressure_mpa"],
        linestyle="none",
        marker="o",
        color="#C2185B"
    )

    ax.set_xscale("log")

    ax.set_xlabel("Расстояние, м")
    ax.set_ylabel("Давление, МПа")
    ax.set_title("Давление в наблюдательных скважинах")

    ax.grid(alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "pressure_by_radius.png"
    svg_path = output_dir / "pressure_by_radius.svg"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)