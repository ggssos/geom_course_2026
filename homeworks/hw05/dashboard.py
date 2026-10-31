"""Заготовка приложения Dash для ДЗ №5.

Скопируйте файл в dashboard.py и заполните участок TODO.
"""

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "homeworks" / "hw05" / "pressure_long.csv"

table = pd.read_csv(DATA_PATH)
y_min = table["pressure_mpa"].min() - 0.1
y_max = table["pressure_mpa"].max() + 0.1
required_columns = {
    "time_h",
    "radius_m",
    "well_id",
    "pressure_mpa",
}
assert required_columns.issubset(table.columns)

time_values = sorted(table["time_h"].unique())

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Давление и расстояние"),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(time_values) - 1,
            step=1,
            value=0,
            marks={
                index: str(time)
                for index, time in enumerate(time_values)
            },
        ),
        dcc.Graph(id="pressure-graph"),
    ]
)


@app.callback(
    Output("pressure-graph", "figure"),
    Input("time-slider", "value"),
)

def update_graph(time_index):
    selected_time = time_values[time_index]
    current = table[table["time_h"] == selected_time]

    fig = px.line(
        current,
        x="radius_m",
        y="pressure_mpa",
        markers=True
    )

    fig.update_xaxes(
        type="log",
        title="Расстояние, м"
    )

    fig.update_yaxes(
        title="Давление, МПа",
        range=[y_min, y_max]
    )

    fig.update_layout(
        title=f"Давление при времени {selected_time} ч"
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)