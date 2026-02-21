from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hurricane_forecasting.config import SAFFIR_SIMPSON_THRESHOLDS, TrainingConfig
from hurricane_forecasting.data import load_raw_data
from hurricane_forecasting.features import prepare_inference_features


st.set_page_config(
    page_title="Hurricane Forecast Decision Cockpit",
    page_icon="🌀",
    layout="wide",
)


@st.cache_data
def load_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Model manifest not found. Run `python scripts/train_models.py` first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model_registry(items: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for model_name, model_path in items:
        p = Path(model_path)
        if p.exists():
            loaded[model_name] = joblib.load(p)
    return loaded


@st.cache_data
def load_base_data() -> tuple[pd.DataFrame, pd.Series]:
    cfg = TrainingConfig()
    return load_raw_data(cfg=cfg)


@st.cache_data
def build_engineered_frame(feature_order_tuple: tuple[str, ...]) -> pd.DataFrame:
    raw_features, _ = load_base_data()
    return prepare_inference_features(raw_features, list(feature_order_tuple))


def risk_band(predicted_wind: float) -> tuple[str, str]:
    for low, high, label, color in SAFFIR_SIMPSON_THRESHOLDS:
        if low <= predicted_wind <= high:
            return label, color
    return "Unclassified", "#6c757d"


def _sorted_suffix_columns(columns: list[str], prefix: str) -> list[str]:
    cols = [col for col in columns if col.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def latest_column(columns: list[str], prefix: str) -> str | None:
    cols = _sorted_suffix_columns(columns, prefix)
    return cols[-1] if cols else None


def _basin_from_row(row: pd.Series, basin_cols: list[str]) -> str:
    if not basin_cols:
        return "Unknown"
    active = max(basin_cols, key=lambda c: float(row.get(c, 0.0)))
    return active.replace("basin_", "").replace("_0", "")


def make_single_prediction(raw_row: pd.Series, model: Any, feature_order: list[str]) -> tuple[float, pd.DataFrame]:
    raw_df = pd.DataFrame([raw_row])
    X = prepare_inference_features(raw_df, feature_order)
    prediction = float(model.predict(X)[0])
    return prediction, X


def candidate_spread(raw_row: pd.Series, candidate_models: dict[str, Any], feature_order: list[str]) -> dict[str, float] | None:
    if not candidate_models:
        return None

    raw_df = pd.DataFrame([raw_row])
    X = prepare_inference_features(raw_df, feature_order)

    values = [float(model.predict(X)[0]) for model in candidate_models.values()]
    if not values:
        return None

    s = pd.Series(values)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def apply_scenario(raw_row: pd.Series, columns: list[str]) -> pd.Series:
    row = raw_row.copy()

    wind_col = latest_column(columns, "WMO_WIND_")
    pressure_col = latest_column(columns, "WMO_PRESSURE_")
    dist_col = latest_column(columns, "DISTANCE_TO_LAND_")
    speed_col = latest_column(columns, "STORM_TRANSLATION_SPEED_")

    st.subheader("Scenario Testing")
    st.caption("What-if adjustments on the current storm snapshot.")

    c1, c2, c3, c4 = st.columns(4)
    wind_delta = c1.slider("Wind Δ (knots)", -30, 30, 0)
    pressure_delta = c2.slider("Pressure Δ (hPa)", -40, 40, 0)
    distance_delta = c3.slider("Distance-to-Land Δ (km)", -300, 300, 0)
    speed_delta = c4.slider("Translation Speed Δ (km/h)", -30, 30, 0)

    if wind_col:
        row[wind_col] = float(row[wind_col]) + wind_delta
    if pressure_col:
        row[pressure_col] = float(row[pressure_col]) + pressure_delta
    if dist_col:
        row[dist_col] = max(0.0, float(row[dist_col]) + distance_delta)
    if speed_col:
        row[speed_col] = max(0.0, float(row[speed_col]) + speed_delta)

    return row


def render_key_drivers(model: Any, feature_order: list[str]) -> None:
    st.subheader("Key Drivers")

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(
            {
                "feature": feature_order,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    else:
        importance_path = PROJECT_ROOT / "reports" / "feature_importance.csv"
        if not importance_path.exists():
            st.info("Feature importance is not available for the selected model.")
            return
        imp = pd.read_csv(importance_path)

    if imp.empty:
        st.info("Feature-importance data is empty.")
        return

    top = imp.head(12).iloc[::-1]
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Drivers",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def build_track_dataframe(raw_row: pd.Series) -> pd.DataFrame:
    columns = list(raw_row.index)
    lat_cols = _sorted_suffix_columns(columns, "LATITUDE_")

    points: list[dict[str, float | int | str]] = []
    for lat_col in lat_cols:
        suffix = lat_col.split("_")[-1]
        lon_col = f"LONGITUDE_{suffix}"
        speed_col = f"STORM_TRANSLATION_SPEED_{suffix}"

        if lon_col not in raw_row.index:
            continue

        lat = raw_row.get(lat_col)
        lon = raw_row.get(lon_col)
        speed = raw_row.get(speed_col)

        if pd.isna(lat) or pd.isna(lon):
            continue

        timestep = int(suffix)
        points.append(
            {
                "timestep": timestep,
                "latitude": float(lat),
                "longitude": float(lon),
                "translation_speed_kmh": float(speed) if pd.notna(speed) else 0.0,
            }
        )

    if not points:
        return pd.DataFrame()

    track = pd.DataFrame(points).sort_values("timestep")
    max_t = int(track["timestep"].max())
    track["hours_from_now"] = (track["timestep"] - max_t) * 3
    track["time_label"] = track["hours_from_now"].apply(lambda h: "Now" if h == 0 else f"{int(h)}h")
    return track


def render_storm_map(raw_row: pd.Series | None) -> None:
    st.subheader("Storm Track Map")

    if raw_row is None:
        st.info("Storm map appears here once a storm row is selected.")
        return

    track = build_track_dataframe(raw_row)
    if track.empty:
        st.info("No valid latitude/longitude track points found for this row.")
        return

    fig = px.scatter_geo(
        track,
        lat="latitude",
        lon="longitude",
        color="translation_speed_kmh",
        hover_name="time_label",
        hover_data={
            "timestep": True,
            "latitude": ":.2f",
            "longitude": ":.2f",
            "translation_speed_kmh": ":.2f",
        },
        color_continuous_scale="Turbo",
        projection="natural earth",
        title="Track Colored by Translation Speed (km/h)",
    )

    fig.add_trace(
        go.Scattergeo(
            lat=track["latitude"],
            lon=track["longitude"],
            mode="lines",
            line={"width": 2.5, "color": "#1f2a44"},
            name="Track",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        geo={
            "showland": True,
            "landcolor": "#f2efe9",
            "showocean": True,
            "oceancolor": "#dff3ff",
            "showcoastlines": True,
            "coastlinecolor": "#5b6b7a",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def render_prediction_cards(prediction: float, spread: dict[str, float] | None, model_name: str) -> None:
    label, color = risk_band(prediction)

    st.markdown(
        f"""
        <div style=\"padding:16px;border-radius:12px;border:1px solid #ddd;background:#fafafa\">
          <h3 style=\"margin:0 0 8px 0\">Forecast Output</h3>
          <p style=\"font-size:28px;margin:0\"><strong>{prediction:.1f} knots</strong></p>
          <p style=\"margin:6px 0 0 0\">Risk Band: <span style=\"color:{color};font-weight:700\">{label}</span></p>
          <p style=\"margin:6px 0 0 0\">Model used: <strong>{model_name}</strong></p>
          <p style=\"margin:10px 0 0 0;color:#555\">Plain-language summary: Current conditions indicate {label.lower()} intensity in ~24h.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if spread:
        st.markdown(
            f"""
            <div style=\"padding:16px;border-radius:12px;border:1px solid #ddd;background:#fcfcff;margin-top:10px\">
              <h4 style=\"margin:0 0 8px 0\">Confidence / Uncertainty Proxy</h4>
              <p style=\"margin:0\">Candidate model spread: <strong>{spread['min']:.1f} - {spread['max']:.1f} knots</strong></p>
              <p style=\"margin:4px 0 0 0\">Std Dev: <strong>{spread['std']:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_downloads(single_payload: dict[str, Any], batch_df: pd.DataFrame | None = None) -> None:
    st.subheader("Download Report")

    json_bytes = json.dumps(single_payload, indent=2).encode("utf-8")
    st.download_button(
        label="Download Single Prediction (JSON)",
        data=json_bytes,
        file_name="prediction_report.json",
        mime="application/json",
    )

    single_csv = pd.DataFrame([single_payload])
    st.download_button(
        label="Download Single Prediction (CSV)",
        data=single_csv.to_csv(index=False).encode("utf-8"),
        file_name="prediction_report.csv",
        mime="text/csv",
    )

    if batch_df is not None:
        st.download_button(
            label="Download Batch Predictions (CSV)",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def workflow_example(
    raw_features: pd.DataFrame,
    model: Any,
    feature_order: list[str],
    candidate_models: dict[str, Any],
    model_name: str,
) -> pd.Series:
    st.subheader("Load Example Storm")
    st.caption("Choose an existing storm sample and run forecast instantly.")

    sample_indices = raw_features.index.tolist()
    selected_idx = st.selectbox("Example row index", sample_indices, index=min(10, len(sample_indices) - 1))
    base_row = raw_features.loc[selected_idx].copy()

    with st.expander("Preview selected example row"):
        st.dataframe(pd.DataFrame([base_row]), use_container_width=True)

    scenario_row = apply_scenario(base_row, list(raw_features.columns))
    prediction, _ = make_single_prediction(scenario_row, model, feature_order)
    spread = candidate_spread(scenario_row, candidate_models, feature_order)

    render_prediction_cards(prediction, spread, model_name)

    payload = {
        "mode": "example",
        "row_index": str(selected_idx),
        "model": model_name,
        "prediction_24h_wind_knots": prediction,
        "risk_band": risk_band(prediction)[0],
        "uncertainty_proxy": spread,
    }
    render_downloads(payload)
    return scenario_row


def workflow_manual(
    raw_features: pd.DataFrame,
    model: Any,
    feature_order: list[str],
    candidate_models: dict[str, Any],
    model_name: str,
) -> pd.Series:
    st.subheader("Manual Input")
    st.caption("Start from a real storm snapshot and edit key controls.")

    base_idx = st.selectbox("Base storm row", raw_features.index.tolist(), index=min(20, len(raw_features) - 1))
    row = raw_features.loc[base_idx].copy()

    basin_cols = [c for c in raw_features.columns if c.startswith("basin_")]
    active_basin = next((c for c in basin_cols if float(row[c]) == 1.0), basin_cols[0] if basin_cols else None)

    c1, c2, c3 = st.columns(3)
    year = c1.number_input("YEAR_0", min_value=1900, max_value=2100, value=int(row.get("YEAR_0", 2020)), step=1)
    month = c2.number_input("MONTH_0", min_value=1, max_value=12, value=int(row.get("MONTH_0", 8)), step=1)
    day = c3.number_input("DAY_0", min_value=1, max_value=31, value=int(row.get("DAY_0", 15)), step=1)

    row["YEAR_0"] = year
    row["MONTH_0"] = month
    row["DAY_0"] = day

    if active_basin:
        selected_basin = st.selectbox("Basin", basin_cols, index=basin_cols.index(active_basin))
        for c in basin_cols:
            row[c] = 1.0 if c == selected_basin else 0.0

    latest_controls = {
        "LATITUDE_": (-40.0, 40.0),
        "LONGITUDE_": (-180.0, 180.0),
        "WMO_WIND_": (0.0, 220.0),
        "WMO_PRESSURE_": (850.0, 1050.0),
        "DISTANCE_TO_LAND_": (0.0, 2000.0),
        "STORM_TRANSLATION_SPEED_": (0.0, 80.0),
    }

    cols = list(raw_features.columns)
    st.markdown("#### Key Control Inputs")
    for prefix, (vmin, vmax) in latest_controls.items():
        col = latest_column(cols, prefix)
        if col is None:
            continue
        row[col] = st.slider(
            col,
            min_value=float(vmin),
            max_value=float(vmax),
            value=float(row[col]),
            step=0.1,
        )

    scenario_row = apply_scenario(row, cols)
    prediction, _ = make_single_prediction(scenario_row, model, feature_order)
    spread = candidate_spread(scenario_row, candidate_models, feature_order)

    render_prediction_cards(prediction, spread, model_name)

    payload = {
        "mode": "manual",
        "base_row": str(base_idx),
        "model": model_name,
        "prediction_24h_wind_knots": prediction,
        "risk_band": risk_band(prediction)[0],
        "uncertainty_proxy": spread,
    }
    render_downloads(payload)
    return scenario_row


def workflow_batch(model: Any, feature_order: list[str], model_name: str) -> pd.Series | None:
    st.subheader("Batch CSV Upload")
    st.caption("Upload a CSV with raw storm columns matching the training schema.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return None

    raw_bytes = uploaded.read()
    df = pd.read_csv(io.BytesIO(raw_bytes), index_col=0)

    if df.shape[1] < 5:
        df = pd.read_csv(io.BytesIO(raw_bytes))

    X = prepare_inference_features(df, feature_order)
    preds = model.predict(X)

    out = df.copy()
    out["predicted_24h_wind_knots"] = preds
    out["risk_band"] = [risk_band(float(v))[0] for v in preds]

    st.dataframe(out.head(100), use_container_width=True)
    render_downloads(
        {
            "mode": "batch",
            "model": model_name,
            "rows": int(len(out)),
            "mean_prediction_24h_wind_knots": float(pd.Series(preds).mean()),
        },
        batch_df=out,
    )

    if len(df) == 0:
        return None

    map_candidates = df.index.tolist()[: min(200, len(df))]
    map_idx = st.selectbox("Map sample row from batch", map_candidates, index=0)
    return df.loc[map_idx].copy()


def render_intro_tab() -> None:
    st.header("Project Introduction")
    st.markdown(
        """
This dashboard operationalizes a hurricane intensity forecasting pipeline end-to-end.

### What this project does
- Forecasts **24h future wind intensity (knots)** from historical storm features.
- Trains and compares multiple ML models (`dummy`, `cart`, `rf`, `lgbm`).
- Selects best model by **RMSE** and serves all trained models for scenario analysis.

### Data and features
- Input data includes basin, date, latitude/longitude history, wind, pressure, distance to land, and translation speed.
- Feature engineering includes cyclical transforms:
  - `cos/sin(latitude)`
  - `cos/sin(longitude)`
  - `cos/sin(day_of_year)`

### How to use this dashboard
1. Go to **Forecast** for prediction workflows (example/manual/batch).
2. Use model selector in sidebar to switch prediction model.
3. Check **Metrics** for model benchmarking and SHAP explainability.
4. Explore **Hurricane Atlas** for global spatial view colored by speed.
5. Review **Descriptive Analytics** for EDA and dataset patterns.
        """
    )


def render_metrics_comparison(metrics: dict[str, dict[str, float]], primary_metric: str) -> None:
    st.subheader("Model Metrics Comparison")

    if not metrics:
        st.info("No metrics found in manifest.")
        return

    df = pd.DataFrame.from_dict(metrics, orient="index").reset_index(names="model")
    df = df.sort_values("test_rmse")

    best_model = str(df.iloc[0]["model"])
    st.success(f"Best test RMSE model: **{best_model}**")

    show_cols = [
        "model",
        "cv_rmse",
        "test_rmse",
        "cv_mae",
        "test_mae",
        "cv_r2",
        "test_r2",
        "fit_seconds",
    ]
    present_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[present_cols], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        rmse_mae = df[["model", "test_rmse", "test_mae"]].melt("model", var_name="metric", value_name="value")
        fig1 = px.bar(
            rmse_mae,
            x="model",
            y="value",
            color="metric",
            barmode="group",
            title="Test RMSE / MAE by Model",
            color_discrete_sequence=["#ef476f", "#118ab2"],
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.bar(
            df,
            x="model",
            y="test_r2",
            color="test_r2",
            color_continuous_scale="Viridis",
            title="Test R² by Model",
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Primary selection metric: {primary_metric}")


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    return np.asarray(shap_values)


def _expected_scalar(expected_value: Any) -> float:
    arr = np.asarray(expected_value).reshape(-1)
    return float(arr[0])


def render_shap_section(
    selected_model_name: str,
    model: Any,
    feature_order: list[str],
    X_full: pd.DataFrame,
) -> None:
    st.subheader("SHAP Explainability")
    st.caption("Blue/red dots show how low/high feature values shift predictions.")

    if selected_model_name == "dummy":
        st.info("SHAP is not informative for the dummy baseline. Select CART/RF/LGBM.")
        return

    sample_max = min(1500, len(X_full))
    sample_n = st.slider("SHAP sample size", min_value=100, max_value=max(100, sample_max), value=min(500, sample_max), step=50)
    sample_df = X_full.sample(n=min(sample_n, len(X_full)), random_state=42)

    try:
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_df)
            shap_arr = _normalize_shap_values(shap_values)

        # Summary beeswarm (classic blue/red dot plot).
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_arr, sample_df, show=False, max_display=15)
        st.pyplot(fig, clear_figure=True)

        # Force plot for one point.
        force_row_idx = st.selectbox("Force plot row", sample_df.index.tolist(), index=0)
        force_row = sample_df.loc[[force_row_idx]]
        force_values = _normalize_shap_values(explainer.shap_values(force_row))
        if force_values.ndim == 2:
            force_values = force_values[0]

        force_plot = shap.force_plot(
            _expected_scalar(explainer.expected_value),
            force_values,
            force_row,
            matplotlib=False,
        )
        components.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=320,
            scrolling=True,
        )
    except Exception as exc:
        st.warning(f"Could not generate SHAP plots for this model: {exc}")


def build_global_points(raw_features: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw_features.columns)
    lat_cols = _sorted_suffix_columns(columns, "LATITUDE_")
    basin_cols = [c for c in columns if c.startswith("basin_")]

    points: list[dict[str, Any]] = []
    for idx, row in raw_features.iterrows():
        basin = _basin_from_row(row, basin_cols)
        month = row.get("MONTH_0", np.nan)

        for lat_col in lat_cols:
            suffix = lat_col.split("_")[-1]
            lon_col = f"LONGITUDE_{suffix}"
            wind_col = f"WMO_WIND_{suffix}"
            speed_col = f"STORM_TRANSLATION_SPEED_{suffix}"

            if lon_col not in row.index:
                continue

            lat = row.get(lat_col)
            lon = row.get(lon_col)
            if pd.isna(lat) or pd.isna(lon):
                continue

            points.append(
                {
                    "sample_id": str(idx),
                    "timestep": int(suffix),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "translation_speed_kmh": float(row.get(speed_col, 0.0)) if pd.notna(row.get(speed_col, np.nan)) else 0.0,
                    "wind_knots": float(row.get(wind_col, 0.0)) if pd.notna(row.get(wind_col, np.nan)) else 0.0,
                    "basin": basin,
                    "month": int(month) if pd.notna(month) else None,
                }
            )

    if not points:
        return pd.DataFrame()
    return pd.DataFrame(points)


def render_global_hurricane_atlas(raw_features: pd.DataFrame) -> None:
    st.header("Global Hurricane Atlas")
    st.caption("All storm snapshots at once, colored by translation speed.")

    all_points = build_global_points(raw_features)
    if all_points.empty:
        st.info("No track points available for global plotting.")
        return

    basin_options = sorted(all_points["basin"].dropna().unique().tolist())
    timestep_options = sorted(all_points["timestep"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    selected_basins = c1.multiselect("Basins", basin_options, default=basin_options)
    selected_steps = c2.multiselect("Timesteps", timestep_options, default=timestep_options)
    max_points = c3.slider("Max points rendered", min_value=5000, max_value=80000, value=min(30000, len(all_points)), step=5000)

    filtered = all_points[
        all_points["basin"].isin(selected_basins)
        & all_points["timestep"].isin(selected_steps)
    ]

    if len(filtered) > max_points:
        filtered = filtered.sample(max_points, random_state=42)

    fig = px.scatter_geo(
        filtered,
        lat="latitude",
        lon="longitude",
        color="translation_speed_kmh",
        size="wind_knots",
        hover_name="sample_id",
        hover_data={
            "basin": True,
            "month": True,
            "timestep": True,
            "translation_speed_kmh": ":.2f",
            "wind_knots": ":.2f",
            "latitude": ":.2f",
            "longitude": ":.2f",
        },
        color_continuous_scale="Turbo",
        projection="natural earth",
        title="All Hurricanes: Colored by Translation Speed, Sized by Wind",
        opacity=0.72,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        geo={
            "showland": True,
            "landcolor": "#f2efe9",
            "showocean": True,
            "oceancolor": "#dff3ff",
            "showcoastlines": True,
            "coastlinecolor": "#5b6b7a",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def render_descriptive_analytics(raw_features: pd.DataFrame, target: pd.Series) -> None:
    st.header("Descriptive Analytics")
    st.caption("A plot gallery to teach multiple visualization styles on the same hurricane dataset.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw_features):,}")
    c2.metric("Columns", f"{raw_features.shape[1]:,}")
    c3.metric("Target Mean", f"{target.mean():.2f}")
    c4.metric("Target Std", f"{target.std():.2f}")

    basin_cols = [c for c in raw_features.columns if c.startswith("basin_")]
    month_col = "MONTH_0" if "MONTH_0" in raw_features.columns else None
    wind_col = latest_column(list(raw_features.columns), "WMO_WIND_")
    pressure_col = latest_column(list(raw_features.columns), "WMO_PRESSURE_")
    speed_col = latest_column(list(raw_features.columns), "STORM_TRANSLATION_SPEED_")
    distance_col = latest_column(list(raw_features.columns), "DISTANCE_TO_LAND_")
    lat_col = latest_column(list(raw_features.columns), "LATITUDE_")
    lon_col = latest_column(list(raw_features.columns), "LONGITUDE_")

    basin_label = (
        raw_features[basin_cols].idxmax(axis=1).str.replace("basin_", "").str.replace("_0", "", regex=False)
        if basin_cols
        else pd.Series(["Unknown"] * len(raw_features), index=raw_features.index)
    )

    ana_df = pd.DataFrame(index=raw_features.index)
    ana_df["basin"] = basin_label
    if month_col:
        ana_df["month"] = raw_features[month_col].astype("Int64")
    if wind_col:
        ana_df["wind_knots"] = raw_features[wind_col]
    if pressure_col:
        ana_df["pressure_hpa"] = raw_features[pressure_col]
    if speed_col:
        ana_df["translation_speed_kmh"] = raw_features[speed_col]
    if distance_col:
        ana_df["distance_to_land_km"] = raw_features[distance_col]
    if lat_col:
        ana_df["latitude"] = raw_features[lat_col]
    if lon_col:
        ana_df["longitude"] = raw_features[lon_col]
    ana_df["target_24h_wind_knots"] = target.reindex(raw_features.index)

    tab_dist, tab_rel, tab_comp, tab_adv = st.tabs(
        ["Distribution Plots", "Relationship Plots", "Composition Plots", "Advanced Plots"]
    )

    with tab_dist:
        st.markdown("#### Distribution-focused examples")
        if wind_col:
            hist_df = ana_df.dropna(subset=["wind_knots"])
            fig_hist = px.histogram(
                hist_df,
                x="wind_knots",
                nbins=45,
                title="Histogram: Latest Wind Intensity Distribution",
                color_discrete_sequence=["#118ab2"],
                marginal="box",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if basin_cols:
                fig_violin = px.violin(
                    hist_df,
                    x="basin",
                    y="wind_knots",
                    color="basin",
                    box=True,
                    points="all",
                    title="Violin Plot: Wind Distribution by Basin",
                )
                st.plotly_chart(fig_violin, use_container_width=True)

                ecdf_df = hist_df.sample(min(8000, len(hist_df)), random_state=42)
                fig_ecdf = px.ecdf(
                    ecdf_df,
                    x="wind_knots",
                    color="basin",
                    title="ECDF: Cumulative Distribution of Wind by Basin",
                )
                st.plotly_chart(fig_ecdf, use_container_width=True)

        if month_col and speed_col:
            box_df = ana_df.dropna(subset=["month", "translation_speed_kmh"])
            box_df["month"] = box_df["month"].astype(str)
            fig_box = px.box(
                box_df,
                x="month",
                y="translation_speed_kmh",
                color="month",
                title="Box Plot: Translation Speed by Month",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_rel:
        st.markdown("#### Relationship-focused examples")
        if wind_col and pressure_col and speed_col:
            scatter_df = ana_df.dropna(subset=["wind_knots", "pressure_hpa", "translation_speed_kmh", "basin"])
            scatter_df = scatter_df.sample(min(5000, len(scatter_df)), random_state=42)

            fig_scatter = px.scatter(
                scatter_df,
                x="pressure_hpa",
                y="wind_knots",
                color="translation_speed_kmh",
                symbol="basin",
                title="Scatter: Wind vs Pressure (Colored by Translation Speed)",
                color_continuous_scale="Turbo",
                opacity=0.75,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            fig_density = px.density_heatmap(
                scatter_df,
                x="pressure_hpa",
                y="wind_knots",
                nbinsx=35,
                nbinsy=35,
                color_continuous_scale="Magma",
                title="2D Density Heatmap: Wind vs Pressure",
            )
            st.plotly_chart(fig_density, use_container_width=True)

            if distance_col:
                bubble_df = ana_df.dropna(subset=["distance_to_land_km", "wind_knots", "translation_speed_kmh", "basin"])
                bubble_df = bubble_df.sample(min(4000, len(bubble_df)), random_state=42)
                fig_bubble = px.scatter(
                    bubble_df,
                    x="distance_to_land_km",
                    y="wind_knots",
                    size="translation_speed_kmh",
                    color="basin",
                    title="Bubble Plot: Wind vs Distance to Land (Bubble Size = Speed)",
                    opacity=0.7,
                )
                st.plotly_chart(fig_bubble, use_container_width=True)

    with tab_comp:
        st.markdown("#### Composition-focused examples")
        if month_col:
            month_counts = ana_df["month"].value_counts(dropna=True).sort_index().rename_axis("month").reset_index(name="count")
            fig_month = px.bar(
                month_counts,
                x="month",
                y="count",
                title="Bar Chart: Distribution of Samples by Month",
                color="count",
                color_continuous_scale="Blues",
            )
            fig_month.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_month, use_container_width=True)

        if basin_cols:
            basin_count = ana_df["basin"].value_counts().rename_axis("basin").reset_index(name="count")
            fig_basin = px.bar(
                basin_count,
                x="basin",
                y="count",
                title="Bar Chart: Samples by Basin",
                color="count",
                color_continuous_scale="Teal",
            )
            fig_basin.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_basin, use_container_width=True)

            if month_col:
                comp_df = ana_df.dropna(subset=["month"]).copy()
                comp_df["month"] = comp_df["month"].astype(str)
                group = comp_df.groupby(["basin", "month"], as_index=False).size().rename(columns={"size": "count"})

                fig_sun = px.sunburst(
                    group,
                    path=["basin", "month"],
                    values="count",
                    title="Sunburst: Basin → Month Composition",
                )
                st.plotly_chart(fig_sun, use_container_width=True)

                fig_treemap = px.treemap(
                    group,
                    path=["basin", "month"],
                    values="count",
                    title="Treemap: Basin → Month Composition",
                    color="count",
                    color_continuous_scale="Agsunset",
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

    with tab_adv:
        st.markdown("#### Advanced examples")
        # Correlation overview on key \"latest\" variables for readability.
        corr_cols = [
            c
            for c in [wind_col, pressure_col, speed_col, distance_col, lat_col, lon_col, "target_24h_wind_knots"]
            if c and c in ana_df.columns
        ]
        if len(corr_cols) >= 2:
            corr = ana_df[corr_cols].corr(numeric_only=True)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix (Latest-Timestep Core Variables)",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        pc_cols = [c for c in ["wind_knots", "pressure_hpa", "translation_speed_kmh", "distance_to_land_km", "target_24h_wind_knots"] if c in ana_df.columns]
        if len(pc_cols) >= 3:
            pc_df = ana_df.dropna(subset=pc_cols).sample(min(2000, len(ana_df.dropna(subset=pc_cols))), random_state=42)
            fig_pc = px.parallel_coordinates(
                pc_df,
                dimensions=pc_cols,
                color="wind_knots" if "wind_knots" in pc_df.columns else pc_cols[0],
                color_continuous_scale="Viridis",
                title="Parallel Coordinates: Multivariate Pattern View",
            )
            st.plotly_chart(fig_pc, use_container_width=True)

        if lat_col and lon_col and wind_col:
            map_df = ana_df.dropna(subset=["latitude", "longitude", "wind_knots", "translation_speed_kmh"])
            map_df = map_df.sample(min(12000, len(map_df)), random_state=42)
            fig_hex = px.density_map(
                map_df,
                lat="latitude",
                lon="longitude",
                z="translation_speed_kmh",
                radius=14,
                center=dict(lat=12, lon=-45),
                zoom=1,
                map_style="carto-positron",
                title="Density Map: Spatial Speed Intensity (Latest Timesteps)",
                color_continuous_scale="Turbo",
            )
            st.plotly_chart(fig_hex, use_container_width=True)


def main() -> None:
    st.title("Hurricane Forecast Decision Cockpit")
    st.caption("Operational 24h intensity forecasting with interpretable, scenario-ready outputs.")

    manifest = load_manifest(str(PROJECT_ROOT / "models" / "model_manifest.json"))
    artifact_paths = manifest.get("artifact_paths", {})
    metrics = manifest.get("metrics", {})

    candidate_model_paths = {
        name: path
        for name, path in (artifact_paths.get("candidate_models") or {}).items()
        if isinstance(path, str)
    }

    best_model_name = str(manifest.get("model_name", "unknown"))
    best_model_path = artifact_paths.get("best_model")
    if best_model_name not in candidate_model_paths and isinstance(best_model_path, str):
        candidate_model_paths[best_model_name] = best_model_path

    if not candidate_model_paths:
        raise FileNotFoundError("No candidate model artifacts found. Run training first.")

    registry_items = tuple(sorted(candidate_model_paths.items()))
    model_registry = load_model_registry(registry_items)

    model_options = list(model_registry.keys())
    default_index = model_options.index(best_model_name) if best_model_name in model_options else 0

    raw_features, target = load_base_data()
    feature_order = manifest.get("feature_order", [])
    X_full = build_engineered_frame(tuple(feature_order))

    with st.sidebar:
        st.header("Model Controls")
        selected_model_name = st.selectbox("Active model", model_options, index=default_index)
        st.write(f"**Best model in manifest:** `{best_model_name}`")
        st.write(f"**Primary metric:** `{manifest.get('primary_metric', 'rmse')}`")

        selected_metrics = metrics.get(selected_model_name, {})
        if selected_metrics:
            st.metric("Test RMSE", f"{selected_metrics.get('test_rmse', float('nan')):.2f}")
            st.metric("Test MAE", f"{selected_metrics.get('test_mae', float('nan')):.2f}")
            st.metric("Test R²", f"{selected_metrics.get('test_r2', float('nan')):.3f}")

    model = model_registry[selected_model_name]

    tab_intro, tab_forecast, tab_metrics, tab_atlas, tab_desc = st.tabs(
        [
            "Intro",
            "Forecast",
            "Metrics",
            "Hurricane Atlas",
            "Descriptive Analytics",
        ]
    )

    with tab_intro:
        render_intro_tab()

    with tab_forecast:
        mode = st.radio(
            "Workflow",
            options=["Load example storm", "Manual input", "Batch CSV upload"],
            horizontal=True,
        )

        selected_row_for_map: pd.Series | None = None

        col_left, col_right = st.columns([1.45, 1.0])

        with col_left:
            if mode == "Load example storm":
                selected_row_for_map = workflow_example(
                    raw_features,
                    model,
                    feature_order,
                    model_registry,
                    selected_model_name,
                )
            elif mode == "Manual input":
                selected_row_for_map = workflow_manual(
                    raw_features,
                    model,
                    feature_order,
                    model_registry,
                    selected_model_name,
                )
            else:
                selected_row_for_map = workflow_batch(model, feature_order, selected_model_name)

        with col_right:
            render_key_drivers(model, feature_order)
            render_storm_map(selected_row_for_map)

    with tab_metrics:
        sub_metrics, sub_shap = st.tabs(["Model Metrics", "SHAP"])
        with sub_metrics:
            render_metrics_comparison(metrics=metrics, primary_metric=str(manifest.get("primary_metric", "rmse")))
        with sub_shap:
            render_shap_section(
                selected_model_name=selected_model_name,
                model=model,
                feature_order=feature_order,
                X_full=X_full,
            )

    with tab_atlas:
        render_global_hurricane_atlas(raw_features)

    with tab_desc:
        render_descriptive_analytics(raw_features, target)


if __name__ == "__main__":
    main()
