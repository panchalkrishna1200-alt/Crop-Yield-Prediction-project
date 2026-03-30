"""
app.py
------
Streamlit dashboard for Crop Yield Prediction.
Run:  streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import (
    load_data, clean_data, encode_features,
    get_feature_matrix, dataset_info
)
from utils.eda import (
    plot_yield_distribution, plot_top_crops, plot_yield_by_season,
    plot_yield_trend, plot_top_states, plot_correlation,
    plot_crop_season_heatmap
)
from utils.model import (
    split_data, train_all_models, results_to_df,
    cross_validate_model, tune_random_forest,
    plot_model_comparison, plot_actual_vs_predicted,
    plot_feature_importance, plot_residuals,
    save_model, load_model
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CropCast — Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #1b4332;
    line-height: 1.2;
}
.sub-title {
    font-size: 1.05rem;
    color: #52796f;
    margin-top: -0.4rem;
}
.metric-card {
    background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    border-left: 4px solid #2d6a4f;
}
.metric-val  { font-size: 2rem; font-weight: 700; color: #1b4332; }
.metric-label{ font-size: 0.82rem; color: #52796f; text-transform: uppercase; letter-spacing: 0.08em; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #1b4332;
    border-bottom: 2px solid #95d5b2;
    padding-bottom: 6px;
    margin: 1.4rem 0 1rem 0;
}
.predict-box {
    background: #f0fdf4;
    border: 1.5px solid #95d5b2;
    border-radius: 14px;
    padding: 28px 32px;
}
.result-pill {
    display: inline-block;
    background: linear-gradient(90deg, #2d6a4f, #40916c);
    color: white;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 12px 36px;
    border-radius: 50px;
    margin-top: 12px;
}
.stButton > button {
    background: linear-gradient(90deg, #2d6a4f, #40916c);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 2rem;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

div[data-testid="stSidebar"] {
    background: #1b4332;
}
div[data-testid="stSidebar"] * { color: #d8f3dc !important; }
div[data-testid="stSidebar"] .stRadio label { color: #d8f3dc !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key in ["df", "df_enc", "encoders", "feature_cols",
            "X", "y", "results", "best_model", "best_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 CropCast")
    st.markdown("*Crop Yield Prediction*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📂 Load Data", "📊 Explore (EDA)",
         "🤖 Train Models", "🔮 Predict Yield", "📋 About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** Kaggle — Crop Production in India")
    st.markdown("**Stack:** Python · Scikit-learn · XGBoost · Streamlit")

# ═══════════════════════════════════════════════
# PAGE: Home
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-title">🌾 CropCast</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Machine Learning–Powered Crop Yield Prediction for India</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-val">60%</div>
            <div class="metric-label">India's workforce in agriculture</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-val">15+</div>
            <div class="metric-label">ML-ready features analysed</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-val">~88%</div>
            <div class="metric-label">Target R² accuracy</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How to use this app")
    steps = [
        ("📂", "Load Data", "Upload the Kaggle Crop Production CSV"),
        ("📊", "Explore (EDA)", "Visualise distributions, trends, and correlations"),
        ("🤖", "Train Models", "Compare Linear Regression, Random Forest, XGBoost, LightGBM"),
        ("🔮", "Predict Yield", "Enter crop details and get instant yield predictions"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)

# ═══════════════════════════════════════════════
# PAGE: Load Data
# ═══════════════════════════════════════════════
elif page == "📂 Load Data":
    st.markdown('<p class="section-header">📂 Load Dataset</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload the Kaggle Crop Production CSV",
        type=["csv"],
        help="Download from: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india"
    )

    if uploaded:
        with st.spinner("Loading and cleaning data …"):
            df_raw = pd.read_csv(uploaded)
            df     = clean_data(df_raw)
            df_enc, encoders = encode_features(df)
            X, y, feature_cols = get_feature_matrix(df_enc)

            st.session_state.df          = df
            st.session_state.df_enc      = df_enc
            st.session_state.encoders    = encoders
            st.session_state.feature_cols = feature_cols
            st.session_state.X           = X
            st.session_state.y           = y

        info = dataset_info(df)
        st.success("✅ Dataset loaded and cleaned successfully!")

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip(
            [c1, c2, c3, c4],
            [("Rows", f"{info['rows']:,}"), ("Crops", info["crops"]),
             ("States", info["states"]), ("Year Range", info["year_range"])]
        ):
            with col:
                st.metric(label, val)

        st.markdown("### Preview (first 200 rows)")
        st.dataframe(df.head(200), use_container_width=True)

        st.markdown("### Missing Values")
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if mv.empty:
            st.info("No missing values after cleaning 🎉")
        else:
            st.dataframe(mv.rename("Missing Count"))

    else:
        st.info("👆 Upload your CSV file above to get started.")
        st.markdown("#### Expected columns")
        st.code("State_Name, District_Name, Crop_Year, Season, Crop, Area, Production")

# ═══════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════
elif page == "📊 Explore (EDA)":
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>',
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please load a dataset first (📂 Load Data).")
        st.stop()

    df         = st.session_state.df
    df_enc     = st.session_state.df_enc
    feature_cols = st.session_state.feature_cols

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Yield Distribution", "Top Crops", "By Season",
        "Trend", "Top States", "Correlation", "Crop × Season"
    ])

    with tab1:
        st.pyplot(plot_yield_distribution(df))
    with tab2:
        n = st.slider("Number of crops", 5, 25, 15)
        st.pyplot(plot_top_crops(df, top_n=n))
    with tab3:
        st.pyplot(plot_yield_by_season(df))
    with tab4:
        st.pyplot(plot_yield_trend(df))
    with tab5:
        n2 = st.slider("Number of states", 5, 20, 12)
        st.pyplot(plot_top_states(df, top_n=n2))
    with tab6:
        st.pyplot(plot_correlation(df_enc, feature_cols))
    with tab7:
        n3 = st.slider("Number of crops in heatmap", 5, 25, 15)
        st.pyplot(plot_crop_season_heatmap(df, top_n_crops=n3))

    # Key stats
    st.markdown("---")
    st.markdown("### Quick Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Yield (kg/ha)", f"{df['Yield'].mean():,.0f}")
    c2.metric("Median Yield",      f"{df['Yield'].median():,.0f}")
    c3.metric("Max Yield",         f"{df['Yield'].max():,.0f}")


# ═══════════════════════════════════════════════
# PAGE: Train Models
# ═══════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown('<p class="section-header">🤖 Model Training & Evaluation</p>',
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please load a dataset first (📂 Load Data).")
        st.stop()

    X            = st.session_state.X
    y            = st.session_state.y
    feature_cols = st.session_state.feature_cols
    encoders     = st.session_state.encoders

    col1, col2 = st.columns([1, 3])
    with col1:
        test_size = st.slider("Test split %", 10, 40, 20) / 100
        run_cv    = st.checkbox("5-fold Cross Validation", value=True)
        run_tune  = st.checkbox("Hyperparameter Tuning (RF)", value=False)

    with col2:
        if st.button("🚀  Train All Models"):
            with st.spinner("Training models … this may take a minute"):
                X_train, X_test, y_train, y_test = split_data(
                    X, y, test_size=test_size
                )
                results  = train_all_models(X_train, X_test, y_train, y_test)
                results_df = results_to_df(results)
                best_name  = results_df.iloc[0]["Model"]
                best_model = results[best_name]["model"]

                st.session_state.results    = results
                st.session_state.best_model = best_model
                st.session_state.best_name  = best_name
                st.session_state.X_train    = X_train
                st.session_state.X_test     = X_test
                st.session_state.y_train    = y_train
                st.session_state.y_test     = y_test

            st.success(f"✅ Training complete! Best model: **{best_name}**")

    if st.session_state.results is not None:
        results    = st.session_state.results
        best_model = st.session_state.best_model
        best_name  = st.session_state.best_name
        y_test     = st.session_state.y_test
        X_test     = st.session_state.X_test

        results_df = results_to_df(results)
        st.markdown("### Model Comparison")
        st.dataframe(results_df.style.highlight_max(
            subset=["R² Score"], color="#b7e4c7"
        ).highlight_min(subset=["RMSE", "MAE"], color="#b7e4c7"),
            use_container_width=True)

        st.pyplot(plot_model_comparison(results_df))

        st.markdown(f"### Best Model Deep Dive — {best_name}")
        tab_a, tab_b, tab_c = st.tabs(
            ["Actual vs Predicted", "Feature Importance", "Residuals"]
        )
        y_pred = results[best_name]["y_pred"]
        with tab_a:
            st.pyplot(plot_actual_vs_predicted(y_test, y_pred, best_name))
        with tab_b:
            fig_fi = plot_feature_importance(best_model, feature_cols)
            if fig_fi:
                st.pyplot(fig_fi)
            else:
                st.info("Feature importance not available for this model type.")
        with tab_c:
            st.pyplot(plot_residuals(y_test, y_pred, best_name))

        if run_cv:
            with st.spinner("Running cross-validation …"):
                cv_info = cross_validate_model(best_model, X, y)
            st.markdown("### Cross-Validation Results (5-fold)")
            c1, c2 = st.columns(2)
            c1.metric("CV Mean R²", cv_info["cv_mean"])
            c2.metric("CV Std Dev", cv_info["cv_std"])

        if run_tune:
            with st.spinner("Tuning Random Forest … (may take ~2 min)"):
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                tuned_model, best_params, best_score = tune_random_forest(
                    X_train, y_train, n_iter=15
                )
            st.markdown("### Hyperparameter Tuning — Random Forest")
            st.metric("Best CV R²", best_score)
            st.json(best_params)

        # Save model button
        st.markdown("---")
        if st.button("💾  Save Best Model"):
            path = save_model(best_model, encoders, feature_cols,
                              path="models/best_model.pkl")
            st.success(f"Model saved to `{path}`")


# ═══════════════════════════════════════════════
# PAGE: Predict
# ═══════════════════════════════════════════════
elif page == "🔮 Predict Yield":
    st.markdown('<p class="section-header">🔮 Predict Crop Yield</p>',
                unsafe_allow_html=True)

    # ── Try to load model ──────────────────────
    model_payload = None

    if st.session_state.best_model is not None:
        model_payload = {
            "model"       : st.session_state.best_model,
            "encoders"    : st.session_state.encoders,
            "feature_cols": st.session_state.feature_cols,
        }
    elif os.path.exists("models/best_model.pkl"):
        model_payload = load_model("models/best_model.pkl")

    if model_payload is None:
        st.warning("No trained model found. Please train a model first (🤖 Train Models).")
        st.stop()

    model        = model_payload["model"]
    encoders     = model_payload["encoders"]
    feature_cols = model_payload["feature_cols"]

    st.markdown('<div class="predict-box">', unsafe_allow_html=True)
    st.markdown("### Enter Crop Details")

    df = st.session_state.df

    if df is None and os.path.exists("models/best_model.pkl"):
        st.info("Load a dataset for full dropdown options. Using fallback lists.")

    # Build dropdown options
    state_opts   = sorted(encoders["State_Name"].classes_)   if "State_Name"   in encoders else ["Unknown"]
    district_opts= sorted(encoders["District_Name"].classes_) if "District_Name" in encoders else ["Unknown"]
    season_opts  = sorted(encoders["Season"].classes_)        if "Season"       in encoders else ["Kharif","Rabi","Whole Year"]
    crop_opts    = sorted(encoders["Crop"].classes_)          if "Crop"         in encoders else ["Rice","Wheat"]

    c1, c2 = st.columns(2)
    with c1:
        state    = st.selectbox("State",    state_opts)
        season   = st.selectbox("Season",   season_opts)
        crop     = st.selectbox("Crop",     crop_opts)
    with c2:
        district = st.selectbox("District", district_opts)
        year     = st.number_input("Crop Year", min_value=1990, max_value=2030, value=2023)
        area     = st.number_input("Area (hectares)", min_value=0.1, max_value=500000.0,
                                    value=1.0, step=0.5, format="%.1f")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🌾  Predict Yield"):
        # Encode inputs
        row = {}
        if "State_Name_enc"   in feature_cols:
            row["State_Name_enc"]   = encoders["State_Name"].transform([state])[0]
        if "District_Name_enc" in feature_cols:
            row["District_Name_enc"]= encoders["District_Name"].transform([district])[0]
        if "Season_enc"        in feature_cols:
            row["Season_enc"]       = encoders["Season"].transform([season])[0]
        if "Crop_enc"          in feature_cols:
            row["Crop_enc"]         = encoders["Crop"].transform([crop])[0]
        if "Crop_Year"         in feature_cols:
            row["Crop_Year"]        = year
        if "Area"              in feature_cols:
            row["Area"]             = area

        X_input = pd.DataFrame([row])[feature_cols]
        predicted_yield      = model.predict(X_input)[0]
        predicted_production = predicted_yield * area

        st.markdown("---")
        st.markdown("### Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("🌾 Predicted Yield",      f"{predicted_yield:,.0f} kg/ha")
        c2.metric("📦 Estimated Production", f"{predicted_production:,.0f} kg")
        c3.metric("🗺️ Area Input",           f"{area:,.1f} ha")

        # Context band
        if df is not None:
            crop_avg = df[df["Crop"] == crop]["Yield"].mean() if crop in df["Crop"].values else None
            if crop_avg:
                delta = predicted_yield - crop_avg
                flag  = "above" if delta > 0 else "below"
                st.info(f"ℹ️  The average yield for **{crop}** in the dataset is "
                        f"**{crop_avg:,.0f} kg/ha**. Your prediction is "
                        f"**{abs(delta):,.0f} kg/ha {flag}** average.")

# ═══════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════
elif page == "📋 About":
    st.markdown('<p class="section-header">📋 About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
**CropCast** is a B.Tech / B.Sc final-year project demonstrating how machine learning can
help Indian farmers and agricultural agencies predict crop yield before the harvest.

---
### Models Used
| Model | Type | Notes |
|---|---|---|
| Linear Regression | Baseline | Fast, interpretable |
| Random Forest | Ensemble | High accuracy, robust |
| Gradient Boosting | Ensemble | Good generalisation |
| XGBoost | Boosting | Best performance on tabular data |
| LightGBM | Boosting | Fastest on large datasets |

### Evaluation Metrics
- **MAE** — Mean Absolute Error (lower = better)
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **R² Score** — Proportion of variance explained (target ≥ 0.85)

### Dataset
- Kaggle: *Crop Production in India* ([link](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india))
- FAO Global Crop Statistics

### Technologies
`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `LightGBM` · `Matplotlib` · `Seaborn` · `Streamlit`

---
*Built as a Machine Learning project for B.Tech / B.Sc (CS / Agriculture / Data Science) students.*
""")
