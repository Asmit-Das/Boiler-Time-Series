import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Boiler Steam Temperature Predictor", layout="wide", page_icon="🔥")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #c0392b; }
    .sub-title  { font-size: 1.1rem; color: #7f8c8d; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1e1e2e; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #333;
    }
    .metric-label { font-size: .85rem; color: #aaa; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

TARGET = "TE_8332A.AV_0#"

COLS_TO_DROP = [
    'PT_8313A.AV_0#', 'PT_8313B.AV_0#', 'PT_8313C.AV_0#',
    'PT_8313D.AV_0#', 'PT_8313E.AV_0#', 'PTCA_8324.AV_0#',
    'TE_8319A.AV_0#', 'TV_8329ZC.AV_0#', 'AIR_8301A.AV_0#', 'ZCLCCY.AV_0#'
]

OUTLIER_THRESHOLDS = {
    'PT_8313F.AV_0#': 3.5,
    'PTCA_8322A.AV_0#': 3.0,
    'TE_8313B.AV_0#': 3.5,
    'FT_8306A.AV_0#': 5.0,
    'FT_8306B.AV_0#': 5.0,
    'AIR_8301B.AV_0#': 3.5,
    'YFJ3_AI.AV_0#':  7.5,
    'YFJ3_ZD1.AV_0#': 4.0,
    'SXLTCYZ.AV_0#':  3.5,
    'SXLTCYY.AV_0#':  3.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df, remove_outliers: bool):
    df = df.copy()
    # Fill nulls
    if "YJJWSLL.AV_0#" in df.columns:
        df["YJJWSLL.AV_0#"].fillna(df["YJJWSLL.AV_0#"].mean(), inplace=True)

    # Scale
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cols_to_scale  = [c for c in numerical_cols if c != TARGET]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Drop correlated
    to_drop = [c for c in COLS_TO_DROP if c in df_scaled.columns]
    df_filtered = df_scaled.drop(columns=to_drop)

    # Convert time
    if "date" in df_filtered.columns:
        df_filtered["Time"] = pd.to_datetime(df_filtered["date"], errors="coerce")
        df_filtered["Time"] = (
            df_filtered["Time"] - df_filtered["Time"].min()
        ).dt.total_seconds()
        df_filtered.drop(columns="date", inplace=True)

    if remove_outliers:
        mask = pd.Series([True] * len(df_filtered), index=df_filtered.index)
        for col, thresh in OUTLIER_THRESHOLDS.items():
            if col in df_filtered.columns:
                z = np.abs(stats.zscore(df_filtered[col].dropna()))
                z = z.reindex(df_filtered.index, fill_value=0)
                mask &= z < thresh
        df_filtered = df_filtered[mask]

    return df_filtered


def get_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, mse, r2


def train_and_predict(df, model_name, lasso_features):
    X = df.drop(columns=[TARGET, "Time"], errors="ignore")
    if lasso_features:
        X = X[[c for c in lasso_features if c in X.columns]]
    y = df[TARGET]

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=200, learning_rate=0.1,
                             max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
    elif model_name == "Hybrid (Linear + XGBoost)":
        lin = LinearRegression()
        xg  = XGBRegressor(n_estimators=200, learning_rate=0.1,
                           max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, random_state=42)
        lin.fit(X_train, y_train)
        xg.fit(X_train, y_train)
        y_pred = 0.5 * lin.predict(X_test) + 0.5 * xg.predict(X_test)
        return y_test, y_pred, df["Time"].iloc[split:] if "Time" in df else None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, df["Time"].iloc[split:] if "Time" in df else None


def get_lasso_features(df):
    X = df.drop(columns=[TARGET, "Time"], errors="ignore")
    y = df[TARGET]
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X, y)
    selected = [c for c, coef in zip(X.columns, lasso.coef_) if coef != 0]
    return selected if selected else list(X.columns)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fire-element.png", width=64)
    st.markdown("## ⚙️ Configuration")
    uploaded = st.file_uploader("Upload your sensor CSV", type=["csv"])
    st.markdown("---")
    remove_outliers = st.toggle("Remove Outliers", value=True)
    use_lasso       = st.toggle("Lasso Feature Selection", value=False)
    model_choice    = st.selectbox("Select Model", [
        "Linear Regression", "Random Forest",
        "XGBoost", "Hybrid (Linear + XGBoost)"
    ])
    st.markdown("---")
    st.caption("Target: Boiler Output Steam Temperature (`TE_8332A.AV_0#`)")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔥 Boiler Steam Temperature Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Time-series forecasting of boiler output steam temperature using industrial sensor data</div>', unsafe_allow_html=True)

if uploaded is None:
    st.info("👈 Upload your `data.csv` file in the sidebar to get started.")
    st.markdown("""
    ### What this app does
    - **Preprocesses** raw sensor data (null handling, standard scaling, feature filtering)
    - **Removes outliers** using Z-score thresholds (optional)
    - **Selects features** via Lasso regularisation (optional)
    - **Trains & evaluates** ML models and plots Actual vs Predicted
    """)
    st.stop()

# Load
df_raw = pd.read_csv(uploaded)

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model & Predictions", "📋 Raw Data"])

# ── Tab 1 – EDA ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("#### Feature Correlation Heatmap")
    num_df = df_raw.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Target Distribution")
    if TARGET in df_raw.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df_raw[TARGET].dropna(), kde=True, ax=ax, color="#e74c3c")
        ax.set_title(f"Distribution of {TARGET}")
        st.pyplot(fig)
        plt.close()

# ── Tab 2 – Model ─────────────────────────────────────────────────────────────
with tab2:
    with st.spinner("Preprocessing data…"):
        df_proc = preprocess(df_raw, remove_outliers)

    lasso_features = None
    if use_lasso:
        with st.spinner("Running Lasso feature selection…"):
            lasso_features = get_lasso_features(df_proc)
        st.success(f"Lasso selected **{len(lasso_features)}** features: `{'`, `'.join(lasso_features)}`")

    st.markdown(f"**Rows after preprocessing:** {len(df_proc)}")

    with st.spinner(f"Training {model_choice}…"):
        y_test, y_pred, time_axis = train_and_predict(df_proc, model_choice, lasso_features)

    mae, rmse, mse, r2 = get_metrics(y_test, y_pred)

    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",  f"{mae:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("MSE",  f"{mse:.4f}")
    m4.metric("R²",   f"{r2:.4f}")

    st.markdown("### Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(14, 5))
    x_vals = time_axis.values if time_axis is not None else range(len(y_test))
    ax.plot(x_vals, y_test.values, label="Actual",    color="#2980b9", linewidth=0.8)
    ax.plot(x_vals, y_pred,        label="Predicted", color="#e74c3c", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(TARGET)
    ax.set_title(f"{model_choice} — Actual vs Predicted Steam Temperature")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Residuals
    st.markdown("### Residual Plot")
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(x_vals, residuals, color="#8e44ad", linewidth=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals (Actual − Predicted)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 3 – Raw data ──────────────────────────────────────────────────────────
with tab3:
    st.dataframe(df_raw, use_container_width=True)
