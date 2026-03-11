import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Dominance DSS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp { background-color: #0d0d1a; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #111122; border-right: 1px solid #222244; }
    section[data-testid="stSidebar"] * { color: #c0c0d0 !important; }

    /* Metric cards */
    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00b4d8; }
    .metric-label { font-size: 0.8rem; color: #9090aa; text-transform: uppercase; letter-spacing: 1px; }

    /* Section headings */
    h1 { color: #ffffff !important; font-size: 2rem !important; }
    h2 { color: #00b4d8 !important; border-bottom: 1px solid #0f3460; padding-bottom: 6px; }
    h3 { color: #e0e0f0 !important; }

    /* Table */
    .dataframe { background-color: #16213e !important; }

    /* Divider */
    hr { border-color: #1e2a4a; }

    /* Prediction result box */
    .pred-dominant {
        background: linear-gradient(135deg, #1a3a2a, #0a2a1a);
        border: 2px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-not-dominant {
        background: linear-gradient(135deg, #3a1a1a, #2a0a0a);
        border: 2px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pred-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
    .pred-subtitle { font-size: 0.9rem; color: #aaaacc; }

    /* Sidebar radio label */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00b4d8;
        padding: 8px 0;
    }

    /* Info box */
    .info-box {
        background: #0f2a3a;
        border-left: 4px solid #00b4d8;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555577;
        font-size: 0.75rem;
        padding: 20px 0 10px;
        border-top: 1px solid #1e2a4a;
        margin-top: 40px;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ───────────────────────────────────────────────────────────────
ASSET_DIR = os.path.dirname(__file__)
META_PATH  = os.path.join(os.path.dirname(__file__), 'model_meta.json')

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    clf    = joblib.load(os.path.join(base, 'best_clf.joblib'))
    reg    = joblib.load(os.path.join(base, 'best_reg.joblib'))
    scaler = joblib.load(os.path.join(base, 'scaler.joblib'))
    fcols  = joblib.load(os.path.join(base, 'feature_cols.joblib'))
    return clf, reg, scaler, fcols

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'ev_vs_petrol_dataset_v3.csv'))

clf_model, reg_model, scaler, feature_cols = load_models()
meta = load_meta()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ DSS Navigation</div>', unsafe_allow_html=True)
    st.markdown("---")
    module = st.radio(
        "Select Module",
        ["🏠 Executive Summary",
         "🔍 Exploratory Data Analysis",
         "🤖 Models Used",
         "📊 Evaluation Metrics",
         "🚗 EV Dominance Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**System Status:** 🟢 Operational")
    st.markdown(f"**Best Classifier:** {meta['best_clf_name']}")
    st.markdown(f"**Best Regressor:** {meta['best_reg_name']}")
    st.markdown(f"**Last Update:** 2026-03-11")

def asset(name):
    return os.path.join(ASSET_DIR, name)

def footer():
    st.markdown('<div class="footer">© 2026 EV Dominance DSS | From Combustion to Electric | Developed for Academic Research</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if module == "🏠 Executive Summary":
    st.markdown("<h1 style='text-align:center'>From Combustion to Electric</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9090aa;'>Global Transition to Electric Vehicle Dominance — AI-Driven Decision Support System</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        <div class="info-box">
        Predicts whether a vehicle market achieves <b>EV dominance</b> (binary classification)
        and estimates the expected <b>EV market share</b> (regression) using socioeconomic,
        infrastructure, and policy-related features across 1,200 records spanning multiple
        countries from 2010 onward.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Dataset Size")
        st.markdown('<div class="metric-card"><div class="metric-value">1,200</div><div class="metric-label">Records</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><div class="metric-value">22</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Regions</div></div>', unsafe_allow_html=True)

        clf_acc = meta['clf_results'][meta['best_clf_name']]['Accuracy']
        reg_r2  = meta['reg_results'][meta['best_reg_name']]['R2']
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{clf_acc*100:.1f}%</div><div class="metric-label">Classifier Accuracy</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{reg_r2:.3f}</div><div class="metric-label">Regression R²</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Core Objectives")
        objectives = [
            ("1.", "Build and evaluate ML classification models to predict **EV market dominance** (`is_ev_dominant`)"),
            ("2.", "Develop regression models estimating **EV market share** as a continuous target"),
            ("3.", "Perform **Exploratory Data Analysis** to uncover patterns, trends, and correlations"),
            ("4.", "Compare multiple ML models using classification and regression **evaluation metrics**"),
            ("5.", "Deploy the best-performing model as an interactive **Streamlit web application**"),
        ]
        for num, obj in objectives:
            st.markdown(f"**{num}** {obj}")

    st.markdown("---")
    st.markdown("### 📂 Dataset Features")

    features_data = {
        "Feature": [
            "ev_market_share", "ev_growth_rate_yoy", "charging_stations",
            "fast_chargers_share", "avg_ev_range_km", "fuel_price_usd_per_liter",
            "electricity_price_usd_per_kwh", "gdp_per_capita", "urban_population_percent",
            "co2_emissions_transport_mt", "ev_subsidy_usd", "emission_regulation_score"
        ],
        "Type": [
            "Continuous (0–1)", "Continuous (%)", "Integer",
            "Continuous (0–1)", "Continuous (km)", "Continuous (USD)",
            "Continuous (USD)", "Continuous (USD)", "Continuous (%)",
            "Continuous (MT)", "Continuous (USD)", "Continuous (0–100)"
        ],
        "Description": [
            "EV sales as proportion of total vehicle sales",
            "Year-over-year EV sales growth rate",
            "Total number of EV charging stations",
            "Proportion of fast chargers available",
            "Average range of EV models in km",
            "Average fuel price per liter",
            "Electricity cost per kilowatt-hour",
            "Economic strength of each country",
            "Urbanization rate",
            "CO₂ emissions from transport sector",
            "Government EV subsidies in USD",
            "Stringency score of emission regulations"
        ]
    }
    st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🧮 Mathematical Framework")
    st.markdown("The Law of Large Numbers ensures that as market sample sizes grow, observed EV market share converges to the true population mean:")
    st.latex(r"S_n = \frac{1}{n} \sum_{i=1}^{n} X_i")
    st.markdown("As sample size grows, the Central Limit Theorem guarantees approximate normality of sample means:")
    st.latex(r"\text{As } n \to \infty,\ \sqrt{n}(S_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)")

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🔍 Exploratory Data Analysis":
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Uncovering patterns, distributions, and relationships within the EV vs. Petrol dataset.")
    st.markdown("---")

    st.markdown("## 1. EV Market Share Distribution")
    st.image(asset('eda_market_share_dist.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Insight:</b> EV market share is heavily right-skewed — most markets still have very low EV penetration, with a small number of high-adoption markets pulling the distribution right.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 2. Class Balance: EV Dominance")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(asset('eda_class_balance.png'), use_container_width=True)
    with col2:
        st.markdown("""
        <br><br>
        <div class="info-box">
        <b>Key Finding:</b> The dataset is <b>highly imbalanced</b> — only 22 out of 1,200 records
        (≈1.8%) represent EV-dominant markets. This necessitated <b>class-weight balancing</b>
        during model training to avoid the classifier simply predicting the majority class.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 3. EV Market Share Over Time (2010–Present)")
    st.image(asset('eda_temporal.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Insight:</b> EV market share shows a clear upward trend over time, consistent with IEA reports on accelerating global EV adoption, particularly from 2018 onward.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 4. Regional Comparison")
    st.image(asset('eda_regional.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Insight:</b> EV adoption varies significantly by region, reflecting differences in policy environments, infrastructure investment, and economic development.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 5. Feature Correlation Heatmap")
    st.image(asset('eda_correlation.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Key Correlations:</b> EV growth rate YoY, charging stations, and emission regulation score show the strongest positive correlations with EV market share. GDP per capita and fast charger share are also notable predictors.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 6. GDP per Capita vs. EV Market Share")
    st.image(asset('eda_gdp_scatter.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Insight:</b> Higher GDP per capita markets tend to cluster at higher EV market shares. EV-dominant markets (shown in red) are concentrated in wealthier economies with stronger infrastructure support.</div>', unsafe_allow_html=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MODELS USED
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🤖 Models Used":
    st.markdown("<h1>Models Used</h1>", unsafe_allow_html=True)
    st.markdown("Five machine learning models were trained and compared for both classification and regression tasks.")
    st.markdown("---")

    st.markdown("## 🏷️ Classification Task")
    st.markdown("**Target:** `is_ev_dominant` — Binary (0 = Not EV Dominant, 1 = EV Dominant)")

    clf_cards = [
        ("📐 Logistic Regression", "Baseline linear classifier. Models the log-odds of EV dominance as a linear combination of input features. Simple, interpretable, and effective as a baseline.", "Baseline", "#0f3460"),
        ("🌿 Decision Tree", "Tree-structured model that splits data based on feature thresholds. Highly interpretable and captures non-linear relationships.", "Interpretable", "#1a4a2a"),
        ("🌲 Random Forest", "Ensemble of 100 decision trees using bagging. Reduces variance and improves generalization. **Best classifier in this study.**", "Best Classifier ⭐", "#3a2a0a"),
        ("🚀 Gradient Boosting", "Sequential boosting ensemble (XGBoost-style). Builds trees to correct residuals from prior trees. Powerful on structured data.", "High Performance", "#2a0a3a"),
        ("📍 K-Nearest Neighbors", "Non-parametric method that classifies based on the k nearest training samples. Serves as an additional non-linear baseline.", "Baseline", "#0a2a3a"),
    ]
    for name, desc, badge, color in clf_cards:
        st.markdown(f"""
        <div style="background:{color}22; border:1px solid {color}88; border-radius:10px; padding:16px; margin-bottom:10px;">
            <b style="font-size:1rem;">{name}</b>
            <span style="float:right; background:{color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem;">{badge}</span>
            <p style="margin:8px 0 0; color:#c0c0d0; font-size:0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📈 Regression Task")
    st.markdown("**Target:** `ev_market_share` — Continuous (0–1)")

    reg_cards = [
        ("📐 Ridge Regression", "Linear regression with L2 regularization to prevent overfitting. Serves as the baseline regression model.", "Baseline", "#0f3460"),
        ("🌿 Decision Tree Regressor", "Tree-based regression using recursive feature splitting. Captures non-linear relationships in the data.", "Interpretable", "#1a4a2a"),
        ("🌲 Random Forest Regressor", "Ensemble of decision tree regressors. Excellent variance reduction through bagging of predictions.", "High R²", "#3a2a0a"),
        ("🚀 Gradient Boosting Regressor", "Boosting ensemble that minimizes regression loss iteratively. **Best regressor — R² = 0.977.**", "Best Regressor ⭐", "#2a0a3a"),
        ("📍 KNN Regressor", "Predicts target as the average of k nearest neighbors. Simple non-parametric approach.", "Baseline", "#0a2a3a"),
    ]
    for name, desc, badge, color in reg_cards:
        st.markdown(f"""
        <div style="background:{color}22; border:1px solid {color}88; border-radius:10px; padding:16px; margin-bottom:10px;">
            <b style="font-size:1rem;">{name}</b>
            <span style="float:right; background:{color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.75rem;">{badge}</span>
            <p style="margin:8px 0 0; color:#c0c0d0; font-size:0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Training Setup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Preprocessing:**
        - One-hot encoding for `region`, `vehicle_segment`
        - `StandardScaler` normalization
        - 80/20 stratified train-test split
        """)
    with col2:
        st.markdown("""
        **Class Imbalance Handling:**
        - `class_weight='balanced'` for applicable classifiers
        - Balanced weights computed from training set distribution
        """)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif module == "📊 Evaluation Metrics":
    st.markdown("<h1>Evaluation Metrics</h1>", unsafe_allow_html=True)
    st.markdown("Comprehensive performance comparison across all trained models.")
    st.markdown("---")

    st.markdown("## 🏷️ Classification Results")
    clf_df = pd.DataFrame(meta['clf_results']).T.reset_index()
    clf_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    clf_df = clf_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    def highlight_best(col):
        is_best = col == col.max()
        return ['background-color: #0f3460; color: #00b4d8; font-weight: bold' if v else '' for v in is_best]

    st.dataframe(clf_df.style.apply(highlight_best, subset=['Accuracy','Precision','Recall','F1-Score']),
                 use_container_width=True, hide_index=True)

    st.image(asset('eval_clf_comparison.png'), use_container_width=True)

    best = meta['best_clf_name']
    br = meta['clf_results'][best]
    st.markdown(f"""
    <div class="info-box">
    <b>Best Classifier: {best}</b> — 
    Accuracy: {br['Accuracy']*100:.1f}% | 
    Precision: {br['Precision']:.4f} | 
    Recall: {br['Recall']:.4f} | 
    F1-Score: {br['F1-Score']:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔢 Confusion Matrix — Best Classifier")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(asset('eval_confusion_matrix.png'), use_container_width=True)
    with col2:
        st.markdown(f"""
        <br>
        <div class="info-box">
        The confusion matrix shows predictions from the best classifier ({meta['best_clf_name']})
        on the 20% test set. Given the significant class imbalance (only ~1.8% EV-dominant),
        class-weight balancing was critical to achieving meaningful recall on the minority class.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📈 Regression Results")
    reg_df = pd.DataFrame(meta['reg_results']).T.reset_index()
    reg_df.columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R²']
    reg_df = reg_df.sort_values('R²', ascending=False).reset_index(drop=True)

    def highlight_reg(col):
        if col.name == 'R²':
            is_best = col == col.max()
        else:
            is_best = col == col.min()
        return ['background-color: #0f3460; color: #00b4d8; font-weight: bold' if v else '' for v in is_best]

    st.dataframe(reg_df.style.apply(highlight_reg, subset=['MAE','MSE','RMSE','R²']),
                 use_container_width=True, hide_index=True)

    st.image(asset('eval_reg_comparison.png'), use_container_width=True)

    best_r = meta['best_reg_name']
    rr = meta['reg_results'][best_r]
    st.markdown(f"""
    <div class="info-box">
    <b>Best Regressor: {best_r}</b> — 
    MAE: {rr['MAE']:.4f} | MSE: {rr['MSE']:.4f} | RMSE: {rr['RMSE']:.4f} | R²: {rr['R2']:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔑 Feature Importance (Random Forest)")
    st.image(asset('eval_feature_importance.png'), use_container_width=True)
    st.markdown('<div class="info-box"><b>Key Finding:</b> EV growth rate YoY, charging station count, and GDP per capita are among the strongest predictors of EV dominance — consistent with EDA findings and domain knowledge.</div>', unsafe_allow_html=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🚗 EV Dominance Predictor":
    st.markdown("<h1 style='text-align:center'>⚡ Interactive EV Dominance Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#9090aa;'>Enter market conditions to predict EV dominance and estimated EV market share</p>", unsafe_allow_html=True)
    st.markdown("---")

    regions = ['Americas', 'Asia', 'Europe', 'Oceania']
    segments = ['Commercial', 'Mass Market', 'Premium']

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("🌍 Region", regions)
        vehicle_segment = st.selectbox("🚘 Vehicle Segment", segments)
        charging_stations = st.slider("🔌 Charging Stations", 100, 50000, 5000, step=100)
        fast_chargers_share = st.slider("⚡ Fast Chargers Share (0–1)", 0.0, 1.0, 0.3, step=0.01)
        avg_ev_range_km = st.slider("🛣️ Average EV Range (km)", 100, 800, 350, step=10)
        gdp_per_capita = st.slider("💰 GDP per Capita (USD)", 5000, 120000, 35000, step=1000)

    with col2:
        fuel_price = st.slider("⛽ Fuel Price (USD/liter)", 0.5, 4.0, 1.5, step=0.05)
        electricity_price = st.slider("💡 Electricity Price (USD/kWh)", 0.05, 0.50, 0.15, step=0.01)
        urban_pop = st.slider("🏙️ Urban Population (%)", 20.0, 100.0, 65.0, step=1.0)
        co2_emissions = st.slider("🏭 CO₂ Emissions Transport (MT)", 5.0, 300.0, 80.0, step=1.0)
        ev_subsidy = st.slider("🏛️ EV Subsidy (USD)", 0, 15000, 3000, step=100)
        emission_reg_score = st.slider("📋 Emission Regulation Score (0–100)", 0.0, 100.0, 50.0, step=1.0)
        ev_growth_rate = st.slider("📈 EV Growth Rate YoY (%)", -10.0, 200.0, 25.0, step=0.5)

    st.markdown("---")

    if st.button("🔮 Generate Prediction", use_container_width=True):
        # Build input vector
        input_dict = {
            'charging_stations': charging_stations,
            'fast_chargers_share': fast_chargers_share,
            'avg_ev_range_km': avg_ev_range_km,
            'fuel_price_usd_per_liter': fuel_price,
            'electricity_price_usd_per_kwh': electricity_price,
            'gdp_per_capita': gdp_per_capita,
            'urban_population_percent': urban_pop,
            'co2_emissions_transport_mt': co2_emissions,
            'ev_subsidy_usd': ev_subsidy,
            'emission_regulation_score': emission_reg_score,
            'ev_growth_rate_yoy': ev_growth_rate,
        }

        # One-hot encode region and vehicle segment
        for r in regions:
            input_dict[f'region_{r}'] = 1 if region == r else 0
        for s in segments:
            seg_key = s.replace(' ', '_')
            input_dict[f'vehicle_segment_{seg_key}'] = 1 if vehicle_segment == s else 0

        input_df = pd.DataFrame([input_dict])

        # Align columns
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        input_scaled = scaler.transform(input_df)

        prediction = clf_model.predict(input_scaled)[0]
        proba = clf_model.predict_proba(input_scaled)[0]
        market_share = reg_model.predict(input_scaled)[0]
        market_share = float(np.clip(market_share, 0, 100))

        st.markdown("### 🎯 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="pred-dominant">
                    <div class="pred-title" style="color:#00c853;">✅ EV DOMINANT</div>
                    <div class="pred-subtitle">This market is predicted to achieve EV dominance.</div>
                    <hr style="border-color:#00c85355; margin:10px 0">
                    <div style="font-size:0.85rem; color:#aaffaa;">Confidence: {proba[1]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-not-dominant">
                    <div class="pred-title" style="color:#e94560;">❌ NOT YET DOMINANT</div>
                    <div class="pred-subtitle">This market has not yet reached EV dominance.</div>
                    <hr style="border-color:#e9456055; margin:10px 0">
                    <div style="font-size:0.85rem; color:#ffaaaa;">Confidence: {proba[0]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="height:100%;">
                <div class="metric-label">Estimated EV Market Share</div>
                <div class="metric-value" style="font-size:2.5rem;">{market_share:.2f}%</div>
                <div class="metric-label" style="margin-top:8px;">of total vehicle sales</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown("#### 📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='#0d0d1a')
        ax.set_facecolor('#16213e')
        ax.barh(['EV Dominant', 'Not Dominant'], [proba[1], proba[0]], color=['#00c853', '#e94560'], height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e2a4a')
        for i, v in enumerate([proba[1], proba[0]]):
            ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"""
        <div class="info-box">
        <b>Model used:</b> {meta['best_clf_name']} (Classification) + {meta['best_reg_name']} (Regression)<br>
        <b>Note:</b> Predictions are based on patterns in the training dataset. Real-world adoption
        depends on additional local factors not captured in this model.
        </div>
        """, unsafe_allow_html=True)

    footer()
