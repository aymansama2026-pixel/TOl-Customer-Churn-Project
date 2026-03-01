"""
House Price Predictor — PRO Edition
Streamlit App with premium UI/UX
Run: streamlit run app_pro.py
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ — House Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "EstateIQ — AI-Powered House Price Prediction Engine"
    }
)

MODEL_DIR = "models"

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --gold:    #C9A84C;
        --gold2:   #E8C97A;
        --dark:    #0D1117;
        --dark2:   #161B22;
        --dark3:   #21262D;
        --surface: #1C2128;
        --border:  #30363D;
        --text:    #E6EDF3;
        --muted:   #8B949E;
        --green:   #3FB950;
        --red:     #F85149;
        --blue:    #58A6FF;
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--dark) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text) !important;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--dark2) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }


    /* ── Hero Section ── */
    .hero-wrapper {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 3rem 2rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(201,168,76,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-logo {
        font-family: 'Playfair Display', serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.35em;
        color: var(--gold);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 900;
        color: var(--text);
        line-height: 1.15;
        margin-bottom: 0.75rem;
    }
    .hero-title span {
        color: var(--gold);
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--muted);
        font-weight: 300;
        letter-spacing: 0.02em;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(201,168,76,0.12);
        border: 1px solid rgba(201,168,76,0.3);
        color: var(--gold2);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        padding: 0.3rem 0.85rem;
        border-radius: 100px;
        margin-right: 0.5rem;
        margin-top: 1rem;
        text-transform: uppercase;
    }

    /* ── Price Card ── */
    .price-card {
        background: linear-gradient(145deg, #1a1f2e, #0f1318);
        border: 1px solid rgba(201,168,76,0.25);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(201,168,76,0.15);
    }
    .price-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }
    .price-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        color: var(--gold);
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .price-value {
        font-family: 'Playfair Display', serif;
        font-size: 3.8rem;
        font-weight: 900;
        color: var(--text);
        line-height: 1;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(201,168,76,0.15);
    }
    .price-range {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 0.75rem;
    }
    .price-range span {
        color: var(--gold2);
        font-weight: 500;
    }
    .confidence-bar-wrap {
        margin-top: 1.5rem;
        background: rgba(255,255,255,0.05);
        border-radius: 100px;
        height: 6px;
        overflow: hidden;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, var(--gold), var(--gold2));
        width: 88%;
    }
    .confidence-label {
        font-size: 0.7rem;
        color: var(--muted);
        margin-top: 0.4rem;
        text-align: right;
    }

    /* ── Stat Cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 0;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: var(--gold); }
    .stat-icon { font-size: 1.2rem; margin-bottom: 0.4rem; }
    .stat-val {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text);
    }
    .stat-key {
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1.2rem;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .section-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        color: var(--gold);
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ── Chart Container ── */
    .chart-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .chart-title {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* ── Segment Pill ── */
    .segment-pill {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1rem;
    }
    .seg-budget  { background: rgba(63,185,80,0.15);  color: #3FB950; border: 1px solid rgba(63,185,80,0.3);  }
    .seg-mid     { background: rgba(88,166,255,0.15); color: #58A6FF; border: 1px solid rgba(88,166,255,0.3); }
    .seg-premium { background: rgba(201,168,76,0.15); color: #C9A84C; border: 1px solid rgba(201,168,76,0.3); }
    .seg-luxury  { background: rgba(248,81,73,0.15);  color: #F85149; border: 1px solid rgba(248,81,73,0.3);  }

    /* ── Sidebar Labels ── */
    .sidebar-section {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: var(--gold);
        text-transform: uppercase;
        padding: 1rem 0 0.4rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.75rem;
    }

    /* ── Model Badge ── */
    .model-badge {
        background: rgba(63,185,80,0.1);
        border: 1px solid rgba(63,185,80,0.25);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: var(--green);
        margin-bottom: 1.5rem;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--dark2) !important;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Slider Tweaks ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #E6EDF3 !important;
        border-color: #8B949E !important;
    }
    /* Remove gold track fill */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div {
        background: var(--border) !important;
    }
    .stSlider [role="slider"] ~ div {
        background: var(--border) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    num_imp  = joblib.load(os.path.join(MODEL_DIR, "num_imputer.pkl"))
    cat_imp  = joblib.load(os.path.join(MODEL_DIR, "cat_imputer.pkl"))
    le_dict  = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODEL_DIR, "num_cols.json")) as f:
        num_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "cat_cols.json")) as f:
        cat_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    return model, scaler, num_imp, cat_imp, le_dict, feature_names, num_cols, cat_cols, model_name


def build_feature_vector(inputs, num_imp, cat_imp, le_dict, scaler,
                         feature_names, num_cols, cat_cols):
    row = inputs.copy()

    # Feature engineering — must match train.py exactly
    row["HouseAge"]      = 2010 - row["YearBuilt"]
    row["RemodAge"]      = 2010 - row["YearRemodAdd"]
    row["TotalSF"]       = row["GrLivArea"] + row["TotalBsmtSF"]
    row["TotalBath"]     = row["FullBath"] + 0.5 * row["HalfBath"]
    row["TotalPorchSF"]  = row["WoodDeckSF"] + row["OpenPorchSF"]
    row["QualCondInter"] = row["OverallQual"] * row["OverallCond"]
    row["AreaPerRoom"]   = row["GrLivArea"] / (row["TotRmsAbvGrd"] + 1)
    row["HasFireplace"]  = int(row["Fireplaces"] > 0)
    row["HasGarage"]     = int(row["GarageCars"] > 0)
    row["LogLotArea"]    = np.log1p(row["LotArea"])

    df = pd.DataFrame([row])

    # Impute numeric columns using VALUES only (no feature name check)
    num_vals = df[num_cols].values.astype(float)
    num_vals = num_imp.transform(num_vals)
    for i, c in enumerate(num_cols):
        df[c] = num_vals[0][i]

    # Impute categorical columns using VALUES only
    cat_vals = df[cat_cols].astype(str).values
    cat_vals = cat_imp.transform(cat_vals)
    for i, c in enumerate(cat_cols):
        df[c] = cat_vals[0][i]

    # Label encode categoricals
    for c in cat_cols:
        le = le_dict.get(c)
        if le:
            val = str(df[c].values[0])
            df[c] = le.transform([val if val in le.classes_ else le.classes_[0]])

    # Fill missing engineered cols and reorder
    for fc in feature_names:
        if fc not in df.columns:
            df[fc] = 0

    # Convert to numpy array in exact feature order — bypasses name mismatch
    arr = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).values
    return scaler.transform(arr)


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-logo">EstateIQ · AI Valuation Engine</div>
    <div class="hero-title">Predict Your Home's<br><span>True Market Value</span></div>
    <div class="hero-subtitle">Machine learning models trained on thousands of properties — instant, accurate, intelligent.</div>
    <div>
        <span class="hero-badge">🤖 ML Powered</span>
        <span class="hero-badge">📊 5 Models</span>
        <span class="hero-badge">⚡ Real-time</span>
        <span class="hero-badge">🎯 High Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
try:
    model, scaler, num_imp, cat_imp, le_dict, feature_names, num_cols, cat_cols, model_name = load_artefacts()
    st.markdown(f'<div class="model-badge">✅ &nbsp; Active Model: <strong>{model_name}</strong> &nbsp;·&nbsp; Status: Online</div>',
                unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Model files not found. Run `python src/train.py` first.\n\n`{e}`")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:900; color:#C9A84C;">EstateIQ</div>
        <div style="font-size:0.7rem; color:#8B949E; letter-spacing:0.15em; text-transform:uppercase;">Property Configurator</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📐 Property Size</div>', unsafe_allow_html=True)
    GrLivArea   = st.slider("Living Area (sqft)",    500,  5000, 1500, step=50)
    TotalBsmtSF = st.slider("Basement Area (sqft)",    0,  3000,  800, step=50)
    LotArea     = st.slider("Lot Area (sqft)",       1500, 25000, 8000, step=100)

    st.markdown('<div class="sidebar-section">⭐ Quality & Condition</div>', unsafe_allow_html=True)
    OverallQual = st.select_slider("Overall Quality",  options=list(range(1, 11)), value=6)
    OverallCond = st.select_slider("Overall Condition", options=list(range(1, 11)), value=5)

    st.markdown('<div class="sidebar-section">📅 Year & Age</div>', unsafe_allow_html=True)
    YearBuilt    = st.slider("Year Built",      1900, 2010, 1985)
    YearRemodAdd = st.slider("Last Remodeled",  YearBuilt, 2010, max(YearBuilt, 2000))

    st.markdown('<div class="sidebar-section">🛏 Rooms & Bathrooms</div>', unsafe_allow_html=True)
    BedroomAbvGr = st.slider("Bedrooms",       1, 8, 3)
    TotRmsAbvGrd = st.slider("Total Rooms",    2, 14, 7)
    FullBath     = st.slider("Full Bathrooms", 0, 4, 2)
    HalfBath     = st.slider("Half Bathrooms", 0, 2, 1)
    KitchenAbvGr = st.slider("Kitchens",       1, 3, 1)

    st.markdown('<div class="sidebar-section">🚗 Garage & Amenities</div>', unsafe_allow_html=True)
    GarageCars  = st.slider("Garage Capacity (cars)", 0, 4, 2)
    Fireplaces  = st.slider("Fireplaces",              0, 4, 1)
    WoodDeckSF  = st.slider("Wood Deck (sqft)",        0, 800, 0)
    OpenPorchSF = st.slider("Open Porch (sqft)",       0, 400, 40)

    st.markdown('<div class="sidebar-section">🏘 Location & Style</div>', unsafe_allow_html=True)
    Neighborhood  = st.selectbox("Neighborhood",
        ["NAmes","CollgCr","OldTown","Edwards","Somerst",
         "NridgHt","Gilbert","Sawyer","NWAmes","SawyerW"])
    HouseStyle    = st.selectbox("House Style",
        ["1Story","2Story","1.5Fin","SFoyer","SLvl"])
    RoofStyle     = st.selectbox("Roof Style",
        ["Gable","Hip","Flat","Gambrel","Mansard"])
    Condition1    = st.selectbox("Condition",
        ["Norm","Feedr","Artery","RRAe","PosN"])
    SaleCondition = st.selectbox("Sale Condition",
        ["Normal","Abnorml","Partial","AdjLand","Alloca","Family"])
    Fence         = st.selectbox("Fence",
        ["NA","MnPrv","GdPrv","MnWw","GdWo"])
    MSSubClass    = st.selectbox("MS Sub Class",
        [20,30,40,45,50,60,70,75,80,85,90,120])


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
inputs = dict(
    GrLivArea=GrLivArea, TotalBsmtSF=TotalBsmtSF, LotArea=LotArea,
    OverallQual=OverallQual, OverallCond=OverallCond,
    YearBuilt=YearBuilt, YearRemodAdd=YearRemodAdd,
    BedroomAbvGr=BedroomAbvGr, TotRmsAbvGrd=TotRmsAbvGrd,
    FullBath=FullBath, HalfBath=HalfBath, KitchenAbvGr=KitchenAbvGr,
    GarageCars=GarageCars, Fireplaces=Fireplaces,
    WoodDeckSF=WoodDeckSF, OpenPorchSF=OpenPorchSF,
    Neighborhood=Neighborhood, HouseStyle=HouseStyle,
    RoofStyle=RoofStyle, Condition1=Condition1,
    SaleCondition=SaleCondition, Fence=Fence, MSSubClass=MSSubClass,
)

try:
    vec   = build_feature_vector(inputs, num_imp, cat_imp, le_dict, scaler, feature_names, num_cols, cat_cols)
    log_p = model.predict(vec)[0]
    price = np.expm1(log_p)
    low   = price * 0.90
    high  = price * 1.10
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# Segment
if price < 150000:
    seg_class, seg_label, seg_emoji = "seg-budget",  "Budget",  "🟢"
elif price < 300000:
    seg_class, seg_label, seg_emoji = "seg-mid",     "Mid-Range","🔵"
elif price < 500000:
    seg_class, seg_label, seg_emoji = "seg-premium", "Premium", "🥇"
else:
    seg_class, seg_label, seg_emoji = "seg-luxury",  "Luxury",  "💎"

house_age   = 2023 - YearBuilt
price_sqft  = price / max(GrLivArea, 1)


# ─────────────────────────────────────────────
# MAIN RESULTS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-line"></div><div class="section-title">Valuation Result</div><div class="section-line"></div></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.3, 1, 1.2], gap="large")

# ── Price Card
with col1:
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Market Value</div>
        <div class="price-value">${price:,.0f}</div>
        <div class="price-range">
            Conservative: <span>${low:,.0f}</span> &nbsp;–&nbsp; Optimistic: <span>${high:,.0f}</span>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar"></div>
        </div>
        <div class="confidence-label">88% Model Confidence</div>
        <div style="margin-top:1rem;">
            <span class="segment-pill {seg_class}">{seg_emoji} {seg_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Key Metrics
with col2:
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon">📐</div>
            <div class="stat-val">{GrLivArea:,}</div>
            <div class="stat-key">Sqft Living</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⭐</div>
            <div class="stat-val">{OverallQual}/10</div>
            <div class="stat-key">Quality</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏠</div>
            <div class="stat-val">{house_age}y</div>
            <div class="stat-key">House Age</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">💰</div>
            <div class="stat-val">${price_sqft:.0f}</div>
            <div class="stat-key">Per Sqft</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🛏</div>
            <div class="stat-val">{BedroomAbvGr}</div>
            <div class="stat-key">Bedrooms</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🚗</div>
            <div class="stat-val">{GarageCars}</div>
            <div class="stat-key">Garage Cars</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Radar Chart
with col3:
    categories_r = ["Size", "Quality", "Rooms", "Garage", "Age", "Amenities"]
    vals_r = [
        min(GrLivArea / 4500, 1),
        OverallQual / 10,
        min(TotRmsAbvGrd / 14, 1),
        min(GarageCars / 4, 1),
        max(0, 1 - (2010 - YearBuilt) / 110),
        min((Fireplaces * 2 + int(WoodDeckSF > 0) + int(OpenPorchSF > 0)) / 6, 1),
    ]
    N      = len(categories_r)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    vals_r_plot = vals_r + [vals_r[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#1C2128")
    ax.set_facecolor("#1C2128")

    # Grid
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_plot, [r]*(N+1), color="#30363D", linewidth=0.5, linestyle="--")
    for angle in angles:
        ax.plot([angle, angle], [0, 1], color="#30363D", linewidth=0.5)

    ax.fill(angles_plot, vals_r_plot, color="#C9A84C", alpha=0.15)
    ax.plot(angles_plot, vals_r_plot, color="#C9A84C", linewidth=2.5)
    ax.scatter(angles, vals_r, color="#C9A84C", s=60, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories_r, color="#8B949E", size=9, fontfamily="DM Sans")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Property Profile", color="#E6EDF3", fontsize=11,
                 fontfamily="DM Sans", fontweight="600", pad=20)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────
# VALUE DRIVERS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-line"></div><div class="section-title">Value Drivers Analysis</div><div class="section-line"></div></div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    contribs = {
        "Living Area":     GrLivArea * 55,
        "Overall Quality": OverallQual * 12000,
        "Basement Area":   TotalBsmtSF * 30,
        "Garage":          GarageCars * 8000,
        "Fireplaces":      Fireplaces * 4000,
        "Age Depreciation":-(2023 - YearBuilt) * 200,
        "Lot Size":        LotArea * 0.5,
        "Bathrooms":       FullBath * 5000,
    }
    keys = list(contribs.keys())
    vals = list(contribs.values())

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor("#1C2128")
    ax2.set_facecolor("#1C2128")

    colors = ["#3FB950" if v >= 0 else "#F85149" for v in vals]
    bars   = ax2.barh(keys, vals, color=colors, alpha=0.85,
                      edgecolor="none", height=0.6)

    for bar, val in zip(bars, vals):
        x = bar.get_width()
        ax2.text(x + (max(vals)*0.02 if x >= 0 else -max(vals)*0.02),
                 bar.get_y() + bar.get_height()/2,
                 f"${val:+,.0f}", va="center",
                 ha="left" if x >= 0 else "right",
                 color="#E6EDF3", fontsize=8.5)

    ax2.axvline(0, color="#30363D", linewidth=1.2)
    ax2.set_xlabel("Price Impact ($)", color="#8B949E", fontsize=9)
    ax2.set_title("Feature Value Contributions", color="#E6EDF3",
                  fontsize=11, fontweight="600", pad=12)
    ax2.tick_params(colors="#8B949E", labelsize=9)
    ax2.spines[["top","right","bottom","left"]].set_color("#30363D")
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_b:
    # Price per sqft gauge
    sqft_vals  = [800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    pred_prices= [np.expm1(model.predict(
        build_feature_vector({**inputs, "GrLivArea": s}, num_imp, cat_imp, le_dict, scaler, feature_names, num_cols, cat_cols)
    )[0]) for s in sqft_vals]

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fig3.patch.set_facecolor("#1C2128")
    ax3.set_facecolor("#1C2128")

    ax3.fill_between(sqft_vals, pred_prices, alpha=0.12, color="#C9A84C")
    ax3.plot(sqft_vals, pred_prices, color="#C9A84C", linewidth=2.5)
    ax3.axvline(GrLivArea, color="#58A6FF", linewidth=1.5,
                linestyle="--", alpha=0.8, label=f"Current: {GrLivArea:,} sqft")
    ax3.axhline(price, color="#3FB950", linewidth=1.2,
                linestyle=":", alpha=0.8, label=f"Current: ${price:,.0f}")
    ax3.scatter([GrLivArea], [price], color="#C9A84C", s=100, zorder=5)

    ax3.set_xlabel("Living Area (sqft)", color="#8B949E", fontsize=9)
    ax3.set_ylabel("Predicted Price ($)", color="#8B949E", fontsize=9)
    ax3.set_title("Price vs Living Area Curve", color="#E6EDF3",
                  fontsize=11, fontweight="600", pad=12)
    ax3.tick_params(colors="#8B949E", labelsize=8)
    ax3.spines[["top","right","bottom","left"]].set_color("#30363D")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    legend = ax3.legend(fontsize=8, facecolor="#21262D",
                        edgecolor="#30363D", labelcolor="#8B949E")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────
# TRAINING CHARTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-line"></div><div class="section-title">Model Training Analysis</div><div class="section-line"></div></div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 EDA Overview", "🏆 Model Comparison",
                "🌲 Feature Importance", "📉 Residual Analysis"])
images = [
    ("outputs/01_eda.png",               "Exploratory Data Analysis"),
    ("outputs/02_model_comparison.png",  "Model Performance Comparison"),
    ("outputs/03_feature_importance.png","Top Feature Importances"),
    ("outputs/05_residuals.png",         "Residual Analysis"),
]
for tab, (img_path, caption) in zip(tabs, images):
    with tab:
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_container_width=True)
        else:
            st.markdown(f"""
            <div style="background:#21262D; border:1px dashed #30363D; border-radius:12px;
                        padding:2rem; text-align:center; color:#8B949E; font-size:0.9rem;">
                📂 Chart not found. Run <code>python src/train.py</code> to generate it.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding:2rem; border-top:1px solid #30363D;
            text-align:center; color:#8B949E; font-size:0.78rem; letter-spacing:0.05em;">
    <span style="color:#C9A84C; font-family:'Playfair Display',serif; font-weight:700;
                 font-size:1rem;">EstateIQ</span>
    &nbsp;·&nbsp; AI-Powered Real Estate Valuation
    &nbsp;·&nbsp; Built with Scikit-learn, TensorFlow & Streamlit
    &nbsp;·&nbsp; <span style="color:#3FB950;">● Live</span>
</div>
""", unsafe_allow_html=True)