"""
House Prices ML Pipeline
Covers: EDA, Data Cleaning, Feature Engineering, ML Models, Deep Learning, Evaluation
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

PALETTE = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
           "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#E91E63"]

# ─────────────────────────────────────────────
# 1. DATA GENERATION  (simulates Kaggle dataset)
# ─────────────────────────────────────────────
def generate_dataset(n=1500):
    """Synthetic dataset that mirrors Kaggle House Prices schema."""
    rng = np.random.default_rng(42)
    neighborhoods = ["NAmes","CollgCr","OldTown","Edwards","Somerst",
                     "NridgHt","Gilbert","Sawyer","NWAmes","SawyerW"]
    house_styles  = ["1Story","2Story","1.5Fin","SFoyer","SLvl"]
    roof_styles   = ["Gable","Hip","Flat","Gambrel","Mansard"]
    conditions    = ["Norm","Feedr","Artery","RRAe","PosN"]
    quality       = [1,2,3,4,5,6,7,8,9,10]

    n_neigh = rng.integers(0, len(neighborhoods), n)
    n_style = rng.integers(0, len(house_styles),  n)
    overall_qual = rng.choice(quality, n, p=[0.01,0.02,0.04,0.08,0.15,0.20,0.22,0.15,0.09,0.04])
    overall_cond = rng.choice(quality, n, p=[0.01,0.02,0.04,0.10,0.20,0.30,0.18,0.09,0.04,0.02])
    year_built   = rng.integers(1900, 2011, n)
    year_remod   = np.clip(year_built + rng.integers(0, 50, n), year_built, 2010)
    gr_liv_area  = rng.integers(500, 4500, n).astype(float)
    lot_area     = rng.integers(1500, 20000, n).astype(float)
    total_bsmt   = rng.integers(0, 2000, n).astype(float)
    garage_cars  = rng.integers(0, 4, n).astype(float)
    full_bath    = rng.integers(0, 4, n).astype(float)
    half_bath    = rng.integers(0, 2, n).astype(float)
    bedroom_abv  = rng.integers(1, 6, n).astype(float)
    kitchen_abv  = rng.integers(1, 3, n).astype(float)
    totrms_abv   = bedroom_abv + rng.integers(2, 5, n)
    fireplaces   = rng.integers(0, 4, n).astype(float)
    wood_deck    = rng.integers(0, 800, n).astype(float) * (rng.random(n) > 0.4)
    open_porch   = rng.integers(0, 400, n).astype(float) * (rng.random(n) > 0.5)
    fence        = rng.choice(["NA","MnPrv","GdPrv","MnWw","GdWo"], n,
                               p=[0.6,0.15,0.1,0.08,0.07])
    sale_cond    = rng.choice(["Normal","Abnorml","Partial","AdjLand","Alloca","Family"],
                               n, p=[0.82,0.07,0.06,0.01,0.01,0.03])

    # Price formula
    price = (
        40000
        + overall_qual * 12000
        + gr_liv_area  * 55
        + total_bsmt   * 30
        + lot_area     * 0.5
        + (2023 - year_built) * -200
        + garage_cars  * 8000
        + full_bath    * 5000
        + fireplaces   * 4000
        + rng.normal(0, 15000, n)
    )
    price = np.clip(price, 50000, 800000)

    # Inject missing values
    for col_arr, pct in [(gr_liv_area, 0.01), (total_bsmt, 0.05),
                         (garage_cars, 0.05), (lot_area, 0.02)]:
        mask = rng.random(n) < pct
        col_arr[mask] = np.nan

    df = pd.DataFrame({
        "Id":            np.arange(1, n+1),
        "MSSubClass":    rng.choice([20,30,40,45,50,60,70,75,80,85,90,120], n),
        "Neighborhood":  [neighborhoods[i] for i in n_neigh],
        "HouseStyle":    [house_styles[i]  for i in n_style],
        "RoofStyle":     rng.choice(roof_styles, n),
        "Condition1":    rng.choice(conditions,  n),
        "SaleCondition": sale_cond,
        "Fence":         fence,
        "OverallQual":   overall_qual,
        "OverallCond":   overall_cond,
        "YearBuilt":     year_built,
        "YearRemodAdd":  year_remod,
        "GrLivArea":     gr_liv_area,
        "LotArea":       lot_area,
        "TotalBsmtSF":   total_bsmt,
        "GarageCars":    garage_cars,
        "FullBath":      full_bath,
        "HalfBath":      half_bath,
        "BedroomAbvGr":  bedroom_abv,
        "KitchenAbvGr":  kitchen_abv,
        "TotRmsAbvGrd":  totrms_abv,
        "Fireplaces":    fireplaces,
        "WoodDeckSF":    wood_deck,
        "OpenPorchSF":   open_porch,
        "SalePrice":     price,
    })
    return df


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def run_eda(df):
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"Shape : {df.shape}")
    print(f"Dtypes:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("House Prices – Exploratory Data Analysis", fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1 – SalePrice distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["SalePrice"], bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax1.set_title("SalePrice Distribution", fontweight="bold")
    ax1.set_xlabel("Price ($)"); ax1.set_ylabel("Count")

    # 2 – Log SalePrice
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log1p(df["SalePrice"]), bins=40, color=PALETTE[2], edgecolor="white", alpha=0.85)
    ax2.set_title("Log(SalePrice) Distribution", fontweight="bold")
    ax2.set_xlabel("log(Price)")

    # 3 – GrLivArea vs SalePrice
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(df["GrLivArea"], df["SalePrice"],
                     c=df["OverallQual"], cmap="viridis", alpha=0.6, s=15)
    plt.colorbar(sc, ax=ax3, label="OverallQual")
    ax3.set_title("GrLivArea vs SalePrice", fontweight="bold")
    ax3.set_xlabel("Above-Ground Living Area (sqft)")
    ax3.set_ylabel("Sale Price ($)")

    # 4 – OverallQual box
    ax4 = fig.add_subplot(gs[1, :2])
    qual_groups = [df[df["OverallQual"]==q]["SalePrice"].dropna()
                   for q in sorted(df["OverallQual"].unique())]
    bp = ax4.boxplot(qual_groups, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE[:len(qual_groups)]):
        patch.set_facecolor(color)
    ax4.set_xticklabels(sorted(df["OverallQual"].unique()))
    ax4.set_title("SalePrice by Overall Quality", fontweight="bold")
    ax4.set_xlabel("Overall Quality"); ax4.set_ylabel("Sale Price ($)")

    # 5 – Neighborhood median price
    ax5 = fig.add_subplot(gs[1, 2])
    neigh_med = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
    ax5.barh(neigh_med.index, neigh_med.values / 1000,
             color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax5.set_title("Median Price by Neighborhood", fontweight="bold")
    ax5.set_xlabel("Median Price (k$)")

    # 6 – Correlation heatmap
    ax6 = fig.add_subplot(gs[2, :])
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax6,
                square=True, linewidths=0.3, annot=False, fmt=".2f",
                cbar_kws={"shrink": 0.6})
    ax6.set_title("Correlation Matrix", fontweight="bold")

    # 7 – Missing value bar
    ax7 = fig.add_subplot(gs[3, 0])
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    ax7.barh(miss.index, miss.values, color=PALETTE[1], edgecolor="white", alpha=0.9)
    ax7.set_title("Missing Values per Feature", fontweight="bold")
    ax7.set_xlabel("Count")

    # 8 – YearBuilt vs Price
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.scatter(df["YearBuilt"], df["SalePrice"], alpha=0.4, s=12, color=PALETTE[4])
    ax8.set_title("YearBuilt vs SalePrice", fontweight="bold")
    ax8.set_xlabel("Year Built"); ax8.set_ylabel("Sale Price ($)")

    # 9 – Porch/Deck features
    ax9 = fig.add_subplot(gs[3, 2])
    porch_cols = ["WoodDeckSF", "OpenPorchSF", "TotalBsmtSF"]
    corr_vals  = [df[c].fillna(0).corr(df["SalePrice"]) for c in porch_cols]
    bars = ax9.bar(porch_cols, corr_vals, color=PALETTE[6:9], edgecolor="white", alpha=0.9)
    ax9.set_title("Correlation with SalePrice", fontweight="bold")
    ax9.set_ylabel("Pearson r")
    ax9.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, corr_vals):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    path = os.path.join(OUTPUT_DIR, "01_eda.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ EDA saved → {path}")
    return path


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=["Id"], errors="ignore", inplace=True)

    # Feature engineering
    df["HouseAge"]        = 2010 - df["YearBuilt"]
    df["RemodAge"]        = 2010 - df["YearRemodAdd"]
    df["TotalSF"]         = df["GrLivArea"].fillna(0) + df["TotalBsmtSF"].fillna(0)
    df["TotalBath"]       = df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
    df["TotalPorchSF"]    = df["WoodDeckSF"].fillna(0) + df["OpenPorchSF"].fillna(0)
    df["QualCondInter"]   = df["OverallQual"] * df["OverallCond"]
    df["AreaPerRoom"]     = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)
    df["HasFireplace"]    = (df["Fireplaces"] > 0).astype(int)
    df["HasGarage"]       = (df["GarageCars"].fillna(0) > 0).astype(int)
    df["LogLotArea"]      = np.log1p(df["LotArea"].fillna(df["LotArea"].median()))

    # Log-transform target
    df["SalePrice"] = np.log1p(df["SalePrice"])

    # Separate features / target
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # Impute numeric
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_imp = SimpleImputer(strategy="median")
    X[num_cols] = num_imp.fit_transform(X[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

    # Encode categoricals
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        le_dict[c] = le

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(num_imp, os.path.join(MODEL_DIR, "num_imputer.pkl"))
    joblib.dump(cat_imp, os.path.join(MODEL_DIR, "cat_imputer.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.pkl"))

    feature_names = X.columns.tolist()
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(MODEL_DIR, "num_cols.json"), "w") as f:
        json.dump(num_cols, f)
    with open(os.path.join(MODEL_DIR, "cat_cols.json"), "w") as f:
        json.dump(cat_cols, f)

    return X_scaled, y, feature_names


# ─────────────────────────────────────────────
# 4 & 5. MODEL TRAINING
# ─────────────────────────────────────────────
def train_ml_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  TRAINING ML MODELS")
    print("="*60)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.001),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                                    min_samples_leaf=3, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=5, random_state=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results[name] = {"RMSE": round(rmse, 5), "R²": round(r2, 4)}
        trained[name] = model
        print(f"  {name:<22} RMSE={rmse:.5f}  R²={r2:.4f}")

    return results, trained


def train_deep_learning(X_train, X_test, y_train, y_test):
    """Simple Keras Neural Network for regression."""
    print("\n" + "="*60)
    print("  TRAINING DEEP LEARNING MODEL (Keras)")
    print("="*60)
    try:
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

        cb = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5),
        ]

        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=200,
            batch_size=32,
            callbacks=cb,
            verbose=0,
        )

        preds = model.predict(X_test, verbose=0).flatten()
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        print(f"  Neural Network           RMSE={rmse:.5f}  R²={r2:.4f}")

        # Save Keras model
        model.save(os.path.join(MODEL_DIR, "deep_model.keras"))

        # Plot training curves
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Neural Network Training Curves", fontsize=14, fontweight="bold")
        a1.plot(history.history["loss"],     color=PALETTE[0], label="Train")
        a1.plot(history.history["val_loss"], color=PALETTE[1], label="Val")
        a1.set_title("Loss (MSE)"); a1.set_xlabel("Epoch"); a1.legend()
        a2.plot(history.history["mae"],     color=PALETTE[2], label="Train MAE")
        a2.plot(history.history["val_mae"], color=PALETTE[3], label="Val MAE")
        a2.set_title("MAE"); a2.set_xlabel("Epoch"); a2.legend()
        path = os.path.join(OUTPUT_DIR, "04_dl_training.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ DL training curves → {path}")

        return {"Neural Network": {"RMSE": round(rmse, 5), "R²": round(r2, 4)}}

    except ImportError:
        print("  ⚠ TensorFlow not installed – skipping DL model")
        return {}


# ─────────────────────────────────────────────
# 6 & 7. EVALUATION & COMPARISON
# ─────────────────────────────────────────────
def plot_comparison(all_results):
    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)
    df_res = pd.DataFrame(all_results).T.sort_values("RMSE")
    print(df_res.to_string())
    df_res.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_res))]

    bars1 = ax1.bar(df_res.index, df_res["RMSE"], color=colors, edgecolor="white", alpha=0.9)
    ax1.set_title("RMSE (lower is better)", fontweight="bold")
    ax1.set_ylabel("RMSE"); ax1.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars1, df_res["RMSE"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(df_res.index, df_res["R²"], color=colors, edgecolor="white", alpha=0.9)
    ax2.set_title("R² Score (higher is better)", fontweight="bold")
    ax2.set_ylabel("R² Score"); ax2.tick_params(axis="x", rotation=30)
    ax2.set_ylim(max(0, df_res["R²"].min() - 0.05), 1.0)
    for bar, v in zip(bars2, df_res["R²"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    path = os.path.join(OUTPUT_DIR, "02_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Comparison chart → {path}")
    return df_res


# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]],
            color=[PALETTE[i % len(PALETTE)] for i in range(20)],
            edgecolor="white", alpha=0.9)
    ax.set_title("Top 20 Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    path = os.path.join(OUTPUT_DIR, "03_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance → {path}")


# ─────────────────────────────────────────────
# 9. SAVE BEST MODEL
# ─────────────────────────────────────────────
def save_best_model(trained_models, df_results):
    best_name = df_results["RMSE"].idxmin()
    # Only save sklearn models here
    if best_name in trained_models:
        best = trained_models[best_name]
        path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(best, path)
        with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
            f.write(best_name)
        print(f"\n  ✓ Best model saved: {best_name} → {path}")
    else:
        # fallback to Random Forest
        path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(trained_models["Random Forest"], path)
        with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
            f.write("Random Forest")
        print(f"\n  ✓ Best model saved: Random Forest (fallback) → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "★"*60)
    print("  HOUSE PRICES ML PIPELINE  — Full Run")
    print("★"*60)

    # Auto-create all required folders
    for folder in ["data", "models", "outputs", "src"]:
        os.makedirs(folder, exist_ok=True)

    # 1. Data
    df = generate_dataset()
    df.to_csv("data/house_prices.csv", index=False)
    print(f"\n  ✓ Dataset generated: {df.shape} rows×cols")

    # 2. EDA
    run_eda(df)

    # 3. Preprocessing
    X, y, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train/Test: {X_train.shape} / {X_test.shape}")

    # 4. ML Models
    ml_results, trained = train_ml_models(X_train, X_test, y_train, y_test)

    # 5. Deep Learning
    dl_results = train_deep_learning(
        X_train.values, X_test.values, y_train.values, y_test.values
    )

    # 6. Combine & compare
    all_results = {**ml_results, **dl_results}
    df_results  = plot_comparison(all_results)

    # 7. Feature importance
    plot_feature_importance(trained["Random Forest"], feature_names)

    # 8. Residual plot for best ML model
    best_ml = df_results[df_results.index.isin(trained.keys())]["RMSE"].idxmin()
    preds   = trained[best_ml].predict(X_test)
    residuals = y_test.values - preds

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residual Analysis – {best_ml}", fontsize=13, fontweight="bold")
    a1.scatter(preds, residuals, alpha=0.4, s=15, color=PALETTE[0])
    a1.axhline(0, color="red", linewidth=1.5, linestyle="--")
    a1.set_xlabel("Predicted"); a1.set_ylabel("Residual"); a1.set_title("Residuals vs Fitted")
    a2.hist(residuals, bins=40, color=PALETTE[2], edgecolor="white", alpha=0.85)
    a2.set_title("Residual Distribution"); a2.set_xlabel("Residual")
    path = os.path.join(OUTPUT_DIR, "05_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ Residuals → {path}")

    # 9. Save
    save_best_model(trained, df_results)

    print("\n" + "★"*60)
    print("  PIPELINE COMPLETE")
    print("★"*60)


if __name__ == "__main__":
    main()