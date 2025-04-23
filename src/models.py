import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
import logging

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def add_time_features(df):
    df['Hour'] = pd.to_datetime(df['Period Ending Time']).dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

def filter_periods(df, frac=0.02):
    df_covid = df[df['Year'].isin([2020, 2021])].sample(frac=frac, random_state=42)
    cny_mask = ((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | ((df['Month'] == 2) & (df['Date'].dt.day <= 10))
    df_cny = df[cny_mask].sample(frac=frac, random_state=42)
    typical_mask = (df['Year'] == 2019) & (df['TreatAs_DayType_Code'] == 0)
    df_typical = df[typical_mask].sample(frac=frac, random_state=42)
    return df_covid, df_cny, df_typical

def evaluate_period(data, features, target, label, model_type="xgboost"):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if model_type == "xgboost":
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=30, max_depth=2, learning_rate=0.2)
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "lightgbm":
        model = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "catboost":
        model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=False)
    else:
        raise ValueError("Invalid model_type. Choose from 'xgboost', 'random_forest', 'lightgbm', 'catboost'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Save model
    model_dir = os.path.join("models", label.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/{model_type}_model.pkl")

    # Save prediction plot
    img_dir = os.path.join("images", label.replace(" ", "_").lower())
    os.makedirs(img_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k', linewidths=0.2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title(f"{model_type.title()}: Predictions vs Actuals - {label}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{img_dir}/{model_type}_predictions.png")
    plt.close()

    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        fi.to_csv(f"{img_dir}/{model_type}_feature_importance.csv")

    logging.info(f"✅ {model_type.title()} evaluation completed for {label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    return pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "R²": [r2]}, index=[f"{label}-{model_type.title()}"])

def evaluate_combined_model(df, features, target, label="Combined", model_type="xgboost"):
    return evaluate_period(df, features, target, label, model_type)

def plot_model_comparison(results_df, model_type, output_path=None):
    if output_path is None:
        output_path = f"images/comparison_chart_{model_type}.png"

    rename_map = {}
    for scenario in ["Scenario 1", "Scenario 2", "Scenario 3"]:
        prefix = scenario.replace("Scenario ", "S")
        for period in ["COVID", "CNY", "Typical_Day", "Combined"]:
            rename_map[f"{scenario}_{period}-{model_type.title()}"] = f"{prefix} - {period.replace('_', ' ')}"

    results_df = results_df.copy()
    results_df.index = results_df.index.map(lambda x: rename_map.get(x, x))

    model_names = results_df.index.tolist()
    mae_values = results_df['MAE'].tolist()
    rmse_values = results_df['RMSE'].tolist()
    r2_values = results_df['R²'].tolist()

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    def add_labels(ax, values, decimals=2):
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.015, f"{v:.{decimals}f}", ha='center', fontsize=10)

    axs[0].bar(model_names, mae_values, color='skyblue', edgecolor='black')
    axs[0].set_title("MAE Comparison", fontsize=13)
    axs[0].set_ylabel("Mean Absolute Error")
    axs[0].tick_params(axis='x', rotation=20)
    add_labels(axs[0], mae_values)

    axs[1].bar(model_names, rmse_values, color='lightgreen', edgecolor='black')
    axs[1].set_title("RMSE Comparison", fontsize=13)
    axs[1].set_ylabel("Root Mean Squared Error")
    axs[1].tick_params(axis='x', rotation=20)
    add_labels(axs[1], rmse_values)

    axs[2].bar(model_names, r2_values, color='salmon', edgecolor='black')
    axs[2].set_title("R² Comparison", fontsize=13)
    axs[2].set_ylabel("R² Score")
    axs[2].tick_params(axis='x', rotation=20)
    add_labels(axs[2], r2_values, decimals=4)

    plt.suptitle(f"{model_type.title()} Model Performance Comparison Across Calendar Periods", fontsize=16)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.15, wspace=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"📊 Comparison chart saved to: {output_path}")