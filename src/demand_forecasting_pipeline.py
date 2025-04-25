import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# === Feature Engineering ===
def add_time_features(df):
    df['Hour'] = pd.to_datetime(df['Period Ending Time'], errors='coerce').dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

def filter_periods(df, frac=0.2):
    df_covid = df[df['Year'].isin([2020, 2021])].sample(frac=frac, random_state=42)
    cny_mask = ((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | ((df['Month'] == 2) & (df['Date'].dt.day <= 10))
    df_cny = df[cny_mask].sample(frac=frac, random_state=42)
    typical_mask = (df['Year'] == 2019) & (df['TreatAs_DayType_Code'] == 0)
    df_typical = df[typical_mask].sample(frac=frac, random_state=42)
    df_saturday = df[df['DayOfWeek'] == 5].sample(frac=frac, random_state=42)
    df_sunday = df[df['DayOfWeek'] == 6].sample(frac=frac, random_state=42)
    return df_covid, df_cny, df_typical, df_saturday, df_sunday

# === Model Evaluation ===
def evaluate_period(data, features, target, label, model_type="xgboost"):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_type == "xgboost":
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=30, max_depth=2, learning_rate=0.2)
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "lightgbm":
        model = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "catboost":
        model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=False)
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save outputs
    output_dir = os.path.join("Output", model_type)
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv(
        os.path.join(output_dir, f"actual_vs_predicted_{model_type}.csv"), index=False)
    joblib.dump(model, os.path.join(output_dir, f"{model_type}_model.pkl"))

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_type.title()} - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_predictions.png"))
    plt.close()

    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        fi.to_csv(os.path.join(output_dir, f"{model_type}_feature_importance.csv"))

    return pd.DataFrame({"MAE": [mean_absolute_error(y_test, y_pred)],
                         "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred))],
                         "R²": [r2_score(y_test, y_pred)]}, index=[f"{label}-{model_type.title()}"])

def evaluate_combined_model(df, features, target, label, model_type):
    return evaluate_period(df, features, target, label, model_type)

# === Plot Comparison Charts ===
def plot_model_comparison(results_df, model_type, output_path):
    results_df = results_df.copy()
    results_df.index = results_df.index.str.replace("_", " ")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ['MAE', 'RMSE', 'R²']
    for i, metric in enumerate(metrics):
        axs[i].bar(results_df.index, results_df[metric], color='skyblue', edgecolor='black')
        axs[i].set_title(f"{metric} Comparison")
        axs[i].tick_params(axis='x', rotation=30)
    plt.suptitle(f"{model_type.title()} Model Performance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === Main Pipeline ===
def run_full_demand_forecasting_pipeline():
    DATASETS = {
        "Scenario 1": "Data/Combined_Demand_Data.xlsx",
        "Scenario 2": "Data/EMA_Demand Data (2015-2025).xlsx",
        "Scenario 3": "Data/EMA_Demand_Lagged.csv"
    }

    FEATURE_SETS = {
        "Scenario 1": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
        "Scenario 2": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
        "Scenario 3": [
            'NEM Demand (Forecast)',
            'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3',
            'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3',
            'Hour', 'DayOfWeek', 'TreatAs_DayType_Code']
    }

    MODEL_TYPES = ["xgboost", "random_forest", "lightgbm", "catboost"]
    target = 'NEM Demand (Actual)'
    final_results = []

    for model_type in MODEL_TYPES:
        for scenario_label, filepath in DATASETS.items():
            input_path = os.path.abspath(filepath)
            if filepath.endswith(".xlsx"):
                df = pd.read_excel(input_path).drop(index=0)
            else:
                df = pd.read_csv(input_path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Period Ending Time'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M", errors='coerce')
            df = add_time_features(df)

            df_covid, df_cny, df_typical, df_saturday, df_sunday = filter_periods(df, frac=0.2)
            df_combined = pd.concat([df_covid, df_cny, df_typical, df_saturday, df_sunday], axis=0)
            features = FEATURE_SETS[scenario_label]

            results_covid = evaluate_period(df_covid, features, target, f"{scenario_label}_COVID", model_type)
            results_cny = evaluate_period(df_cny, features, target, f"{scenario_label}_CNY", model_type)
            results_typical = evaluate_period(df_typical, features, target, f"{scenario_label}_Typical_Day", model_type)
            results_saturday = evaluate_period(df_saturday, features, target, f"{scenario_label}_Saturday", model_type)
            results_sunday = evaluate_period(df_sunday, features, target, f"{scenario_label}_Sunday", model_type)
            results_combined = evaluate_combined_model(df_combined, features, target, f"{scenario_label}_Combined", model_type)

            all_results = pd.concat([results_covid, results_cny, results_typical, results_saturday, results_sunday, results_combined])
            all_results['Model_Type'] = model_type

            out_dir = os.path.join("Output", scenario_label.replace(" ", "_"))
            os.makedirs(out_dir, exist_ok=True)
            all_results.to_csv(os.path.join(out_dir, f"{model_type}_all_periods_evaluation_summary.csv"))

            chart_path = os.path.join(out_dir, f"comparison_chart_{model_type}.png")
            plot_model_comparison(all_results, model_type, chart_path)
            final_results.append(all_results)

    if final_results:
        summary_df = pd.concat(final_results).reset_index()
        summary_df.to_csv("Output/final_combined_all_models_results.csv", index=False)
        print("✅ Full pipeline completed and results saved!")