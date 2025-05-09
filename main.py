from src.SARIMAX_model import run_sarimax_forecast_with_outputs
from src.lstm_model import run_lstm_analysis_all_periods,regenerate_demand_lagged_csv
from src.EDA import run_eda_pipeline
from src.ensemble_model import run_ensemble_model,preprocess_data
from src.linear_regression import run_linear_regression_analysis
from src.utils_sarimax import run_sarimax_comparison
from src.demand_forecasting_pipeline import run_full_demand_forecasting_pipeline
from src.plot_utils import generate_all_plots
from src.shap_model import run_lstm_with_shap_interpretation
from src.models import (
    add_time_features,
    filter_periods,
    evaluate_period,
    evaluate_combined_model,
    plot_model_comparison
)

import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("Output/Charts", exist_ok=True)
    os.makedirs("Output/Configuration", exist_ok=True)

    print("🚀 Running EDA pipeline...")
    run_eda_pipeline()

    print("🧠 Running ensemble pipeline...")
    df = pd.read_csv("Data/EMA_Demand_Lagged.csv")
    df = preprocess_data(df)
    os.makedirs("images", exist_ok=True)

    results = []

    run_ensemble_model(df[df['DayOfWeek'] < 5], "Weekdays", "images", results)
    run_ensemble_model(df[df['DayOfWeek'] == 5], "Saturday", "images", results)
    run_ensemble_model(df[df['DayOfWeek'] == 6], "Sunday", "images", results)

    cny_mask = ((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | ((df['Month'] == 2) & (df['Date'].dt.day <= 10))
    run_ensemble_model(df[cny_mask], "CNY", "images", results)

    run_ensemble_model(df[df['Year'].isin([2020, 2021])], "COVID", "images", results)

    run_ensemble_model(df, "Overall", "images", results)

    results_df = pd.DataFrame(results)
    os.makedirs("Output/Ensemble", exist_ok=True)
    results_df.to_csv("Output/Ensemble/ensemble_model_metrics.csv", index=False)
    print("📁 Ensemble metrics saved to Output/Ensemble/ensemble_model_metrics.csv")
    
    print("🧠 Running Linear Regression pipeline...")
    run_linear_regression_analysis()
    
    print("🚀 Running Demand pipeline...")
    run_full_demand_forecasting_pipeline()
    

    print("🚀 Running plot pipeline...")
    # Paths to actual_vs_predicted CSVs
    file_paths = {
    "Linear Regression": "Output/Linear/actual_vs_predicted_linear.csv",
    "Ensemble": "Output/Ensemble/actual_vs_predicted_ensemble.csv",
    "XGBoost": "Output/xgboost/actual_vs_predicted_xgboost.csv",
    "CatBoost": "Output/catboost/actual_vs_predicted_catboost.csv",
    "LightGBM": "Output/lightgbm/actual_vs_predicted_lightgbm.csv",
    "Random Forest": "Output/random_forest/actual_vs_predicted_random_forest.csv"
    }
    summary_df = pd.read_csv("Output/final_model_performance_summary.csv")
    models = ["xgboost", "random_forest", "lightgbm", "catboost"] 
    # Combine all into one DataFrame
    dfs = []
    for model_name, path in file_paths.items():
        df = pd.read_csv(path)
        df["Model"] = model_name
        df["Absolute_Error"] = abs(df["Actual"] - df["Predicted"])
        df["Residual"] = df["Actual"] - df["Predicted"]
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    
    
    generate_all_plots(df_combined, summary_df, models)
    
    print("🧠 Running LSTM  pipeline...")
    # Load preprocessed lagged dataset
    regenerate_demand_lagged_csv()
    demand_lagged = pd.read_csv("Data/demand_feature_dataset_LSTM.csv",parse_dates=["Datetime"])
    demand_lagged.set_index("Datetime", inplace=True)  # Reapply index if needed

    # Run LSTM analysis across all scenarios
    result_df = run_lstm_analysis_all_periods(demand_lagged)

    # Save results
    os.makedirs("Output/LSTM", exist_ok=True)
    result_df.to_csv("Output/LSTM/LSTM_Performance_By_Period.csv", index=False)

    print("✅ LSTM results saved to Output/LSTM/")
    
    print("🚀 Running SHAP pipeline...")
    run_lstm_with_shap_interpretation()
    
    print("📊 Running SARIMAX forecasting...")
    run_sarimax_comparison()


if __name__ == "__main__":
    main()