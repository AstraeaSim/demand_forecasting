# === demand_model_utils.py ===

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


def add_time_features(df):
    df['Hour'] = pd.to_datetime(df['Period Ending Time']).dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df


def evaluate_period(data, features, target, label):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=30,
        max_depth=2,
        learning_rate=0.2,
        eval_metric=["rmse", "mae"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    
    # Save plot with better styling
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k', linewidths=0.2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title(f"XGBoost Prediction vs Actuals (20% COVID, CNY, Typical Each)")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/xgboost_predictions_{label.lower()}.png")
    plt.close()
    

    return pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    }, index=[f"{label}-XGBoost"])


def filter_periods(df):
    df_covid = df[df['Year'].isin([2020, 2021])].sample(frac=0.02, random_state=42)
    cny_mask = ((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | \
               ((df['Month'] == 2) & (df['Date'].dt.day <= 10))
    df_cny = df[cny_mask].sample(frac=0.02, random_state=42)
    typical_mask = (df['Year'] == 2019) & (df['TreatAs_DayType_Code'] == 0)
    df_typical = df[typical_mask].sample(frac=0.02, random_state=42)
    return df_covid, df_cny, df_typical


def evaluate_combined_model(df, features, target, label="Combined"):
    from sklearn.model_selection import train_test_split
    import os

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=30,
        max_depth=2,
        learning_rate=0.2,
        eval_metric=["rmse", "mae"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Save plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title(f"XGBoost: Predictions vs Actuals - {label}")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/xgboost_predictions_{label.lower()}.png")
    plt.close()

    return pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    }, index=[f"{label}-XGBoost"])
    

import matplotlib.pyplot as plt
import os

def plot_model_comparison(results_df, output_path="images/comparison_chart.png"):
    """
    Plots MAE, RMSE, and R² for multiple models from a DataFrame.
    """
    model_names = results_df.index.tolist()
    mae_values = results_df['MAE'].tolist()
    rmse_values = results_df['RMSE'].tolist()
    r2_values = results_df['R²'].tolist()

    # Create subplots with increased spacing and padding
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # Wider and taller

    def add_labels(ax, values, decimals=2):
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.015, f"{v:.{decimals}f}", ha='center', fontsize=10)

    # --- MAE Plot ---
    axs[0].bar(model_names, mae_values, color='skyblue', edgecolor='black')
    axs[0].set_title("MAE Comparison", fontsize=13)
    axs[0].set_ylabel("Mean Absolute Error")
    axs[0].tick_params(axis='x', rotation=20)
    add_labels(axs[0], mae_values)

    # --- RMSE Plot ---
    axs[1].bar(model_names, rmse_values, color='lightgreen', edgecolor='black')
    axs[1].set_title("RMSE Comparison", fontsize=13)
    axs[1].set_ylabel("Root Mean Squared Error")
    axs[1].tick_params(axis='x', rotation=20)
    add_labels(axs[1], rmse_values)

    # --- R² Plot ---
    axs[2].bar(model_names, r2_values, color='salmon', edgecolor='black')
    axs[2].set_title("R² Comparison", fontsize=13)
    axs[2].set_ylabel("R² Score")
    axs[2].tick_params(axis='x', rotation=20)
    add_labels(axs[2], r2_values, decimals=4)

    # Global title and layout
    plt.suptitle("XGBoost Model Performance Comparison Across Calendar Periods", fontsize=16)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.15, wspace=0.3)  # More padding

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Comparison chart saved to: {output_path}")