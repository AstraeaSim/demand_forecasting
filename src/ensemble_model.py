
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === Data Preprocessing ===
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M").dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayType_Hour'] = df['TreatAs_DayType_Code'] * df['Hour']
    return df

# === Ensemble Modeling and Evaluation ===
def run_ensemble_model(df, label, output_dir, results_list):
    filtered_df = df.sample(frac=0.2, random_state=42)

    features = ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code', 'DayType_Hour']
    target = 'NEM Demand (Actual)'

    X = filtered_df[features]
    y = filtered_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
        # Save results
    df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': dt_pred
    })
    output_dir = "Output/Ensemble"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv('Output/Ensemble/actual_vs_predicted_ensemble.csv', index=False)

    ensemble_pred = (lr_pred + dt_pred) / 2

    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)

    print(f"=== {label} ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.5f}")

    os.makedirs(output_dir, exist_ok=True)  # <- Fix: no overwrite
    fig_path = os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_ensemble_scatter.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, ensemble_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title(f"{label} - Ensemble Model (LR + DT)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    results_list.append({
        "Label": label,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 5)
    })
# === Main Runner ===

if __name__ == "__main__":
    run_ensemble_model()
