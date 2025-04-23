import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M").dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayType_Hour'] = df['TreatAs_DayType_Code'] * df['Hour']
    return df

def run_ensemble_model(df, output_dir):
    features = ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code', 'DayType_Hour']
    target = 'NEM Demand (Actual)'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    # Ensemble: average predictions
    ensemble_pred = (lr_pred + dt_pred) / 2

    # Evaluation
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)

    print("=== Ensemble Model Evaluation ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.5f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, ensemble_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title("Ensemble Model (LR + Decision Tree)")
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ensemble_model_scatter.png")
    plt.savefig(fig_path)
    plt.close()

def main():
    df = pd.read_csv("dDta/EMA_Demand_Lagged.csv")
    df = preprocess_data(df)
    os.makedirs("images", exist_ok=True)
    run_ensemble_model(df, "images")

if __name__ == "__main__":
    main()