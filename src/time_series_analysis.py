import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def run_sarimax_forecast():
    # Load lagged dataset
    df_lagged = pd.read_csv("Data/EMA_Demand_Lagged.csv", parse_dates=['Datetime'], index_col='Datetime')
    df_lagged = df_lagged.reset_index()

    # Load feature dataset
    df_feature = pd.read_csv("Output/EDA/demand_feature_dataset.csv")
    df_feature['Datetime'] = pd.to_datetime(df_feature['Date'].astype(str) + ' ' + df_feature['Period Ending Time'])

    # Merge on 'Datetime'
    merged_df = pd.merge(df_lagged, df_feature, on='Datetime', how='outer')
    merged_df = merged_df.drop(columns=['Date_y', 'Day_y', 'Period Ending Time_y',
                                        'System Demand (Actual)_y', 'NEM Demand (Actual)_y', 'NEM Demand (Forecast)_y'])
    merged_df.rename(columns={
        'Date_x': 'Date',
        'Day_x': 'Day',
        'Period Ending Time_x': 'Period Ending Time',
        'System Demand (Actual)_x': 'System Demand (Actual)',
        'NEM Demand (Actual)_x': 'NEM Demand (Actual)',
        'NEM Demand (Forecast)_x': 'NEM Demand (Forecast)'
    }, inplace=True)

    # Drop nulls and convert boolean columns
    merged_df = merged_df.dropna()
    merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']] = merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']].astype(int)

    # Set and sort index
    merged_df = merged_df.set_index('Datetime')
    merged_df = merged_df.sort_index()

    # Split data
    train = merged_df.loc['2018-01-01':'2020-01-01']
    test = merged_df.loc['2023-01-01':]

    target_col = 'System Demand (Actual)'
    exog_cols = ['NEM Demand (Forecast)', 'NEM Demand (Actual)',
                 'NEM Demand (Actual)_lag1', 'NEM Demand (Forecast)_lag1',
                 'IsWeekend', 'IsHoliday', 'IsCOVID']

    y_train = train[target_col]
    X_train = train[exog_cols]
    y_test = test[target_col]
    X_test = test[exog_cols]

    # Align test set: same columns, same index, no NaNs
    X_test = X_test[X_train.columns]                      # Same columns
    X_test = X_test.loc[y_test.index].dropna()            # Drop rows with NaNs
    y_test = y_test.loc[X_test.index]                     # Align y_test with cleaned X_test

    # Final check on shape consistency
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Mismatch: X_test has {X_test.shape[0]} rows, but y_test has {y_test.shape[0]} rows")

    print(f"✅ X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"✅ y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Fit SARIMAX model
    print("\n⏳ Fitting SARIMAX model...")
    model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=False)

    
    # Forecast
    print("✅ Forecasting...")
    # Convert to NumPy and validate shape
    X_test_array = X_test.to_numpy()

    # Confirm required shape
    expected_rows = len(pd.date_range(start=y_test.index[0], end=y_test.index[-1], freq='30min'))
    required_shape = (expected_rows, X_test.shape[1])

    if X_test_array.shape != required_shape:
        raise ValueError(f"❌ Shape mismatch: expected {required_shape}, got {X_test_array.shape}")

    # Predict using valid shape
    y_pred = results.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test_array)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 SARIMAX Forecast Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}")

    # Save chart
    os.makedirs("Output/SARIMAX", exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title('SARIMAX Forecast vs Actual')
    plt.xlabel('DateTime')
    plt.ylabel('System Demand (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Output/SARIMAX/sarimax_forecast.png", dpi=300)
    plt.close()

    # Save metrics
    pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "R²": [r2]}).to_csv("Output/SARIMAX/sarimax_metrics.csv", index=False)
    print("✅ Forecast and metrics saved to Output/SARIMAX/")

if __name__ == "__main__":
    run_sarimax_forecast()
