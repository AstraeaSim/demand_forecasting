
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def run_sarimax_forecast_with_outputs(
    df,
    target_col,
    exog_cols,
    output_dir="Output/Sarimax",
    train_range=('2019-01-01', '2020-01-01'),
    test_start='2023-01-01',
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 48)
):
    """
    Runs SARIMAX model and saves forecast outputs and graph.
    Uses integer-based prediction to avoid shape mismatch issues and adds optimization method.

    Parameters:
    - df (pd.DataFrame): DataFrame indexed by datetime.
    - target_col (str): Column name of the target variable.
    - exog_cols (list): List of exogenous feature column names.
    - output_dir (str): Directory to save outputs.
    - train_range (tuple): Start and end date for training (YYYY-MM-DD).
    - test_start (str): Start date for testing (YYYY-MM-DD).
    - order (tuple): ARIMA (p, d, q) order.
    - seasonal_order (tuple): Seasonal order (P, D, Q, s).

    Returns:
    - results (SARIMAXResults): Fitted model.
    - forecast_df (pd.DataFrame): DataFrame with Actual vs Forecast.
    - metrics (dict): MAE, RMSE, R² scores.
    - plot_path (str): File path of saved plot image.
    """

    os.makedirs(output_dir, exist_ok=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.freq = pd.infer_freq(df.index)

    train = df.loc[train_range[0]:train_range[1]]
    test = df.loc[test_start:]

    # Reindex to ensure continuous 30min intervals in test set
    full_range = pd.date_range(start=test.index.min(), end=test.index.max(), freq='30min')
    test = test.reindex(full_range)
    test = test.dropna(subset=exog_cols + [target_col])

    y_train = train[target_col]
    X_train = train[exog_cols]
    y_test = test[target_col]
    X_test = test[exog_cols]

    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False, method='powell')

    # Use integer-based indexing for prediction to avoid shape mismatch
    start = len(y_train)
    end = start + len(y_test) - 1
    prediction = results.get_prediction(start=start, end=end, exog=X_test, dynamic=False)
    
    forecast = prediction.predicted_mean

    # Ensure forecast length matches y_test
    forecast = forecast[:len(y_test)]

    forecast_df = pd.DataFrame({
    'Datetime': y_test.index,
    'Actual': y_test.values,
    'Forecast': forecast.values
})
    forecast_df.to_csv(f"{output_dir}/sarimax_forecast.csv", index=False)
    
    # Evaluate
    mae = mean_absolute_error(y_test, forecast)
    rmse = root_mean_squared_error(y_test, forecast)
    r2 = r2_score(y_test, forecast)
    
    metrics_df = pd.DataFrame([{'MAE': mae, 'RMSE': rmse, 'R2': r2}])
    metrics_df.to_csv(f"{output_dir}/sarimax_metrics.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
    plt.plot(forecast.index, forecast, label='Forecast', alpha=0.7)
    plt.title('SARIMAX Forecast vs Actual')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"{output_dir}/sarimax_forecast_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return results, forecast_df, {'MAE': mae, 'RMSE': rmse, 'R2': r2}, plot_path
