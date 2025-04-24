
import pandas as pd
def generate_sarimax_merged(output_path='Data/SARIMAX_merged.csv'):
    df_lagged = pd.read_csv('Data/EMA_Demand_Lagged.csv', parse_dates=['Datetime'])
    df_feature = pd.read_csv('Data/demand_feature_dataset.csv')

    # Create Datetime and merge
    df_feature['Datetime'] = pd.to_datetime(df_feature['Date'].astype(str) + ' ' + df_feature['Period Ending Time'])
    merged_df = pd.merge(df_lagged, df_feature, on='Datetime', how='outer')

    # Clean and rename columns
    merged_df.drop(columns=[
        'Date_y', 'Day_y', 'Period Ending Time_y',
        'System Demand (Actual)_y', 'NEM Demand (Actual)_y', 'NEM Demand (Forecast)_y'
    ], inplace=True, errors='ignore')

    merged_df.rename(columns={
        'Date_x': 'Date', 'Day_x': 'Day', 'Period Ending Time_x': 'Period Ending Time',
        'System Demand (Actual)_x': 'System Demand (Actual)',
        'NEM Demand (Actual)_x': 'NEM Demand (Actual)',
        'NEM Demand (Forecast)_x': 'NEM Demand (Forecast)'
    }, inplace=True)

    merged_df.dropna(inplace=True)
    merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']] = merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']].astype(int)

    # Ensure Datetime is a column, not index
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(output_path, index=False)

    df = pd.read_csv(output_path, parse_dates=['Datetime'])
    # print("test")
    # print(df.columns.tolist())  # Debugging: Check if 'Datetime' is really present
    df.set_index('Datetime', inplace=True)
    df.index.freq = pd.infer_freq(df.index)
    return df