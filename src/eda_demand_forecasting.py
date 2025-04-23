import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_eda_pipeline():
    # === LOAD DATA ===
    demand = pd.read_excel('Data/EMA_Demand Data (2015-2025).xlsx')
    holiday = pd.read_excel('Data/sg_holiday_cny_covid_2015_2025.xlsx')

    # === DATE FORMATTING ===
    demand['Date'] = pd.to_datetime(demand['Date'])
    holiday['Date'] = pd.to_datetime(holiday['Date'])

    # === MAP HOLIDAY INFORMATION ===
    treat_as_map = dict(zip(holiday['Date'], holiday['Treat As']))
    holiday_name_map = dict(zip(holiday['Date'], holiday['Holiday Name']))

    # === COVID PERIOD ===
    covid_start = pd.to_datetime("2020-04-07")
    covid_end = pd.to_datetime("2021-08-10")

    def classify_daytype(row):
        date = row['Date']
        if covid_start <= date <= covid_end:
            return 'COVID'
        elif date in treat_as_map:
            return treat_as_map[date]
        elif row['Day'] == 'Sun':
            return 'Sun'
        elif row['Day'] == 'Sat':
            return 'Sat'
        else:
            return row['Day']

    demand['TreatAs_DayType'] = demand.apply(classify_daytype, axis=1)
    demand['Holiday Name'] = demand['Date'].apply(lambda x: holiday_name_map.get(x, 'Normal Day'))

    # === FEATURE ENGINEERING ===
    demand['Hour'] = pd.to_datetime(demand['Period Ending Time'], format='%H:%M').dt.hour
    demand['Minute'] = pd.to_datetime(demand['Period Ending Time'], format='%H:%M').dt.minute
    demand['DayOfWeek'] = demand['Date'].dt.dayofweek
    demand['Month'] = demand['Date'].dt.month
    demand['IsWeekend'] = demand['DayOfWeek'] >= 5
    demand['IsHoliday'] = demand['Holiday Name'] != 'Normal Day'
    demand['IsCOVID'] = (demand['Date'] >= covid_start) & (demand['Date'] <= covid_end)

    # === OUTPUT SETUP ===
    eda_dir = 'Output/EDA'
    os.makedirs(eda_dir, exist_ok=True)

    # === SAVE FEATURED DATASET ===
    demand.to_csv(f'{eda_dir}/demand_feature_dataset.csv', index=False)

    # === BASELINE ACCURACY OVERALL ===
    y_true = demand['NEM Demand (Actual)']
    y_pred = demand['NEM Demand (Forecast)']
    overall_stats = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    pd.DataFrame([overall_stats]).to_csv(f"{eda_dir}/baseline_overall_accuracy.csv", index=False)

    # === BASELINE ACCURACY BY DAYTYPE ===
    def get_metrics(group):
        y_true = group['NEM Demand (Actual)']
        y_pred = group['NEM Demand (Forecast)']
        return pd.Series({
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score(y_true, y_pred)
        })

    metrics_by_daytype = demand.groupby('TreatAs_DayType').apply(get_metrics).reset_index()
    metrics_by_daytype.to_csv(f"{eda_dir}/baseline_metrics_by_daytype.csv", index=False)

    # === TREND LINE CHART ===
    daily_avg = demand.groupby('Date')['NEM Demand (Actual)'].mean().reset_index()
    daily_avg['DateOrdinal'] = daily_avg['Date'].map(pd.Timestamp.toordinal)
    model = LinearRegression()
    model.fit(daily_avg[['DateOrdinal']], daily_avg['NEM Demand (Actual)'])
    daily_avg['Trend'] = model.predict(daily_avg[['DateOrdinal']])

    plt.figure(figsize=(14, 5))
    sns.lineplot(data=daily_avg, x='Date', y='NEM Demand (Actual)', label='Actual')
    sns.lineplot(data=daily_avg, x='Date', y='Trend', label='Trend', linestyle='--')
    plt.title('Daily Average NEM Demand with Trend Line (2015–2025)')
    plt.ylabel('MW')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/demand_trend_line.png", dpi=300)
    plt.close()

    # === BOX PLOT ===
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=demand, x='TreatAs_DayType', y='NEM Demand (Actual)', palette='Set3')
    plt.xticks(rotation=45)
    plt.title('Demand Distribution by Day Type')
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/demand_boxplot_by_daytype.png", dpi=300)
    plt.close()

    # === HOURLY PROFILES BY DAY TYPE ===
    avg_demand = demand.groupby(['TreatAs_DayType', 'Period Ending Time'])['NEM Demand (Actual)'].mean().reset_index()
    avg_demand['Time Label'] = avg_demand['Period Ending Time'].astype(str)

    weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=avg_demand[avg_demand['TreatAs_DayType'].isin(weekday_order)], x='Time Label', y='NEM Demand (Actual)', hue='TreatAs_DayType')
    plt.title('Weekday Demand Profiles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/weekday_profiles.png", dpi=300)
    plt.close()

    weekend_special = ['Sat', 'Sun', 'COVID', 'CNY']
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=avg_demand[avg_demand['TreatAs_DayType'].isin(weekend_special)], x='Time Label', y='NEM Demand (Actual)', hue='TreatAs_DayType')
    plt.title('Weekend, COVID & CNY Profiles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{eda_dir}/weekend_covid_cny_profiles.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_eda_pipeline()
