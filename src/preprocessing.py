import pandas as pd
# Load base files
demand = pd.read_excel("Data/EMA_Demand Data (2015-2025).xlsx")
holiday = pd.read_excel("Data/sg_holiday_cny_covid_2015_2025.xlsx")

# Convert dates
demand['Date'] = pd.to_datetime(demand['Date'])
holiday['Date'] = pd.to_datetime(holiday['Date'])

# Mapping
treat_as_map = dict(zip(holiday['Date'], holiday['Treat As']))

# COVID range
covid_start = pd.to_datetime("2020-04-07")
covid_end = pd.to_datetime("2021-08-10")

def classify_daytype(row):
    date = row['Date']
    if covid_start <= date <= covid_end:
        return 'COVID'
    elif date in treat_as_map:
        return treat_as_map[date]
    elif row['Day'] == 'Sat':
        return 'Sat'
    elif row['Day'] == 'Sun':
        return 'Sun'
    else:
        return row['Day']

# Add new column
demand['TreatAs_DayType'] = demand.apply(classify_daytype, axis=1)

# Add lag features (reuse your previous code here)
# Combine Date + Period Ending Time
demand['Datetime'] = pd.to_datetime(demand['Date'].astype(str) + ' ' + demand['Period Ending Time'])
demand.set_index('Datetime', inplace=True)

# Create lag features
for col in ['NEM Demand (Actual)', 'NEM Demand (Forecast)']:
    for lag in range(1, 4):
        demand[f'{col}_lag{lag}'] = demand[col].shift(lag)

# Drop nulls from lagging
demand_lagged = demand.dropna()

# Save
demand_lagged.to_csv("Data/EMA_Demand_Lagged.csv")