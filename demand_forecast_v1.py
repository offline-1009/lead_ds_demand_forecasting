
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset
fact = pd.read_csv('lead_ds_interview_fact_v2.csv')
fact['month'] = pd.to_datetime(fact['month'])
series_df = fact.groupby(['month','region','product_id'])['volume_sold'].sum().reset_index()

# Global series
global_df = series_df.groupby('month')['volume_sold'].sum().reset_index().rename(columns={'volume_sold':'global_volume'})
last_month = pd.to_datetime(global_df['month']).max()
val_months = pd.date_range(end=last_month, periods=6, freq='MS')

y = global_df.copy(); y['month'] = pd.to_datetime(y['month']); y = y.set_index('month')['global_volume'].asfreq('MS')
train = y[~y.index.isin(val_months)]
val = y[y.index.isin(val_months)]

fit = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
fc_val_glob = fit.forecast(6)

# Compute global accuracy
comp_glob = pd.DataFrame({'actual': val, 'forecast': fc_val_glob}).dropna()
comp_glob['abs_err'] = (comp_glob['actual'] - comp_glob['forecast']).abs()
WAPE_glob = float(comp_glob['abs_err'].sum() / comp_glob['actual'].sum())
sMAPE_glob = float((200*comp_glob['abs_err']/(comp_glob['actual'].abs()+comp_glob['forecast'].abs())).mean())

# Fixed shares based on train period
train_mix = series_df[~series_df['month'].isin(val_months)].copy()
train_tot = train_mix.groupby('month')['volume_sold'].sum().reset_index().rename(columns={'volume_sold':'total'})
train_mix = train_mix.merge(train_tot, on='month')
train_mix['share'] = train_mix['volume_sold']/train_mix['total']
avg_share = train_mix.groupby(['region','product_id'], as_index=False)['share'].mean()

# Allocate future forecast (6 months beyond last history)
full_fit = ExponentialSmoothing(y, trend='add', seasonal='mul', seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=6, freq='MS')
fc_future_glob = full_fit.forecast(6)

alloc_future_rows = []
for m, gv in fc_future_glob.items():
    for _, row in avg_share.iterrows():
        alloc_future_rows.append({
            'month': m.strftime('%Y-%m-01'),
            'region': row['region'],
            'product_id': row['product_id'],
            'forecast_volume_mt': round(float(gv*row['share']),2),
            'model': 'HW (global + share allocation)'
        })

# Add a GLOBAL roll-up row for clarity
global_future = pd.DataFrame({
    'month': [m.strftime('%Y-%m-01') for m in future_months],
    'region': 'GLOBAL',
    'product_id': 'ALL',
    'forecast_volume_mt': [round(float(v),2) for v in fc_future_glob.values],
    'model': 'Global-only HW (sum)'
})

forecast = pd.concat([pd.DataFrame(alloc_future_rows), global_future], ignore_index=True)
forecast.to_csv('demand_forecast_v1.csv', index=False)

print('Wrote demand_forecast_v1.csv; global validation WAPE=%.3f, sMAPE=%.2f%%' % (WAPE_glob, 100*sMAPE_glob))
