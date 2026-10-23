# GB Electricity Demand Forecasting (NESO open data)

Day-ahead forecasting of GB National Demand (ND, MW) for each half-hour settlement period, using NESO's Historic Demand Data. XGBoost is compared with two naive baselines on a strict chronological hold-out.

## Method
- **Data:** NESO Historic Demand Data, half-hourly outturn, 2019 and 2021 to 2 October 2026 (2020 excluded as a COVID-lockdown outlier year; the time grid keeps the gap so no lag crosses it).
- **Target:** National Demand (ND) in MW.
- **Features:** settlement period, day of week, month, day of year, weekend, GB bank holiday; demand lags at 24h, 48h and 7 days; rolling means of the previous day and previous week.
- **No leakage:** every feature is at least 24 hours older than the target half-hour, so the forecast could be issued the day before.
- **Split:** train 2019 to March 2025 (excluding 2020); validation (early stopping) April to October 2025; test the final 12 months, 2 Oct 2025 to 2 Oct 2026, never used for tuning or model choice.
- **Baselines:** same half-hour last week (seasonal naive); same half-hour yesterday.
- **Metrics:** MAPE, MAE, RMSE.

## Results (test set, 17,182 half-hours)

| Model | MAPE % | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| Seasonal naive (last week) | 8.20 | 2,085 | 2,835 |
| Naive (yesterday) | 7.84 | 1,959 | 2,717 |
| **XGBoost (calendar + demand lags)** | **5.56** | **1,359** | **1,848** |
| XGBoost (+ lagged embedded solar/wind) | 5.76 | 1,419 | 1,902 |
![One week of actual vs forecast demand](results/week_forecast.png)
The headline model is 29% lower in MAPE than the best baseline. The model with lagged embedded solar and wind was specified in advance as an ablation, and it did not help: yesterday's solar output says little about tomorrow's.

## Where the model struggles
- Errors are largest on very sunny days. ND excludes embedded solar, so a bright spring or summer midday can pull ND down by around 10 GW compared with the previous day. For example, on 29 April 2026 midday ND was about 16.6 GW against 25.2 GW the day before, and the worst half-hours (errors above 50%) all fall on such days.
- Weekends are harder than weekdays (MAPE 6.0 to 6.6% against 5.1 to 5.6%).
- The model has no weather input, which is the main reason for the errors above.

## Limitations and next steps
- Add day-ahead weather (solar irradiance, temperature, wind) to explain the midday solar swings.
- Forecast embedded solar separately, or model ND plus embedded generation as a total.
- Day-ahead point forecast only; no prediction intervals.
- Settlement periods 49 and 50 on clock-change days are dropped (about two half-hours a year).

## Run
```
pip install pandas scikit-learn xgboost matplotlib holidays
python gb_demand_forecast.py
```
Put the NESO yearly CSVs in `data/` named `demanddata_YYYY.csv`. Outputs go to `results/`. `--synthetic` only tests the code on fake data; never report those numbers.
