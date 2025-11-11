import os, io, sys, math, datetime as dt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== CONFIG ===============================================================
# OPTION A: Use a file inside the repo
LOCAL_INPUT_PATH = "Simulated_Student_Expenses.xlsx"   # commit this once (or replace with your real file name)

# OPTION B: Use a shared OneDrive link (uncomment and paste)
# ONEDRIVE_SHARED_URL = "https://.../Simulated_Student_Expenses.xlsx?download=1"

FORECAST_HORIZON = 7
# ==========================================================================

def load_expenses():
    # Option B: download from OneDrive shared link
    # if "ONEDRIVE_SHARED_URL" in globals() and ONEDRIVE_SHARED_URL:
    #     import requests
    #     r = requests.get(ONEDRIVE_SHARED_URL)
    #     r.raise_for_status()
    #     return pd.read_excel(io.BytesIO(r.content))

    # Option A: read file committed to repo
    if not os.path.exists(LOCAL_INPUT_PATH):
        print(f"ERROR: '{LOCAL_INPUT_PATH}' not found.", file=sys.stderr)
        sys.exit(1)
    return pd.read_excel(LOCAL_INPUT_PATH)

def prepare_daily(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    daily = (df.groupby("Date")["Amount (RM)"].sum()
               .reset_index()
               .rename(columns={"Amount (RM)":"Total_Spend"}))
    daily = daily.sort_values("Date").reset_index(drop=True)

    # cap extreme outliers to 1%/99% to reduce noise
    lo, hi = daily["Total_Spend"].quantile([0.01, 0.99])
    daily["Total_Spend"] = daily["Total_Spend"].clip(lo, hi)

    # features
    daily["Day_Number"] = np.arange(len(daily))
    daily["dow"] = daily["Date"].dt.dayofweek
    daily["dow_sin"] = np.sin(2*np.pi*daily["dow"]/7.0)
    daily["dow_cos"] = np.cos(2*np.pi*daily["dow"]/7.0)

    # rolling mean (7d) & lags
    daily["roll7"] = daily["Total_Spend"].rolling(7, min_periods=1).mean()
    daily["lag1"] = daily["Total_Spend"].shift(1)
    daily["lag7"] = daily["Total_Spend"].shift(7)
    daily = daily.dropna().reset_index(drop=True)
    daily = daily.set_index("Date")
    return daily

def train_eval(daily):
    feats = ["Day_Number","dow_sin","dow_cos","roll7","lag1","lag7"]
    X = daily[feats]
    y = daily["Total_Spend"]

    split = int(len(daily)*0.8)
    X_train, X_test = X.iloc[:split],  X.iloc[split:]
    y_train, y_test = y.iloc[:split],  y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = (abs((y_test - y_pred)/y_test).replace([np.inf, -np.inf], np.nan).dropna().mean())*100
    r2   = r2_score(y_test, y_pred)

    # Export test window for accuracy visuals
    test_df = pd.DataFrame({
        "Date": X_test.index,
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    test_df["APE_%"] = (abs(test_df["Actual"] - test_df["Predicted"]) /
                        test_df["Actual"].clip(lower=1e-8))*100
    test_df.to_csv("model_test_predictions.csv", index=False)

    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    return model, feats, split

def make_forecast(daily, model, feats, horizon=7):
    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future = pd.DataFrame(index=future_dates)
    start_dn = int(daily["Day_Number"].iloc[-1]) + 1
    future["Day_Number"] = np.arange(start_dn, start_dn + horizon)
    future["dow"] = future.index.dayofweek
    future["dow_sin"] = np.sin(2*np.pi*future["dow"]/7.0)
    future["dow_cos"] = np.cos(2*np.pi*future["dow"]/7.0)

    # roll7/lag features seeded from history
    work = pd.concat([daily[["Total_Spend"]], pd.DataFrame(index=future.index)], axis=0)
    work["roll7"] = work["Total_Spend"].rolling(7, min_periods=1).mean()
    work["lag1"]  = work["Total_Spend"].shift(1)
    work["lag7"]  = work["Total_Spend"].shift(7)

    # Fill forward day by day using predictions for lag/roll
    preds = []
    temp = work.copy()
    for d in future.index:
        row = {
            "Day_Number": future.loc[d, "Day_Number"],
            "dow_sin":    future.loc[d, "dow_sin"],
            "dow_cos":    future.loc[d, "dow_cos"],
            "roll7":      temp["Total_Spend"].iloc[-7:].mean(),
            "lag1":       temp["Total_Spend"].iloc[-1],
            "lag7":       temp["Total_Spend"].iloc[-7] if len(temp) >= 7 else temp["Total_Spend"].iloc[-1],
        }
        X_row = pd.DataFrame([row])[feats]
        yhat  = float(model.predict(X_row)[0])
        preds.append(yhat)

        # append to temp to keep features rolling
        temp.loc[d, "Total_Spend"] = yhat

    out = pd.DataFrame({
        "Date": future.index,
        "Predicted_Spend": preds
    })
    # simple ±10% band (cosmetic)
    out["Lower_Bound"] = out["Predicted_Spend"]*0.90
    out["Upper_Bound"] = out["Predicted_Spend"]*1.10
    out.to_csv("predicted_expense_next_week.csv", index=False)

def main():
    df = load_expenses()
    daily = prepare_daily(df)
    model, feats, split = train_eval(daily)
    make_forecast(daily, model, feats, FORECAST_HORIZON)

if __name__ == "__main__":
    main()
