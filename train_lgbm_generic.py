#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pickle
from math import sqrt
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def add_time_features(df, ts_col="ts"):
    dt = pd.to_datetime(df[ts_col])
    df["hour"] = dt.dt.hour; df["dow"] = dt.dt.dayofweek
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24); df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"]  = np.sin(2*np.pi*df["dow"]/7);  df["cos_dow"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def add_lag_features(df, target, lags=(1,2,3), rolls=(3,6,12)):
    for l in lags: df[f"{target}_lag{l}"] = df[target].shift(l)
    for r in rolls:
        df[f"{target}_ma{r}"] = df[target].rolling(r, min_periods=1).mean()
        df[f"{target}_diffma{r}"] = df[target] - df[f"{target}_ma{r}"]
    df[f"{target}_diff1"] = df[target].diff(1)
    return df

def main():
    ap = argparse.ArgumentParser("LightGBM training for CO2/VOCs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", choices=["co2_ppm","vocs_ppb"], required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--model_out", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "ts" not in df.columns: raise ValueError("CSV에 ts 컬럼 필요")
    if args.target not in df.columns: raise ValueError(f"{args.target} 컬럼 필요")
    for c in ["temp_c","rh","co2_ppm","vocs_ppb"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_time_features(df, "ts")
    df = add_lag_features(df, args.target, lags=(1,2,3), rolls=(3,6,12))

    base_feats = [c for c in ["co2_ppm","vocs_ppb","temp_c","rh","hour","dow","sin_hour","cos_hour","sin_dow","cos_dow"] if c in df.columns]
    lag_feats  = [c for c in df.columns if c.startswith(args.target+"_lag") or c.startswith(args.target+"_ma") or c.startswith(args.target+"_diff")]
    feat_cols  = base_feats + lag_feats

    steps = max(1, args.horizon // args.freq_min)
    df["y"] = df[args.target].shift(-steps)
    use = df.dropna(subset=feat_cols+["y"]).reset_index(drop=True)
    X, y = use[feat_cols], use["y"]

    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros(len(X)); models=[]
    for tr, va in tscv.split(X):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]; X_va, y_va = X.iloc[va], y.iloc[va]
        m = LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=64,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0, random_state=42)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l2",
              callbacks=[early_stopping(100), log_evaluation(0)])
        oof[va] = m.predict(X_va); models.append(m)

    rmse = sqrt(mean_squared_error(y, oof)); mae = mean_absolute_error(y, oof)
    print(f"[OOF] RMSE={rmse:.1f} | MAE={mae:.1f}")

    best_iter = int(np.mean([getattr(m,"best_iteration_",None) or 1500 for m in models])) or 800
    final = LGBMRegressor(n_estimators=best_iter, learning_rate=0.03, num_leaves=64,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0, random_state=42)
    final.fit(X, y, callbacks=[log_evaluation(0)])

    bundle = {"model": final, "feat_cols": feat_cols, "target": args.target,
              "horizon_min": args.horizon, "freq_min": args.freq_min}
    out = args.model_out or (f"./models/{'co2' if args.target=='co2_ppm' else 'vocs'}_lgbm_t{args.horizon}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f: pickle.dump(bundle, f)
    print(f"[OK] Model saved → {out}")

if __name__ == "__main__":
    main()
