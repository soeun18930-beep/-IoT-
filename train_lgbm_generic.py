#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pickle, csv, warnings, re
from math import sqrt
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore", category=UserWarning)

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _num_sanitize(s):
    """문자열에서 숫자만 추출(ex: '636 ppm' -> 636.0). 추출 실패시 NaN."""
    if pd.isna(s): return np.nan
    m = NUM_RE.search(str(s))
    return float(m.group()) if m else np.nan

def sanitize_numeric_cols(df, cols):
    """지정 컬럼을 숫자로 강제. 전부 NaN이면 숫자추출 방식으로 재시도."""
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0:  # '636ppm' 같은 형식 처리
                s = df[c].map(_num_sanitize)
            df[c] = s
    return df

def add_time_features(df, ts_col="ts"):
    dt = pd.to_datetime(df[ts_col], errors="coerce")
    # ts 파싱 실패 행 제거 (시간 파생에 NaN 생기면 이후 전부 드랍되므로 미리 정리)
    bad = dt.isna().sum()
    if bad:
        print(f"[WARN] ts 파싱 실패 {bad}행 제거")
    df = df.loc[~dt.isna()].copy()
    dt = pd.to_datetime(df[ts_col], errors="coerce")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"]  = np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def add_lag_features(df, target, lags=(1,2,3), rolls=(3,6,12)):
    for l in lags:
        df[f"{target}_lag{l}"] = df[target].shift(l)
    for r in rolls:
        ma = df[target].rolling(r, min_periods=1).mean()
        df[f"{target}_ma{r}"] = ma
        df[f"{target}_diffma{r}"] = df[target] - ma
    df[f"{target}_diff1"] = df[target].diff(1)
    return df

def smart_read_csv(path):
    """구분자 자동감지 + 헤더 공백 제거 + ts 파싱 시도(후 실패행 제거)."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    # ts 존재 시 즉시 datetime 변환(실패행은 add_time_features에서 정리)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df

def alias_target_if_needed(df, target):
    """타깃 컬럼이 없을 때 흔한 별칭을 매핑해 생성."""
    if target in df.columns:
        return df, target
    # 흔한 별칭 후보
    if target == "co2_ppm":
        candidates = ["co2", "co2(ppm)", "co2_ppm ", "CO2", "CO2_PPM"]
    else:  # vocs_ppb
        candidates = ["voc", "vocs", "tvoc", "tvocs", "voc_ppb", "vocs(ppb)","TVOC"]
    for c in candidates:
        if c in df.columns:
            print(f"[INFO] '{target}' 없음 → '{c}'를 '{target}'로 사용")
            df[target] = df[c]
            return df, target
    raise ValueError(f"{target} 컬럼 필요(별칭도 없음). CSV 헤더 확인.")

def build_dataset(df, target, horizon, freq_min, lags=(1,2,3), rolls=(3,6,12)):
    base_feats = [c for c in ["co2_ppm","vocs_ppb","temp_c","rh",
                              "hour","dow","sin_hour","cos_hour","sin_dow","cos_dow"]
                  if c in df.columns]
    df2 = add_lag_features(df.copy(), target, lags=lags, rolls=rolls)
    lag_feats = [c for c in df2.columns if c.startswith(target+"_lag")
                 or c.startswith(target+"_ma") or c.startswith(target+"_diff")]
    feat_cols = base_feats + lag_feats

    steps = max(1, horizon // max(1, freq_min))
    df2["y"] = df2[target].shift(-steps)

    use = df2.dropna(subset=["y"]).fillna(method="bfill").fillna(method="ffill").reset_index(drop=True)
    X, y = use[feat_cols], use["y"]
    return X, y, feat_cols, steps

def choose_cv_splits(n_samples):
    if n_samples < 3:
        return 0
    n_splits = min(5, max(2, n_samples // 50))
    n_splits = min(n_splits, max(2, n_samples-1))
    if n_splits >= n_samples:
        return 0
    return n_splits

def main():
    ap = argparse.ArgumentParser("LightGBM training for CO2/VOCs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", choices=["co2_ppm","vocs_ppb"], required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--model_out", type=str, default=None)
    args = ap.parse_args()

    # 1) CSV 로드 + 기본 정리
    df = smart_read_csv(args.csv)
    if "ts" not in df.columns:
        raise ValueError("CSV에 ts 컬럼 필요")

    # 2) 타깃 별칭 허용
    df, target = alias_target_if_needed(df, args.target)

    # 3) 숫자 칼럼 정제(단위/문자 제거)
    num_candidates = ["temp_c","rh","co2_ppm","vocs_ppb"]
    df = sanitize_numeric_cols(df, [c for c in num_candidates if c in df.columns])

    # 4) 시간 파생 + 정렬
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_time_features(df, "ts")

    # 5) 데이터 진단 로그
    def nn(col): return df[col].notna().sum() if col in df.columns else -1
    print(f"[INFO] rows={len(df)} nn(ts)={nn('ts')} nn({target})={nn(target)} nn(temp_c)={nn('temp_c')} nn(rh)={nn('rh')}")

    # 6) 전처리 완화 루프
    trials = [
        dict(lags=(1,2,3), rolls=(3,6,12)),
        dict(lags=(1,2),   rolls=(3,6)),
        dict(lags=(1,),    rolls=(3,)),
        dict(lags=(1,),    rolls=()),  # 최소
    ]
    X = y = feat_cols = None
    for t in trials:
        X_, y_, feats_, steps = build_dataset(df, target, args.horizon, args.freq_min,
                                              lags=t["lags"], rolls=t["rolls"])
        if len(X_) > 0:
            X, y, feat_cols = X_, y_, feats_
            print(f"[INFO] features OK with lags={t['lags']} rolls={t['rolls']} | samples={len(X)}")
            break
        else:
            print(f"[WARN] too few samples after drop (lags={t['lags']}, rolls={t['rolls']}). trying simpler features...")

    if X is None or len(X) == 0:
        # 마지막 구조화 로그 찍고 종료
        print("[DEBUG] columns:", list(df.columns))
        print("[DEBUG] head:\n", df.head(5))
        raise RuntimeError("전처리 후 유효 샘플이 0입니다. CSV 구분자/헤더/결측, 단위문자(예: 'ppm') 포함 여부를 확인하세요.")

    # 7) CV 분할 자동 결정
    n_samples = len(X)
    n_splits = choose_cv_splits(n_samples)

    oof = np.zeros(n_samples)
    models = []

    if n_splits >= 2:
        print(f"[INFO] Using TimeSeriesSplit n_splits={n_splits} (n_samples={n_samples})")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X), 1):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_va, y_va = X.iloc[va], y.iloc[va]
            if len(X_va) == 0 or len(X_tr) == 0:
                print(f"[WARN] fold {i}: empty train/valid; skipping")
                continue
            m = LGBMRegressor(
                n_estimators=1500, learning_rate=0.03, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
                random_state=42
            )
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="l2",
                callbacks=[early_stopping(100), log_evaluation(0)]
            )
            oof[va] = m.predict(X_va)
            models.append(m)
        if len(models) == 0:
            print("[WARN] all folds skipped; falling back to single fit on full data")
            m = LGBMRegressor(
                n_estimators=800, learning_rate=0.03, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
                random_state=42
            )
            m.fit(X, y, callbacks=[log_evaluation(0)])
            models = [m]
            oof[:] = m.predict(X)
    else:
        print(f"[INFO] Too few samples for CV (n_samples={n_samples}); fitting single model")
        m = LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
            random_state=42
        )
        m.fit(X, y, callbacks=[log_evaluation(0)])
        models = [m]
        oof[:] = m.predict(X)

    # 성능 리포트
    rmse = sqrt(mean_squared_error(y, oof))
    mae = mean_absolute_error(y, oof)
    print(f"[OOF] RMSE={rmse:.1f} | MAE={mae:.1f}")

    # 최종 모델 조립
    def _best_iter(m): return getattr(m, "best_iteration_", None)
    bests = [b for b in map(_best_iter, models) if b is not None]
    best_iter = (int(np.mean(bests)) if bests else (800 if n_splits < 2 else 1500))
    if best_iter <= 0:
        best_iter = 800 if n_splits < 2 else 1500

    final = LGBMRegressor(
        n_estimators=best_iter, learning_rate=0.03, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
        random_state=42
    )
    final.fit(X, y, callbacks=[log_evaluation(0)])

    bundle = {
        "model": final,
        "feat_cols": list(X.columns),
        "target": target,
        "horizon_min": args.horizon,
        "freq_min": args.freq_min,
    }
    out = args.model_out or (f"./models/{'co2' if target=='co2_ppm' else 'vocs'}_lgbm_t{args.horizon}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Model saved → {out}")

if __name__ == "__main__":
    main()

