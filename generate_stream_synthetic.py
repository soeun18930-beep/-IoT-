#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def synth_series(kind: str, rows: int, step_min: int = 5, base=600, amp=200, noise=10):
    t0 = datetime.now().replace(second=0, microsecond=0)
    ts = [t0 + timedelta(minutes=step_min*i) for i in range(rows)]
    x = np.linspace(0, 2.5*np.pi, rows)
    if kind == "co2":
        y = base + amp*np.sin(x) + np.random.normal(0, noise, rows)
        y = np.clip(y, 400, None)
        return pd.DataFrame({"ts": ts, "co2_ppm": y.round(0), "vocs_ppb": np.nan})
    else:
        base_v = 150; amp_v = 120; noise_v = 20
        y = base_v + amp_v*np.sin(x) + np.random.normal(0, noise_v, rows)
        y = np.clip(y, 5, None)
        return pd.DataFrame({"ts": ts, "co2_ppm": np.nan, "vocs_ppb": y.round(0)})

def main():
    ap = argparse.ArgumentParser("Synthetic IAQ stream generator")
    ap.add_argument("--dst", required=True, help="출력 CSV(append 모드)")
    ap.add_argument("--kind", choices=["co2","vocs"], default="co2")
    ap.add_argument("--rows", type=int, default=288, help="행 수(5분×24h=288)")
    ap.add_argument("--interval_sec", type=int, default=0, help="0이면 한번에 저장, >0이면 실시간 append")
    ap.add_argument("--virtual_step_min", type=int, default=5)
    ap.add_argument("--temp_c", type=float, default=24.0)
    ap.add_argument("--rh", type=float, default=45.0)
    args = ap.parse_args()

    df = synth_series(args.kind, args.rows, step_min=args.virtual_step_min)
    df["temp_c"] = args.temp_c + np.random.normal(0, 0.2, len(df))
    df["rh"]     = args.rh + np.random.normal(0, 1.0, len(df))

    cols = ["ts","co2_ppm","vocs_ppb","temp_c","rh"]

    if args.interval_sec <= 0:
        df[cols].to_csv(args.dst, index=False)
        print(f"[OK] saved → {args.dst} ({len(df)} rows)")
        return

    # 실시간 append 모드
    print(f"[INFO] streaming → {args.dst} (every {args.interval_sec}s)")
    for i in range(len(df)):
        one = df.iloc[[i]][cols]
        header = not (i or (pd.io.common.file_exists(args.dst)))
        one.to_csv(args.dst, index=False, mode="a", header=header)
        print(one.to_dict("records")[0])
        time.sleep(args.interval_sec)

if __name__ == "__main__":
    main()
