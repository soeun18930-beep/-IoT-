#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wearable Synthetic Generator
----------------------------
- 배치 모드(batch): 하루치 데이터를 일괄 CSV로 생성
- 스트림 모드(stream): interval_sec 간격으로 실시간 append
- 출력 컬럼: ts, heart_rate, activity_level, occupancy_est
"""



import argparse, time, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def make_schedule(rows:int, step_min:int, day_profile:str):
    """시간대별 활동/재실 패턴(사무실 프로파일 예시)"""
    t0 = datetime.now().replace(second=0, microsecond=0)
    ts = [t0 + timedelta(minutes=step_min*i) for i in range(rows)]

    hour = np.array([t.hour for t in ts])
    # 기본 점유 확률/활동 (0~1), 필요시 조정
    if day_profile == "office":
        occ_base = np.where((hour>=9)&(hour<12), 1.0,
                    np.where((hour>=13)&(hour<18), 1.0, 0.2))
        act_base = np.where((hour>=9)&(hour<12), 0.45,
                    np.where((hour>=13)&(hour<18), 0.5, 0.15))
    else:  # 'home' 등 간단 샘플
        occ_base = np.where((hour>=7)&(hour<9), 1.0,
                    np.where((hour>=18)&(hour<23), 1.0, 0.3))
        act_base = np.where((hour>=7)&(hour<9), 0.35,
                    np.where((hour>=18)&(hour<23), 0.4, 0.1))

    # 랜덤성과 일중 리듬
    x = np.linspace(0, 2*np.pi, rows)
    act = np.clip(act_base + 0.1*np.sin(2*x) + np.random.normal(0, 0.05, rows), 0, 1)
    occ = np.clip(occ_base + np.random.normal(0, 0.05, rows), 0, 1)

    # 심박수: 기준 70bpm에서 활동/점유/일중리듬 반영
    hr = 70 + (act*25) + (occ*10) + 4*np.sin(x) + np.random.normal(0, 2.5, rows)
    hr = np.clip(hr, 55, 130)

    df = pd.DataFrame({
        "ts": ts,
        "heart_rate": hr.round(0),
        "activity_level": act.round(2),
        "occupancy_est": occ.round(2),
    })
    return df

def main():
    ap = argparse.ArgumentParser("Wearable synthetic generator")
    ap.add_argument("--dst", required=True, help="출력 CSV 경로")
    ap.add_argument("--rows", type=int, default=288, help="행수(5분×24h=288)")
    ap.add_argument("--step_min", type=int, default=5, help="샘플 간격(분)")
    ap.add_argument("--profile", choices=["office","home"], default="office")
    ap.add_argument("--interval_sec", type=int, default=0, help=">0이면 실시간 append 모드")
    args = ap.parse_args()

    df = make_schedule(args.rows, args.step_min, args.profile)

    cols = ["ts","heart_rate","activity_level","occupancy_est"]
    if args.interval_sec <= 0:
        df[cols].to_csv(args.dst, index=False)
        print(f"[OK] saved → {args.dst} ({len(df)} rows)")
        return

    print(f"[INFO] streaming → {args.dst} every {args.interval_sec}s")
    for i in range(len(df)):
        one = df.iloc[[i]][cols]
        header = not (i or os.path.exists(args.dst))
        one.to_csv(args.dst, index=False, mode="a", header=header)
        print(one.to_dict('records')[0], flush=True)
        time.sleep(args.interval_sec)

if __name__ == "__main__":
    main()
