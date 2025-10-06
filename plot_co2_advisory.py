import csv, os, math
from datetime import datetime
import matplotlib.pyplot as plt

MAIN_CSV = "co2_main_log.csv"
ADV_CSV  = "co2_advisory_log.csv"
OUT_PNG  = "co2_viz.png"

def read_main():
    xs,y,yhat,vent=[],[],[],[]
    if not os.path.exists(MAIN_CSV):
        raise SystemExit(f"[ERROR] {MAIN_CSV} 파일이 없습니다. 먼저 시뮬레이션을 실행하세요.")
    with open(MAIN_CSV,"r",encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try: ts=datetime.strptime(r["ts"],"%Y-%m-%d %H:%M:%S")
            except Exception: continue
            xs.append(ts)
            y.append(float(r.get("value_ppm","nan")))
            fh=r.get("forecast_ppm",None)
            yhat.append(float(fh) if fh not in (None,"") else math.nan)
            vo=r.get("vent_on","0")
            vent.append(int(vo) if str(vo).isdigit() else 0)
    return xs,y,yhat,vent

def read_adv():
    evs=[]
    if not os.path.exists(ADV_CSV): return evs
    with open(ADV_CSV,"r",encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try:
                s=datetime.strptime(r["ts_start"],"%Y-%m-%d %H:%M:%S")
                e=datetime.strptime(r["ts_end"],"%Y-%m-%d %H:%M:%S")
                evs.append({
                    "start":s,"end":e,
                    "max":float(r.get("max","nan")),
                    "level":(r.get("level","") or "").strip().upper()
                })
            except Exception: continue
    return evs

def main():
    xs,y,yhat,vent=read_main()
    evs=read_adv()
    plt.figure(figsize=(11,5))
    if xs:
        plt.plot(xs,y,label="CO₂ (ppm)",color="tab:blue")
        if any(not math.isnan(v) for v in yhat):
            plt.plot(xs,yhat,"--",label="Forecast (ppm)",color="tab:orange")
        seg_start=None
        for i in range(len(xs)):
            on=(vent[i]==1)
            if on and seg_start is None: seg_start=xs[i]
            if (not on or i==len(xs)-1) and seg_start is not None:
                end_ts=xs[i] if not on else xs[i]
                plt.axvspan(seg_start,end_ts,alpha=0.08,color="tab:cyan")
                seg_start=None
    for ev in evs:
        color="orange" if ev["level"]=="B" else "red"
        plt.axvspan(ev["start"],ev["end"],alpha=0.2,color=color)
        mid=ev["start"]+(ev["end"]-ev["start"])/2
        ymax=plt.gca().get_ylim()[1]
        plt.text(mid,ymax*0.9,f"Advisory {ev['level']} (max {ev['max']:.0f} ppm)",
                 ha="center",va="top",fontsize=9,color=color)
    plt.title("CO₂ Trend & Advisory Intervals")
    plt.xlabel("Time"); plt.ylabel("ppm")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_PNG,dpi=150)
    print(f"[INFO] 그래프 저장 완료 → {OUT_PNG}")

if __name__=="__main__":
    main()
