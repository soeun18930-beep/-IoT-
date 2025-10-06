#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time, math, pickle, warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np, pandas as pd

POLLUTANTS = ("co2","vocs")
SENSORS = ("ndir","pid")

def now_local(): return datetime.now()
def log(m): print(m, flush=True)

@dataclass
class Calib:
    gain: float = 1.0
    offset: float = 0.0

@dataclass
class RuntimeCfg:
    pollutant: str
    sensor: str
    horizon_min: int = 15
    freq_min: int = 5
    th_on_low: float = 850.0
    th_off: float = 900.0
    eta_limit_min: float = 5.0
    target_limit: float = 1000.0
    min_on_min: int = 10
    calib: Calib = field(default_factory=Calib)
    plug_ip: Optional[str] = None
    mock: bool = False
    csv_path: Optional[str] = None
    model_path: Optional[str] = None
    poll_sec: int = 1

# -------- 센서 (모의/실) --------
class BaseSensor: 
    def read(self)->Tuple[float, Dict[str,Any]]: raise NotImplementedError

class MockNDIR(BaseSensor):
    def __init__(self): self.v=620.0
    def read(self):
        self.v += np.random.randn()*6 + 2.0
        return max(400.0, self.v), {"temp_c":24.0+np.random.randn()*0.2,"rh":45+np.random.randn()*1.0}

class MockPID(BaseSensor):
    def __init__(self): self.v=150.0
    def read(self):
        self.v += np.random.randn()*4 + 1.0
        return max(5.0, self.v), {"temp_c":24.0+np.random.randn()*0.2,"rh":45+np.random.randn()*1.0}

class NDIRSensor(BaseSensor):
    def __init__(self, port="/dev/ttyAMA0", baud=9600): self.port=port; self.baud=baud
    def read(self): raise NotImplementedError("NDIR 센서 드라이버 연결 필요")

class PIDSensor(BaseSensor):
    def __init__(self, port="/dev/ttyUSB0", baud=115200): self.port=port; self.baud=baud
    def read(self): raise NotImplementedError("PID 센서 드라이버 연결 필요")

def make_sensor(sensor: str, mock: bool)->BaseSensor:
    if mock: return MockNDIR() if sensor=="ndir" else MockPID()
    return NDIRSensor() if sensor=="ndir" else PIDSensor()

# -------- 전처리 --------
class Preprocessor:
    def __init__(self, pollutant: str, calib: Calib):
        self.pollutant=pollutant; self.calib=calib; self.buf:List[float]=[]
    def apply(self, raw: float, aux: Dict[str,Any])->float:
        self.buf.append(raw); 
        if len(self.buf)>5: self.buf.pop(0)
        v = float(np.median(self.buf))
        # VOCs 습도 보정 훅(필요 시 계수 학습 적용)
        rh = aux.get("rh")
        if self.pollutant=="vocs" and rh is not None:
            v = v * (1.0 + 0.0*(rh-50.0)/50.0)
        v = v*self.calib.gain + self.calib.offset
        return v

# -------- 예측 모델 --------
class SimpleRegModel:
    def __init__(self, steps): self.h=max(1,steps); self.alpha=0.9; self.level=None; self.trend=0.0
    def update(self,x):
        if self.level is None: self.level=x; self.trend=0.0
        else:
            nl = self.alpha*x + (1-self.alpha)*self.level
            self.trend = nl - self.level; self.level = nl
    def predict(self)->float:
        if self.level is None: return float("nan")
        return float(self.level + self.trend*self.h)

class ModelWrapper:
    def __init__(self, freq_min:int, horizon_min:int, model_path:Optional[str]):
        self.steps=max(1,horizon_min//freq_min); self.simple=SimpleRegModel(self.steps)
        self.model=None; self.feat_cols=None; self.target=None
        self.model_path=model_path
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path,"rb") as f: obj=pickle.load(f)
                if isinstance(obj,dict) and "model" in obj:
                    self.model=obj["model"]; self.feat_cols=obj.get("feat_cols"); self.target=obj.get("target")
                else:
                    self.model=obj
                log(f"[INFO] Model loaded: {model_path}")
            except Exception as e:
                warnings.warn(f"모델 로드 실패 → 단순 모델 사용: {e}")
    def update_and_predict(self, hist:pd.DataFrame, candidate_feats:List[str], target_col:str)->float:
        latest=float(hist[target_col].iloc[-1]); self.simple.update(latest)
        if self.model is None: return float(self.simple.predict())
        use_cols = [c for c in (self.feat_cols or candidate_feats) if c in hist.columns]
        if not use_cols: return float(self.simple.predict())
        X = hist[use_cols].tail(1).to_numpy()
        try:
            yhat = self.model.predict(X)
            return float(yhat[0] if np.ndim(yhat) else yhat)
        except Exception as e:
            warnings.warn(f"예측 실패 → 단순 모델: {e}")
            return float(self.simple.predict())

# -------- 플러그 제어 --------
class PlugController:
    def __init__(self, ip:Optional[str]):
        self.ip=ip; self.ready=False
        if ip:
            try:
                import asyncio; from kasa import SmartPlug
                self.asyncio=asyncio; self.SmartPlug=SmartPlug; self.ready=True
            except Exception:
                warnings.warn("python-kasa 미설치/초기화 실패 → 로그 모드")
    async def _switch_async(self,on:bool):
        plug=self.SmartPlug(self.ip); await plug.update()
        await (plug.turn_on() if on else plug.turn_off())
    def switch(self,on:bool):
        act="ON" if on else "OFF"
        if not self.ip or not self.ready: log(f"[CTRL] {act} (로그) ip={self.ip}"); return
        try: self.asyncio.run(self._switch_async(on)); log(f"[CTRL] 플러그 {self.ip} {act}")
        except Exception as e: warnings.warn(f"플러그 제어 실패({act}): {e}")

# -------- 메인 루프 --------
def run(cfg:RuntimeCfg):
    ALL = ["ts","co2_ppm","vocs_ppb","temp_c","rh"]
    df = pd.DataFrame(columns=ALL)
    target_col = "co2_ppm" if cfg.pollutant=="co2" else "vocs_ppb"
    unit = "ppm" if cfg.pollutant=="co2" else "ppb"

    sensor=None if cfg.csv_path else make_sensor(cfg.sensor, cfg.mock)
    pre = Preprocessor(cfg.pollutant,cfg.calib)
    model = ModelWrapper(cfg.freq_min,cfg.horizon_min,cfg.model_path)
    ctrl = PlugController(cfg.plug_ip)

    base_feats = ["co2_ppm","vocs_ppb","temp_c","rh"]
    fan_on=False; last_on_ts=None

    log(f"[INFO] pollutant={cfg.pollutant.upper()} h={cfg.horizon_min}m freq={cfg.freq_min}m "
        f"th_on_low={cfg.th_on_low}{unit} th_off={cfg.th_off}{unit} target={cfg.target_limit}{unit}")

    try:
        while True:
            ts = now_local().strftime("%Y-%m-%d %H:%M:%S")

            if cfg.csv_path:
                try:
                    raw = pd.read_csv(cfg.csv_path)
                    for c in ALL:
                        if c not in raw.columns: raw[c]=np.nan
                    raw = raw[ALL].dropna(how="all")
                    if raw.empty: time.sleep(cfg.poll_sec); continue
                    df = raw.tail(288)
                    latest=df.iloc[-1]; raw_value=latest[target_col]
                    aux={"temp_c":latest.get("temp_c"),"rh":latest.get("rh")}
                except Exception as e:
                    warnings.warn(f"CSV 읽기 실패: {e}"); time.sleep(cfg.poll_sec); continue
            else:
                val, aux = sensor.read()
                row = {"ts":ts,"co2_ppm":val if cfg.pollutant=="co2" else np.nan,
                              "vocs_ppb":val if cfg.pollutant=="vocs" else np.nan,
                              "temp_c":aux.get("temp_c",np.nan),"rh":aux.get("rh",np.nan)}
                df = pd.concat([df,pd.DataFrame([row])], ignore_index=True).tail(288)
                raw_value = row[target_col]

            if pd.isna(raw_value): time.sleep(cfg.poll_sec if cfg.csv_path else cfg.freq_min*60); continue

            proc = pre.apply(float(raw_value), aux)
            df.loc[df.index[-1], target_col] = proc

            feats=[c for c in base_feats if c in df.columns]
            yhat = model.update_and_predict(df, feats, target_col)

            steps=max(1,cfg.horizon_min//cfg.freq_min)
            per_step=(yhat - proc)/steps
            eta_to_limit=None
            if per_step>0:
                need=cfg.target_limit - proc
                if need>0: eta_to_limit = math.ceil(need/per_step)*cfg.freq_min

            log(f"[{ts}] {cfg.pollutant.upper()} now={proc:.0f}{unit} yhat{cfg.horizon_min}={yhat:.0f}{unit} "
                f"{('eta~%dm'%eta_to_limit) if eta_to_limit is not None else 'eta=NA'} fan={'ON' if fan_on else 'OFF'}")

            turn_on=False; turn_off=False
            if (yhat >= cfg.th_on_low) or (eta_to_limit is not None and eta_to_limit <= cfg.eta_limit_min):
                turn_on=True
            if fan_on:
                elapsed = (now_local()-last_on_ts).total_seconds()/60.0 if last_on_ts else cfg.min_on_min
                if (proc <= cfg.th_off) and (elapsed >= cfg.min_on_min): turn_off=True

            if turn_on and not fan_on:
                ctrl.switch(True); fan_on=True; last_on_ts=now_local()
            elif turn_off and fan_on:
                ctrl.switch(False); fan_on=False; last_on_ts=None

            time.sleep(cfg.poll_sec if cfg.csv_path else cfg.freq_min*60)
    except KeyboardInterrupt:
        log("[INFO] Stopped by user.")

# -------- CLI --------
def parse_args()->RuntimeCfg:
    ap = argparse.ArgumentParser("IAQ Predict & Control (CO2+VOCs)")
    ap.add_argument("--pollutant", choices=POLLUTANTS, default="co2")
    ap.add_argument("--sensor", choices=SENSORS, default="ndir")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--th_on_low", type=float, default=850.0)
    ap.add_argument("--th_off", type=float, default=900.0)
    ap.add_argument("--eta_limit", type=float, default=5.0)
    ap.add_argument("--target_limit", type=float, default=1000.0)
    ap.add_argument("--min_on", type=int, default=10)
    ap.add_argument("--plug_ip", type=str, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--csv", dest="csv_path", type=str, default=None)
    ap.add_argument("--model", dest="model_path", type=str, default=None)
    ap.add_argument("--poll_sec", type=int, default=1)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--offset", type=float, default=0.0)
    a = ap.parse_args()
    return RuntimeCfg(
        pollutant=a.pollutant, sensor=a.sensor, horizon_min=a.horizon, freq_min=a.freq_min,
        th_on_low=a.th_on_low, th_off=a.th_off, eta_limit_min=a.eta_limit, target_limit=a.target_limit,
        min_on_min=a.min_on, calib=Calib(gain=a.gain, offset=a.offset), plug_ip=a.plug_ip,
        mock=a.mock, csv_path=a.csv_path, model_path=a.model_path, poll_sec=a.poll_sec
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
