import time, random, requests
from datetime import datetime

GATEWAY = "http://127.0.0.1:5001/ingest"   # 같은 PC면 localhost, 폰이면 PC IP

def sample_payload():
    # 간단한 시뮬: 휴식/가벼운 활동/활동 사이를 오가며 값 변동
    hr = random.randint(65, 98)                  # 심박수
    act = max(0.0, min(1.0, random.random()*0.6))# 0~1
    occ = 1.0                                    # 재실(있음)
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "heart_rate": hr,
        "activity_level": round(act, 2),
        "occupancy_est": occ
    }

if __name__ == "__main__":
    while True:
        payload = sample_payload()
        try:
            r = requests.post(GATEWAY, json=payload, timeout=3)
            print("[POST]", payload, "->", r.status_code)
        except Exception as e:
            print("[ERR]", e)
        time.sleep(5)  # 5초마다 전송
