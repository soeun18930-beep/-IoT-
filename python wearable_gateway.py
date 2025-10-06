
from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
OUT = "wearable_stream.csv"
COLUMNS = ["ts", "heart_rate", "activity_level", "occupancy_est"]

def append_row(payload):
    df = pd.DataFrame([{
        "ts": payload.get("ts") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "heart_rate": payload.get("heart_rate"),
        "activity_level": payload.get("activity_level"),
        "occupancy_est": payload.get("occupancy_est")
    }], columns=COLUMNS)
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", index=False, header=header)

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
        # 값 기본 처리
        data.setdefault("activity_level", 0.0)   # 0~1 (휴식~격한활동)
        data.setdefault("occupancy_est", 1.0)    # 재실 추정(0/1 또는 0~1)
        append_row(data)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    # pip install flask pandas
    app.run(host="0.0.0.0", port=5001, debug=False)
