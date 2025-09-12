import argparse
import json
import numpy as np
import pandas as pd
from joblib import load

# load the trained model
def load_model(path="model.joblib"):
    return load(path)

# single prediction
def predict_one(model, pressure, temperature, vibration, cycle_count):
    x = np.array([[pressure, temperature, vibration, cycle_count]], dtype=float)
    proba = model.predict_proba(x)[0, 1]
    label = int(proba >= 0.5)
    return proba, label

# tiny maintenance note
def text_recommendation(prob):
    if prob >= 0.80:
        return "High risk — schedule maintenance ASAP."
    if prob >= 0.50:
        return "Elevated risk — inspect soon."
    if prob >= 0.30:
        return "Watch — consider a quick check."
    return "Low risk — keep running."

def cli_args():
    p = argparse.ArgumentParser()
    # Option A: raw numbers
    p.add_argument("--pressure", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--vibration", type=float)
    p.add_argument("--cycle_count", type=float)
    # Option B: JSON input
    p.add_argument("--json", type=str)
    p.add_argument("--json_file", type=str)
    # Option C: pull a row from CSV
    p.add_argument("--from_csv", type=str)
    p.add_argument("--row_index", type=int, default=-1)  # default last row
    p.add_argument("--model", type=str, default="model.joblib")
    return p.parse_args()

# pull optional metadata if present (device_id/tamper)
def extract_optional(d):
    device_id = d.get("device_id")
    tamper = int(d.get("tamper", 0)) if "tamper" in d else 0
    unauthorized = 1 if (device_id and str(device_id).startswith("UNKNOWN")) else 0
    return device_id, tamper, unauthorized

def resolve_input(args):
    # JSON string
    if args.json:
        d = json.loads(args.json)
        device_id, tamper, unauthorized = extract_optional(d)
        return (float(d["pressure"]), float(d["temperature"]), float(d["vibration"]), float(d["cycle_count"]),
                device_id, tamper, unauthorized)

    # JSON file
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        device_id, tamper, unauthorized = extract_optional(d)
        return (float(d["pressure"]), float(d["temperature"]), float(d["vibration"]), float(d["cycle_count"]),
                device_id, tamper, unauthorized)

    # CSV row
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        row = df.iloc[args.row_index]
        d = row.to_dict()
        device_id, tamper, unauthorized = extract_optional(d)
        return (float(row["pressure"]), float(row["temperature"]), float(row["vibration"]), float(row["cycle_count"]),
                device_id, tamper, unauthorized)

    # plain CLI numbers
    required = [args.pressure, args.temperature, args.vibration, args.cycle_count]
    if all(v is not None for v in required):
        return (float(args.pressure), float(args.temperature), float(args.vibration), float(args.cycle_count),
                None, 0, 0)

    raise SystemExit("No input provided. Use --json / --json_file / --from_csv or pass four numeric flags.")

def main():
    args = cli_args()
    model = load_model(args.model)
    pressure, temperature, vibration, cycle_count, device_id, tamper, unauthorized = resolve_input(args)

    prob, label = predict_one(model, pressure, temperature, vibration, cycle_count)

    print("== Prediction ==")
    if device_id:
        print(f"Device -> {device_id}")
    print(f"Input  -> pressure={pressure:.3f}, temperature={temperature:.3f}, vibration={vibration:.3f}, cycle_count={cycle_count:.0f}")
    print(f"Risk   -> {prob:.3f} (0=low, 1=high)")
    print(f"Label  -> {label}")

    # security note if something looks off
    if tamper or unauthorized:
        notes = []
        if unauthorized:
            notes.append("unknown device id")
        if tamper:
            notes.append("tamper flag set")
        print("SECURITY NOTE -> " + ", ".join(notes))

    print(f"Note   -> {text_recommendation(prob)}")

if __name__ == "__main__":
    main()


