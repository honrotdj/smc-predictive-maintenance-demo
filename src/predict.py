import argparse
import json
import numpy as np
import pandas as pd
from joblib import load

# Loads model.joblib and prints a risk score for one reading.
# Keep comments short / human.

def load_model(path="model.joblib"):
    return load(path)

def predict_one(model, pressure, temperature, vibration, cycle_count):
    x = np.array([[pressure, temperature, vibration, cycle_count]], dtype=float)
    proba = model.predict_proba(x)[0, 1]
    label = int(proba >= 0.5)
    return proba, label

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
    # Option A: pass values on the CLI
    p.add_argument("--pressure", type=float, help="PSI")
    p.add_argument("--temperature", type=float, help="Celsius")
    p.add_argument("--vibration", type=float, help="g")
    p.add_argument("--cycle_count", type=float, help="since last maintenance")
    # Option B: read a JSON string or file
    p.add_argument("--json", type=str, help='Inline JSON like {"pressure":101,"temperature":60,"vibration":1.1,"cycle_count":500}')
    p.add_argument("--json_file", type=str, help="Path to JSON file with the same keys")
    # Option C: pull a row from CSV by index (default: last row)
    p.add_argument("--from_csv", type=str, help="CSV path to read a row from (e.g., data/sample.csv)")
    p.add_argument("--row_index", type=int, default=-1, help="Row index to use if --from_csv is set (default: last row)")
    p.add_argument("--model", type=str, default="model.joblib")
    return p.parse_args()

def resolve_input(args):
    # B1: JSON string
    if args.json:
        d = json.loads(args.json)
        return float(d["pressure"]), float(d["temperature"]), float(d["vibration"]), float(d["cycle_count"])
    # B2: JSON file
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["pressure"]), float(d["temperature"]), float(d["vibration"]), float(d["cycle_count"])
    # C: CSV row
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        row = df.iloc[args.row_index]
        return float(row["pressure"]), float(row["temperature"]), float(row["vibration"]), float(row["cycle_count"])
    # A: plain CLI numbers
    required = [args.pressure, args.temperature, args.vibration, args.cycle_count]
    if all(v is not None for v in required):
        return map(float, required)

    raise SystemExit("No input provided. Use --json / --json_file / --from_csv or pass --pressure --temperature --vibration --cycle_count.")

def main():
    args = cli_args()
    model = load_model(args.model)
    pressure, temperature, vibration, cycle_count = resolve_input(args)
    prob, label = predict_one(model, pressure, temperature, vibration, cycle_count)

    print("== Prediction ==")
    print(f"Input  -> pressure={pressure:.3f}, temperature={temperature:.3f}, vibration={vibration:.3f}, cycle_count={cycle_count:.0f}")
    print(f"Risk   -> {prob:.3f} (0=low, 1=high)")
    print(f"Label  -> {label}")
    print(f"Note   -> {text_recommendation(prob)}")

if __name__ == "__main__":
    main()
