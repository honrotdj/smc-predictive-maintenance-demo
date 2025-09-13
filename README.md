# smc-predictive-maintenance-demo

Small, no-frills project. I fake a stream of sensor readings, train a simple model, and print a risk score for a new reading. It’s meant to be easy to run and easy to read.

> Demo only. Not production code.

---

## What it does (in plain words)

- **Simulate**: make a CSV of pretend readings (pressure, temperature, vibration, usage) with a simple device id and a couple of health flags.
- **Train**: fit a Logistic Regression model, save two small figures (ROC + confusion matrix), and save `model.joblib`.
- **Predict**: print a risk score for a single reading. If the row looks odd (unknown device / tamper flag), print a short security note.

---

## Quick start

```bash
pip install -r requirements.txt
python src/simulate.py --rows 1200
python src/train.py
python src/predict.py --from_csv data/sample.csv
```

That’s it. The figures end up in `reports/figures/roc.png` and `reports/figures/cm.png`.

---

## Files to know

- `src/simulate.py` – creates `data/sample.csv`  
  Columns: `timestamp, device_id, port_class, pressure, temperature, vibration, cycle_count, sensor_health, cable_loss_flag, tamper, fault`

- `src/train.py` – trains the model and saves:
  - `model.joblib`
  - `reports/figures/roc.png`
  - `reports/figures/cm.png`

- `src/predict.py` – prints a risk score and a short note

The tiny `devices.json` file is just a list of example device ids. There’s no real data source; everything is generated locally.

---

## Predict in a couple ways

**Use the last row of the CSV**
```bash
python src/predict.py --from_csv data/sample.csv
```

**Or pass your own numbers**
```bash
python src/predict.py --pressure 102 --temperature 64 --vibration 1.2 --cycle_count 1500
```

**Windows JSON example (cmd.exe)**
```bash
python src/predict.py --json "{\"pressure\":110,\"temperature\":71,\"vibration\":1.6,\"cycle_count\":4200,\"device_id\":\"UNKNOWN-777\",\"tamper\":1}"
```

If a row looks suspicious, you’ll see:
```
SECURITY NOTE -> unknown device id, tamper flag set
```

---

## Why I built it

Just to show I can put a small idea together end-to-end: clear scripts, comments, quick figures, simple CLI. It also reflects my IT/cyber mindset with a tiny check for odd device data.

---

## Requirements

Python 3.10+  
pandas, scikit-learn, matplotlib, numpy, joblib

---

## License

MIT
