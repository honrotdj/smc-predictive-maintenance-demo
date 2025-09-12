# Predictive Maintenance Demo (SMC-inspired)

Beginner-friendly, runnable demo. It **simulates device readings** (pressure / temperature / vibration / usage) with a simple device identity + health flags, **trains a Logistic Regression model**, and **prints a failure-risk score** for new data.  
If a row looks suspicious (**unknown device** / **tamper flag**), it also prints a short **SECURITY NOTE** — tying reliability to basic OT security hygiene.

> This is a small learning project, not production software.

---

## Quickstart

```bash
pip install -r requirements.txt
python src/simulate.py --rows 1200
python src/train.py
python src/predict.py --from_csv data/sample.csv
```

**What these do**  
- `simulate.py` → writes `data/sample.csv` with columns:  
  `timestamp, device_id, port_class, pressure, temperature, vibration, cycle_count, sensor_health, cable_loss_flag, tamper, fault`
- `train.py` → trains a Logistic Regression and saves:
  - `model.joblib`
  - figures: `reports/figures/roc.png`, `reports/figures/cm.png`
- `predict.py` → prints a risk score for a new reading (0=low, 1=high) and a maintenance note.  
  If `device_id` is `UNKNOWN-###` or `tamper==1`, it also prints a **SECURITY NOTE**.

---

## Predict in three ways

**A) Use the last row of the CSV**  
```bash
python src/predict.py --from_csv data/sample.csv
```

**B) Inline JSON (Windows cmd quoting example)**  
```bash
python src/predict.py --json "{\"pressure\":110,\"temperature\":71,\"vibration\":1.6,\"cycle_count\":4200,\"device_id\":\"UNKNOWN-777\",\"tamper\":1}"
```

**C) Raw numbers**  
```bash
python src/predict.py --pressure 102 --temperature 64 --vibration 1.2 --cycle_count 1500
```

---

## Example output

```
== Prediction ==
Device -> SEN-001
Input  -> pressure=97.174, temperature=61.766, vibration=1.251, cycle_count=686
Risk   -> 0.053 (0=low, 1=high)
Label  -> 0
Note   -> Low risk — keep running.
```

If suspicious:
```
SECURITY NOTE -> unknown device id, tamper flag set
```

---

## Figures

- ROC curve: `reports/figures/roc.png`  
- Confusion matrix: `reports/figures/cm.png`

You can attach these images in an email when you share the repo.

---

## Project structure

```
smc-predictive-maintenance-demo/
├── README.md
├── requirements.txt
├── devices.json
├── data/
│   └── sample.csv
├── src/
│   ├── __init__.py
│   ├── simulate.py
│   ├── train.py
│   └── predict.py
└── reports/
    └── figures/
        ├── roc.png
        └── cm.png
```

---

## Why this is relevant

- Mirrors how device readings could look in a factory (simple **device_id + health flags**).
- Trains a **simple, explainable model** (Logistic Regression) for failure risk.
- Adds a **security-aware note** for unknown/tampered readings (IT/cyber angle).

---

## Tech

Python 3.10+, pandas, scikit-learn, matplotlib, numpy

---

## License

MIT
