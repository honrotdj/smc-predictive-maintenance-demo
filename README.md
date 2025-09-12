# Predictive Maintenance Demo (SMC Project)

**Repo:** `smc-predictive-maintenance-demo`  
**Goal:** Simulate factory sensor data and train a simple machine‑learning model (Logistic Regression) to estimate the risk of machine failure.  
**Why this matters for SMC:** It showcases curiosity and alignment with SMC’s focus on automation, reliability, and smart maintenance — in a *clean, beginner‑friendly* demo.

> This is a **learning/demo** project, not production software.

---

## What this v1 demo will include

- **Data Simulation**: Generate fake sensor readings for a few machines  
  Columns: `timestamp, pressure, temperature, vibration, cycle_count, fault`
- **Modeling**: Train a **Logistic Regression** classifier to predict `fault` (0/1)
- **Evaluation**: Save **one figure** (ROC/AUC or confusion matrix) to `reports/figures/`
- **Prediction**: Print a **risk score** (probability of failure) for new readings
- **Documentation**: Clear README and comments throughout

---

## Project Structure

```
smc-predictive-maintenance-demo/
├── README.md
├── requirements.txt
├── data/
│   └── sample.csv
├── src/
│   ├── __init__.py
│   ├── simulate.py
│   ├── train.py
│   └── predict.py
└── reports/
    └── figures/
```

> `data/sample.csv` is a placeholder now. We’ll generate real samples in **Step 2**.

---

## Prerequisites

- Python **3.10+**
- `pip` (comes with Python)
- (Optional) Git + GitHub for version control

Install dependencies (after you create/activate a virtual environment if you want one):

```bash
pip install -r requirements.txt
```

---

## Quickstart (will work after we implement each step)

1. **Simulate data** (Step 2)  
   ```bash
   python src/simulate.py
   ```
2. **Train model** (Step 3)  
   ```bash
   python src/train.py
   ```
3. **Predict risk** for a new reading (Step 4)  
   ```bash
   python src/predict.py
   ```

Figures (ROC/AUC or confusion matrix) will be saved to: `reports/figures/`

---

## Roadmap

- **v1 (MVP)** — simulate → train → predict → save one figure
- **After MVP (optional)**  
  - Feature importance chart  
  - Small **FastAPI** web dashboard  
  - Simulated packet‑loss mode (wireless sensor realism)  
  - Plain‑English maintenance recommendation based on risk

---

## Talking Points (for SMC recruiters)

- Built a **predictive maintenance** demo that mirrors real factory telemetry (pressure/temperature/vibration) and **failure risk** modeling.
- Clear, well‑documented code — **beginner‑friendly** but uses industry tools: `pandas`, `scikit‑learn`, `matplotlib`, `numpy`.
- Thoughtful roadmap (web dashboard, packet loss, recommendations) to extend toward smart maintenance concepts.
- Clean repo structure and one‑command steps for simulation, training, and prediction.

---

## License

MIT (or choose a license you prefer).
