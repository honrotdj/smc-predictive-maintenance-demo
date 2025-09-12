import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# tiny helper for turning a score into a 0–1 probability
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pick_device(rng, manifest):
    # most rows use a legit device id; a few are unknown to mimic bad config
    if rng.random() < 0.015:
        return f"UNKNOWN-{rng.integers(100, 999)}"
    ids = [d["device_id"] for d in manifest["devices"]]
    return rng.choice(ids)

def generate_data(n_rows=1000, n_machines=5, seed=42, manifest_path="devices.json"):
    rng = np.random.default_rng(seed)

    # load a simple device manifest (pretend "EX600" master + a few sensors)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    port_class = manifest.get("port_class", "A")

    # timestamps: one reading per minute (newest last)
    end = pd.Timestamp.now().floor("min")
    timestamps = pd.date_range(end=end, periods=n_rows, freq="min")

    # base signals (nothing fancy)
    pressure = rng.normal(100, 4.5, n_rows)       # ~100 psi
    temperature = rng.normal(60, 3.0, n_rows)     # ~60 °C
    vibration = rng.normal(1.0, 0.18, n_rows)     # ~1 g

    # add a slow pressure drift so it doesn't look flat
    pressure += np.linspace(0, rng.normal(0, 2.0), n_rows)

    # occasional temperature and vibration spikes
    hot_idx = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    temperature[hot_idx] += rng.normal(8, 2.0, len(hot_idx))

    vib_idx = rng.choice(n_rows, size=max(1, n_rows // 40), replace=False)
    vibration[vib_idx] += rng.normal(0.6, 0.15, len(vib_idx))

    # crude "usage since maintenance" counter per machine
    machine_ids = rng.integers(0, n_machines, n_rows)
    counters = {m: 0 for m in range(n_machines)}
    cycle_count = np.zeros(n_rows, dtype=int)
    for i, m in enumerate(machine_ids):
        if rng.random() < 0.002:      # pretend maintenance happened
            counters[int(m)] = 0
        step = max(1, int(rng.normal(5, 2)))
        counters[int(m)] = min(counters[int(m)] + step, 5000)
        cycle_count[i] = counters[int(m)]

    # simple device identity + health bits
    device_id = np.array([pick_device(rng, manifest) for _ in range(n_rows)])
    cable_loss_flag = (rng.random(n_rows) < 0.01).astype(int)  # rare cable issue
    tamper = (rng.random(n_rows) < 0.008).astype(int)          # rare “bad packet”
    sensor_health = np.where(cable_loss_flag == 1, "degraded", "ok")

    # label: higher pressure/temp/vibration/usage => more likely to fault
    x = (
        0.08 * (pressure - 108) +
        0.25 * (temperature - 68) +
        1.2  * (vibration - 1.3) +
        0.35 * ((cycle_count - 3500) / 1000.0)
    )
    x += rng.normal(0, 0.6, n_rows)   # noise so it's not perfectly separable
    fault = (rng.random(n_rows) < sigmoid(x)).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "device_id": device_id,
        "port_class": port_class,
        "pressure": np.round(pressure, 3),
        "temperature": np.round(temperature, 3),
        "vibration": np.round(vibration, 3),
        "cycle_count": cycle_count,
        "sensor_health": sensor_health,
        "cable_loss_flag": cable_loss_flag,
        "tamper": tamper,
        "fault": fault
    })
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--machines", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outfile", type=str, default="data/sample.csv")
    p.add_argument("--manifest", type=str, default="devices.json")
    args = p.parse_args()

    df = generate_data(args.rows, args.machines, args.seed, args.manifest)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved {len(df)} rows to {out}")
    print(f"Fault rate: {df['fault'].mean():.1%}")
    print(df.head())

if __name__ == "__main__":
    main()


