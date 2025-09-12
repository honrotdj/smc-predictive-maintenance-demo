import argparse
import numpy as np
import pandas as pd

# Simple helper to squash any number into a 0–1 probability
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n_rows=1000, n_machines=5, seed=42):
    """
    Makes fake sensor data for a few machines.
    Each row is one reading at a 1-minute interval.
    """

    rng = np.random.default_rng(seed)

    # Time column — pretend we collected one reading per minute
    end = pd.Timestamp.now().floor('min')
    timestamps = pd.date_range(end=end, periods=n_rows, freq='min')

    # Base signals
    pressure = rng.normal(100, 4.5, n_rows)          # around 100 PSI
    temperature = rng.normal(60, 3.0, n_rows)        # around 60°C
    vibration = rng.normal(1.0, 0.18, n_rows)        # around 1 g

    # Make the data feel more “real” — occasional weird spikes
    drift = np.linspace(0, rng.normal(0, 2.0), n_rows)
    pressure += drift

    hot_idx = rng.choice(n_rows, size=n_rows // 50, replace=False)
    temperature[hot_idx] += rng.normal(8, 2.0, len(hot_idx))

    vib_idx = rng.choice(n_rows, size=n_rows // 40, replace=False)
    vibration[vib_idx] += rng.normal(0.6, 0.15, len(vib_idx))

    # Cycle counters — how many times each machine has run
    machine_ids = rng.integers(0, n_machines, n_rows)
    counters = {m: 0 for m in range(n_machines)}
    cycle_count = np.zeros(n_rows, dtype=int)

    for i, m in enumerate(machine_ids):
        # once in a while, pretend the machine got serviced
        if rng.random() < 0.002:
            counters[m] = 0
        # otherwise keep counting up
        counters[m] += max(1, int(rng.normal(5, 2)))
        counters[m] = min(counters[m], 5000)
        cycle_count[i] = counters[m]

    # Simple fake “risk model” to decide faults
    x = (
        0.08 * (pressure - 108) +
        0.25 * (temperature - 68) +
        1.2  * (vibration - 1.3) +
        0.35 * ((cycle_count - 3500) / 1000.0)
    )
    x += rng.normal(0, 0.6, n_rows)  # add randomness
    fault = (rng.random(n_rows) < sigmoid(x)).astype(int)

    # Put it all together
    df = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "pressure": np.round(pressure, 3),
        "temperature": np.round(temperature, 3),
        "vibration": np.round(vibration, 3),
        "cycle_count": cycle_count,
        "fault": fault
    })

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outfile", type=str, default="data/sample.csv")
    args = parser.parse_args()

    df = generate_data(args.rows, args.machines, args.seed)
    df.to_csv(args.outfile, index=False)

    # Quick sanity check printout
    print(f"Saved {len(df)} rows to {args.outfile}")
    print(f"Fault rate: {df['fault'].mean():.1%}")
    print(df.head())

if __name__ == "__main__":
    main()

