import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    cfg = json.loads((ROOT / "data" / "variant18.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"])
    nb, ni = len(cfg["bases"]), len(cfg["incident_points"])
    times = rng.integers(20, 91, size=(nb, ni))
    supply = rng.integers(4, 8, size=nb)
    demand = rng.integers(2, 5, size=ni)
    capacities = rng.integers(1, 5, size=(nb, ni))
    # Keep all generated data in the data file after first generation.
    generated = {
        "travel_time_minutes": times.tolist(),
        "base_supply_teams": supply.tolist(),
        "incident_demand_teams": demand.tolist(),
        "arc_capacity_teams": capacities.tolist()
    }
    data_path = ROOT / "data" / "generated_data.json"
    if data_path.exists():
        generated = json.loads(data_path.read_text(encoding="utf-8"))
    else:
        data_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg, generated
