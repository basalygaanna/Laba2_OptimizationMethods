import numpy as np

def validate_solution(x, times, supply, demand, arc_capacity, tol=1e-7):
    x = np.asarray(x, dtype=float)
    base_used = x.sum(axis=1)
    point_received = x.sum(axis=0)
    checks = {
        "nonnegative": bool(np.all(x >= -tol)),
        "base_supply": bool(np.all(base_used <= np.asarray(supply) + tol)),
        "incident_demand": bool(np.all(np.abs(point_received - np.asarray(demand)) <= tol)),
        "arc_capacity": bool(np.all(x <= np.asarray(arc_capacity) + tol))
    }
    objective = float(np.sum(x * np.asarray(times)))
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "base_used": base_used.tolist(),
        "point_received": point_received.tolist(),
        "objective_team_minutes": objective
    }
